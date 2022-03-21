import argparse
from functools import partial

import numpy as np

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, \
    TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorForTokenClassification, \
    BatchEncoding

from utils import create_run_folder_and_config_dict


def conll2003_dataset(config, tokenizer):

    def labels(example, batch_encoding: BatchEncoding):
        words = batch_encoding.words()  # maps tokens to word indices
        labels = [example['ner_tags'][w] if w is not None else 0 for w in words]
        # only label first token of each word
        labels = [label if cw != pw else -100
                  for label, cw, pw in zip(labels[1:], words[1:], words[0:-1])]
        batch_encoding['labels'] = labels
        return batch_encoding

    dataset = load_dataset("conll2003")
    dataset = dataset.map(
        lambda example: labels(
            example,
            tokenizer(
                example['tokens'],
                is_split_into_words=True,
                truncation=True
            ),
        ), batched=False,
    ).remove_columns(['tokens', 'id', 'pos_tags', 'chunk_tags', 'ner_tags'])
    return dataset


def compute_er_metrics(seq_metric, eval_pred):
    TAGS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # only take predictions for which we have labels
    # and swap indices for label strings
    preds = [
        [TAGS[pred.item()] for lbl, pred in zip(l_row, p_row) if lbl != -100]
        for l_row, p_row in zip(labels, preds)
    ]
    labels = [
        [TAGS[label.item()] for label in label_row if label != -100]
        for label_row in labels
    ]

    er_result = seq_metric.compute(predictions=preds, references=labels, scheme='IOB2')
    return er_result


def train_entity_recognition(config):
    tokenizer = AutoTokenizer.from_pretrained(config['model'])

    # conll2003 dataset
    conll_dataset = conll2003_dataset(config, tokenizer)

    # load model
    model = AutoModelForTokenClassification.from_pretrained(config['model'], num_labels=9)

    training_args = TrainingArguments(
        config['run_path'],
        fp16=True,
        num_train_epochs=config['max_nr_epochs'],
        per_device_train_batch_size=config['batch_size_train'],
        per_device_eval_batch_size=config['batch_size_eval'],
        gradient_accumulation_steps=config['gradient_acc_steps'],
        load_best_model_at_end=True,
        metric_for_best_model='overall_f1',
        evaluation_strategy=config['eval_strategy'],
        eval_steps=config['eval_frequency'],
        warmup_steps=config['warmup_steps'],
        report_to=config['report_to'],
        save_total_limit=5,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=conll_dataset['train'],
        eval_dataset=conll_dataset['validation'],
        compute_metrics=partial(compute_er_metrics, load_metric('seqeval')),
        data_collator=DataCollatorForTokenClassification(
            tokenizer=tokenizer,
        ),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config['early_stopping_patience'])
        ]
    )

    if config['probing']:
        for param in model.base_model.parameters():
            param.requires_grad = False

    if config['continue']:
        trainer.train(resume_from_checkpoint=config['checkpoint'])
    else:
        trainer.train()

        print("Evaluating on CoNLL2003")
        result = trainer.evaluate(conll_dataset['test'], metric_key_prefix='test_conll')
        print(result)


if __name__ == "__main__":
    # parse cmdline arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--report_to', default=None, type=str)

    parser.add_argument('--runs_folder', default='runs')
    parser.add_argument('--run_name', default=None)

    parser.add_argument('--model', default="distilbert-base-cased")
    parser.add_argument('--probing', action='store_true')

    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--continue', action='store_true')

    parser.add_argument('--eval_strategy', default='steps', type=str)
    parser.add_argument('--eval_frequency', default=500, type=int)

    # hyper-parameters
    parser.add_argument('--max_nr_epochs', default=100, type=int)
    parser.add_argument('--warmup_steps', default=2000, type=int)
    parser.add_argument('--early_stopping_patience', default=5, type=int)
    parser.add_argument('--batch_size_train', default=64, type=int)
    parser.add_argument('--batch_size_eval', default=64, type=int)
    parser.add_argument('--gradient_acc_steps', default=1, type=int)

    args = parser.parse_args()
    train_entity_recognition(
        create_run_folder_and_config_dict(args)
    )
