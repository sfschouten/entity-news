import argparse
from functools import partial

import numpy as np

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer,  AutoModelForTokenClassification, \
    TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorForTokenClassification

from utils import create_run_folder_and_config_dict

I, O, B = 0, 1, 2


def conll2003_dataset(config, tokenizer):
    dataset = load_dataset("conll2003")
    dataset = dataset.map(
        lambda example: tokenizer(
            example['tokens'],
            is_split_into_words=True,
            truncation=True
        ), batched=False,
    ).rename_column('ner_tags', 'labels').remove_columns(['tokens', 'id', 'pos_tags', 'chunk_tags'])
    return dataset


def compute_er_metrics(seq_metric, eval_pred):
    TAGS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    def swap4lbl(ndarray):
        return [
            [TAGS[x.item()] for x in row if x.item() != -100]
            for row in ndarray
        ]

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    labels = swap4lbl(labels)
    preds = swap4lbl(preds)
    preds = [pred[:len(lbl)] for pred, lbl in zip(preds, labels)]
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
        evaluation_strategy="steps",
        num_train_epochs=config['max_nr_epochs'],
        per_device_train_batch_size=config['batch_size_train'],
        per_device_eval_batch_size=config['batch_size_eval'],
        gradient_accumulation_steps=config['gradient_acc_steps'],
        load_best_model_at_end=True,
        metric_for_best_model='overall_f1',
        eval_steps=500,
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

    # hyper-parameters
    parser.add_argument('--max_nr_epochs', default=100, type=int)
    parser.add_argument('--early_stopping_patience', default=5, type=int)
    parser.add_argument('--batch_size_train', default=64, type=int)
    parser.add_argument('--batch_size_eval', default=64, type=int)
    parser.add_argument('--gradient_acc_steps', default=1, type=int)

    args = parser.parse_args()
    train_entity_recognition(
        create_run_folder_and_config_dict(args)
    )
