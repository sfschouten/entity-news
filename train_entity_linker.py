import argparse

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, BatchEncoding, AutoModelForTokenClassification, \
    TrainingArguments, Trainer, EarlyStoppingCallback

from utils import create_run_folder_and_config_dict

import el_wiki_dataset

I, O, B = 0, 1, 2


def construct_iob_labels(example, batch_encoding: BatchEncoding):
    labels = [O] * len(batch_encoding['input_ids'])
    start_chars = example['mentions']['start_char']
    end_chars = example['mentions']['end_char']
    for start_char, end_char in zip(start_chars, end_chars):
        start_token = batch_encoding.char_to_token(start_char)
        end_token = batch_encoding.char_to_token(end_char-1)
        labels[start_token] = B
        for t in range(start_token + 1, end_token):
            labels[t] = I

    batch_encoding['labels'] = labels
    return batch_encoding


def entity_recognition_dataset(config, tokenizer):
    dataset = load_dataset(
        el_wiki_dataset.__file__,
        split='full',
        # streaming=True,
        # shuffle_base_dataset=True,
        max_samples=1000,
    )

    # tokenize
    tokenized_dataset = dataset.map(
        lambda example: construct_iob_labels(
            example,
            tokenizer(
                example['mentioning_text'],
                padding='max_length',
                truncation=True
            )
        ),
        batched=False
    ).remove_columns(['mentioning_text', 'mentions'])

    return tokenized_dataset


def train_entity_linking(config):
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    tokenized_dataset = entity_recognition_dataset(config, tokenizer)

    # load model
    model = AutoModel.from_pretrained(config['model'])
    #model = AutoModelForTokenClassification.from_pretrained(config['model'], num_labels=3)

    training_args = TrainingArguments(
        config['run_path'],
        fp16=True,
        evaluation_strategy="steps",
        num_train_epochs=config['max_nr_epochs'],
        per_device_train_batch_size=config['batch_size_train'],
        per_device_eval_batch_size=config['batch_size_eval'],
        gradient_accumulation_steps=config['gradient_acc_steps'],
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        eval_steps=500,
        max_steps=1000000,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        #eval_dataset=tokenized_dataset['validation'],
        #compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config['early_stopping_patience'])
        ]
    )
    if config['continue']:
        trainer.train(resume_from_checkpoint=config['checkpoint'])
    else:
        trainer.train()

    #result = trainer.evaluate(tokenized_dataset['test'])


if __name__ == "__main__":
    # parse cmdline arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--runs_folder', default='runs')
    parser.add_argument('--run_name', default=None)

    parser.add_argument('--model', default="distilbert-base-cased")

    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--continue', action='store_true')

    # parser.add_argument('--train_only', action='store_true')
    # parser.add_argument('--eval_only', action='store_true')

    # hyper-parameters
    parser.add_argument('--max_nr_epochs', default=100, type=int)
    parser.add_argument('--early_stopping_patience', default=5, type=int)
    parser.add_argument('--batch_size_train', default=64, type=int)
    parser.add_argument('--batch_size_eval', default=64, type=int)
    parser.add_argument('--gradient_acc_steps', default=1, type=int)
    # parser.add_argument('--learning_rate_base', default=1e-4, type=float)
    # parser.add_argument('--learning_rate_head', default=1e-3, type=float)

    args = parser.parse_args()
    train_entity_linking(
        create_run_folder_and_config_dict(args)
    )
