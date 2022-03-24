import argparse
import pprint

import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, TrainingArguments, \
    Trainer, EarlyStoppingCallback, DataCollatorWithPadding

import mwep_dataset
from modeling_versatile import SequenceClassification
from utils import create_run_folder_and_config_dict, create_or_load_versatile_model, train_versatile

from sklearn import metrics

import wandb


def news_clf_dataset(config, tokenizer):
    # dataset processing/loading
    dataset = load_dataset(
        mwep_dataset.__file__,
        data_dir=config['nc_data_folder'],
        mwep_path=config['mwep_home'],
    )

    dataset = dataset.flatten().remove_columns(
        [
            'uri',
            'incident.extra_info.sem:hasPlace',
            'incident.extra_info.sem:hasTimeStamp',
            'incident.wdt_id'
        ]
    ).rename_column('incident.incident_type', 'labels')

    # tokenize
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples['content'], padding=False, truncation=True),
        batched=True
    ).remove_columns(['content'])

    return tokenized_dataset


def train_news_clf(cli_config):
    wandb.init(project='entity-news', tags=['NewsCLF'])

    tokenizer = AutoTokenizer.from_pretrained(cli_config['model'])
    tokenized_dataset = news_clf_dataset(cli_config, tokenizer).rename_column('labels', 'nc_labels')
    class_names = tokenized_dataset['train'].features['nc_labels'].names

    # load model
    heads = {
        "nc-0": (1., SequenceClassification),
    }
    model = create_or_load_versatile_model(
        cli_config,
        {
            'nc-0_num_labels': len(class_names),
        },
        heads
    )

    acc_metric = load_metric('accuracy')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        print(metrics.classification_report(labels, preds, target_names=class_names))
        print(metrics.confusion_matrix(labels, preds))
        wandb.log({"nc_conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                              y_true=labels, preds=preds,
                                                              class_names=class_names)})
        return acc_metric.compute(predictions=preds, references=labels)

    # training
    training_args = TrainingArguments(
        cli_config['run_path'],
        fp16=True,
        num_train_epochs=cli_config['max_nr_epochs'],
        per_device_train_batch_size=cli_config['batch_size_train'],
        per_device_eval_batch_size=cli_config['batch_size_eval'],
        gradient_accumulation_steps=cli_config['gradient_acc_steps'],
        load_best_model_at_end=True,
        metric_for_best_model=cli_config['eval_metric'],
        remove_unused_columns=False,
        label_names=['nc_labels'],
        evaluation_strategy=cli_config['eval_strategy'],
        save_strategy=cli_config['eval_strategy'],
        eval_steps=cli_config['eval_frequency'],
        warmup_steps=cli_config['warmup_steps'],
        report_to=cli_config['report_to'],
        save_total_limit=5,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cli_config['early_stopping_patience'])
        ],
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
        ),
    )

    if cli_config['probing']:
        for param in model.base_model.parameters():
            param.requires_grad = False

    # ignore hidden states in output, prevent OOM during eval
    hidden_state_keys = [f'{key}_hidden_states' for key in heads.keys()]

    train_versatile(cli_config, trainer, eval_ignore=hidden_state_keys)

    result = trainer.evaluate(
        tokenized_dataset['test'],
        metric_key_prefix='test',
        ignore_keys=hidden_state_keys,
    )
    pprint.pprint(result)


if __name__ == "__main__":
    # parse cmdline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--report_to', default=None, type=str)

    parser.add_argument('--nc_data_folder', default="../data/minimal")

    parser.add_argument('--mwep_home', default='../mwep')
    parser.add_argument('--runs_folder', default='runs')
    parser.add_argument('--run_name', default=None)

    parser.add_argument('--model', default="distilbert-base-cased")
    parser.add_argument('--probing', action='store_true')

    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--continue', action='store_true')
    parser.add_argument('--eval_only', action='store_true')

    parser.add_argument('--eval_strategy', default='steps', type=str)
    parser.add_argument('--eval_frequency', default=500, type=int)
    parser.add_argument('--eval_metric', default='accuracy', type=str)

    # hyper-parameters
    parser.add_argument('--max_nr_epochs', default=100, type=int)
    parser.add_argument('--warmup_steps', default=2000, type=int)
    parser.add_argument('--early_stopping_patience', default=5, type=int)
    parser.add_argument('--batch_size_train', default=64, type=int)
    parser.add_argument('--batch_size_eval', default=64, type=int)
    parser.add_argument('--gradient_acc_steps', default=1, type=int)
    # parser.add_argument('--learning_rate_base', default=1e-4, type=float)
    # parser.add_argument('--learning_rate_head', default=1e-3, type=float)

    args = parser.parse_args()
    train_news_clf(
        create_run_folder_and_config_dict(args)
    )
