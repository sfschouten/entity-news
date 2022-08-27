import argparse
import pprint
from functools import partial

import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, TrainingArguments, \
    Trainer, EarlyStoppingCallback, DataCollatorWithPadding

import dataset_mwep
import utils
from modeling_versatile import SequenceClassification
from utils import create_run_folder_and_config_dict, create_or_load_versatile_model, train_versatile

from sklearn import metrics

import wandb


def news_clf_dataset(config, tokenizer, **kwargs):
    # dataset processing/loading
    dataset = load_dataset(
        dataset_mwep.__file__,
        data_dir=config['nc_data_folder'],
        mwep_path=config['mwep_home'],
        **kwargs
    )

    dataset = dataset.flatten().remove_columns(
        [
            'uri',
            'incident.extra_info.sem:hasPlace',
            'incident.extra_info.sem:hasTimeStamp',
            'incident.wdt_id'
        ]
    ).rename_column('incident.incident_type', 'labels')

    if tokenizer.name_or_path in tokenizer.max_model_input_sizes:
        max_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]
    else:
        max_length = 512
        print("No max length can be inferred, falling back to 512")

    # tokenize
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(
            examples['content'], padding=False, truncation=True, max_length=max_length),
        batched=True
    ).remove_columns(['content'])

    return tokenized_dataset


def compute_news_clf_metrics(cli_config, acc_metric, class_names, eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # store predictions to text file for later analysis
    np.savetxt(cli_config['run_path'] + '/logits.txt', logits)

    print(metrics.classification_report(labels, preds, target_names=class_names))
    clf_report_dct = metrics.classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True
    )
    clf_report = [
        [key] + list(row.values())
        for key, row in clf_report_dct.items()
        if key != 'accuracy'
    ] + [['accuracy', 0., 0., clf_report_dct['accuracy'], 0]]
    wandb.log({"nc_clf_report": wandb.Table(
        columns=['', 'precision', 'recall', 'f1-score', 'support'],
        data=clf_report
    )})

    print(metrics.confusion_matrix(labels, preds))
    wandb.log({"nc_conf_mat": wandb.plot.confusion_matrix(
        y_true=labels, preds=preds, class_names=class_names)})

    return acc_metric.compute(predictions=preds, references=labels)


def train_news_clf(cli_config, dataset_fn=news_clf_dataset):
    wandb.init(project='entity-news', tags=['NewsCLF'])

    tokenizer = AutoTokenizer.from_pretrained(cli_config['model'])
    tokenized_dataset = dataset_fn(cli_config, tokenizer).rename_column('labels', 'nc_labels')

    train_dataset = tokenized_dataset.remove_columns(
        [c for c in tokenized_dataset['train'].column_names if c not in ['nc_labels', 'input_ids', 'attention_mask']]
    )
    class_names = train_dataset['train'].features['nc_labels'].names

    # load model
    head_id = cli_config['head_id']
    heads = {
        head_id: (1., SequenceClassification),
    }
    model = create_or_load_versatile_model(
        cli_config,
        {
            f'{head_id}_num_labels': len(class_names),
        },
        heads
    )

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
        train_dataset=train_dataset['train'],
        eval_dataset=train_dataset['validation'],
        compute_metrics=partial(
            compute_news_clf_metrics, cli_config, load_metric('accuracy'), class_names),
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

    key = 'test' if cli_config['eval_on_test'] else 'validation'
    result = trainer.evaluate(
        train_dataset[key],
        metric_key_prefix=key,
        ignore_keys=hidden_state_keys,
    )
    pprint.pprint(result)
    return result, trainer, model, tokenized_dataset[key]


def train_news_clf_argparse(parser: argparse.ArgumentParser):
    parser = utils.base_train_argparse(parser)

    parser.add_argument('--nc_data_folder', default="../data/minimal")
    parser.add_argument('--mwep_home', default='../mwep')

    parser.add_argument('--probing', action='store_true')
    parser.add_argument('--head_id', default='nc-0', type=str)

    parser.add_argument('--eval_strategy', default='steps', type=str)
    parser.add_argument('--eval_frequency', default=500, type=int)
    parser.add_argument('--eval_metric', default='accuracy', type=str)
    parser.add_argument('--eval_on_test', action='store_true')

    # hyper-parameters
    parser.add_argument('--max_nr_epochs', default=100, type=int)
    parser.add_argument('--warmup_steps', default=2000, type=int)
    parser.add_argument('--early_stopping_patience', default=5, type=int)
    parser.add_argument('--batch_size_train', default=64, type=int)
    parser.add_argument('--batch_size_eval', default=64, type=int)
    parser.add_argument('--gradient_acc_steps', default=1, type=int)
    return parser


if __name__ == "__main__":
    # parse cmdline arguments
    parser = argparse.ArgumentParser()
    parser = train_news_clf_argparse(parser)
    args = parser.parse_args()

    train_news_clf(create_run_folder_and_config_dict(args), news_clf_dataset)
