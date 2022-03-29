import argparse
import pprint

from datasets import load_metric
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback, \
    DataCollatorWithPadding

from data_collator import DataCollatorForTokenClassification

from modeling_versatile import SequenceClassification, TokenClassification
from multitask_trainer import MultitaskTrainer, EvenMTDL
from utils import create_run_folder_and_config_dict, create_or_load_versatile_model, train_versatile

from train_ner import kilt_for_er_dataset, conll2003_dataset, compute_ner_metrics
from train_news_clf import news_clf_dataset, compute_news_clf_metrics

import wandb


def train_news_clf(cli_config):
    wandb.init(project='entity-news', tags=['NewsCLF+NER'])

    tokenizer = AutoTokenizer.from_pretrained(cli_config['model'])

    # collect datasets
    nc_dataset = news_clf_dataset(cli_config, tokenizer)
    nc_class_names = nc_dataset['train'].features['labels'].names
    nc_dataset = nc_dataset.rename_column('labels', 'nc_labels')

    er_dataset = {
        'kilt': kilt_for_er_dataset,
        'conll': conll2003_dataset
    }[cli_config['ner_dataset']](cli_config, tokenizer).rename_column('labels', 'er_labels')

    datasets = {"nc": nc_dataset['train'], "er": er_dataset['train']}

    # model
    heads = {
        "nc-0": (cli_config['nc_loss_factor'], SequenceClassification),
        "er-0": (cli_config['er_loss_factor'], TokenClassification),
    }
    model = create_or_load_versatile_model(
        cli_config,
        {
            'nc-0_num_labels': len(nc_class_names),
            'er-0_num_labels': 3,
            'er-0_attach_layer': cli_config['ner_attach_layer'],
        },
        heads
    )

    # metric
    acc_metric = load_metric('accuracy')
    seq_metric = load_metric('seqeval')

    def compute_metrics(eval_pred):
        (nc_logits, er_logits), (nc_labels, er_labels) = eval_pred
        results = {}
        if not (er_labels == -1).all():
            er_metrics = compute_ner_metrics(seq_metric, (er_logits, er_labels))
            results.update(er_metrics)
        if not (nc_labels == -1).all():
            nc_metrics = compute_news_clf_metrics(
                acc_metric,
                nc_class_names,
                (nc_logits, nc_labels)
            )
            results.update(nc_metrics)

        return results

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
        label_names=[f"{key}_labels" for key in datasets.keys()],
        evaluation_strategy=cli_config['eval_strategy'],
        save_strategy=cli_config['eval_strategy'],
        eval_steps=cli_config['eval_frequency'],
        warmup_steps=cli_config['warmup_steps'],
        report_to=cli_config['report_to'],
        save_total_limit=5,
    )
    trainer = MultitaskTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets,
        eval_dataset=nc_dataset['validation'],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cli_config['early_stopping_patience'])
        ],
        data_collator={
            'eval': DataCollatorWithPadding(tokenizer=tokenizer),
            'train': {
                'nc': DataCollatorWithPadding(tokenizer=tokenizer),
                'er': DataCollatorForTokenClassification(
                    tokenizer=tokenizer,
                    label_name='er_labels'
                )
            }
        },
        multitask_dataloader_type=EvenMTDL
    )

    # ignore hidden states in output, prevent OOM during eval
    hidden_state_keys = [f'{key}_hidden_states' for key in heads.keys()]

    train_versatile(cli_config, trainer, eval_ignore=hidden_state_keys)

    result = trainer.evaluate(
        nc_dataset['test'],
        metric_key_prefix='test',
        ignore_keys=hidden_state_keys
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
    parser.add_argument('--ner_attach_layer', default=2, type=int)

    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--continue', action='store_true')
    parser.add_argument('--eval_only', action='store_true')

    parser.add_argument('--ner_dataset', choices=['kilt', 'conll'], default='kilt')
    parser.add_argument('--ner_dataset_size', default=None, type=int)

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
    parser.add_argument('--nc_loss_factor', default=1., type=float)
    parser.add_argument('--er_loss_factor', default=1., type=float)

    args = parser.parse_args()
    train_news_clf(
        create_run_folder_and_config_dict(args)
    )
