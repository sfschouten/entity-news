import argparse
import pprint

from datasets import load_metric
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback, \
    DataCollatorWithPadding

import utils
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
    }[cli_config['ner_dataset']](cli_config, tokenizer).rename_column('labels', 'ner_labels')

    datasets = {"nc": nc_dataset['train'], "ner": er_dataset['train']}

    nc_head_id = cli_config['nc_head_id']
    ner_head_id = cli_config['ner_head_id']

    # model
    heads = {
        nc_head_id: (cli_config['nc_loss_factor'], SequenceClassification),
        ner_head_id: (cli_config['ner_loss_factor'], TokenClassification),
    }
    model = create_or_load_versatile_model(
        cli_config,
        {
            f'{nc_head_id}_num_labels': len(nc_class_names),
            f'{ner_head_id}_num_labels': 3,
            f'{ner_head_id}_attach_layer': cli_config['ner_attach_layer'],
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
                'ner': DataCollatorForTokenClassification(
                    tokenizer=tokenizer,
                    label_name='ner_labels'
                )
            }
        },
        multitask_dataloader_type=EvenMTDL
    )

    # ignore hidden states in output, prevent OOM during eval
    hidden_state_keys = [f'{key}_hidden_states' for key in heads.keys()]

    train_versatile(cli_config, trainer, eval_ignore=hidden_state_keys)

    key = 'test' if cli_config['eval_on_test'] else 'validation'
    result = trainer.evaluate(
        nc_dataset[key],
        metric_key_prefix=key,
        ignore_keys=hidden_state_keys
    )
    pprint.pprint(result)


if __name__ == "__main__":
    # parse cmdline arguments
    parser = argparse.ArgumentParser()
    parser = utils.base_train_argparse(parser)

    parser.add_argument('--nc_data_folder', default="../data/minimal")
    parser.add_argument('--mwep_home', default='../mwep')

    parser.add_argument('--ner_attach_layer', default=2, type=int)
    parser.add_argument('--ner_head_id', default='ner-0', type=str)
    parser.add_argument('--nc_head_id', default='nc-0', type=str)

    parser.add_argument('--ner_dataset', choices=['kilt', 'conll'], default='kilt')
    parser.add_argument('--ner_dataset_size', default=None, type=int)

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
    parser.add_argument('--nc_loss_factor', default=1., type=float)
    parser.add_argument('--ner_loss_factor', default=1., type=float)

    args = parser.parse_args()
    train_news_clf(
        create_run_folder_and_config_dict(args)
    )
