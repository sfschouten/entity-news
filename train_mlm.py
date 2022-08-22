import argparse
import pprint
from functools import partial

from datasets import load_metric
from transformers import AutoTokenizer, TrainingArguments, \
    Trainer, EarlyStoppingCallback

from data_collator import DataCollatorForLanguageModeling
from modeling_versatile import MaskedLM
from train_news_clf import news_clf_dataset
from utils import create_run_folder_and_config_dict, create_or_load_versatile_model, train_versatile

import wandb


def news_data(config, tokenizer):
    dataset = news_clf_dataset(config, tokenizer)
    dataset.remove_columns(['labels'])
    return dataset


def compute_mlm_metrics(config, metric):
    return {'': 0}


def train_mlm(cli_config, dataset_fn=news_data):
    wandb.init(project='entity-news', tags=['MLM'])

    head_id = cli_config['head_id']

    tokenizer = AutoTokenizer.from_pretrained(cli_config['model'])
    tokenized_dataset = dataset_fn(cli_config, tokenizer)

    # load model
    heads = {
        head_id: (1., MaskedLM),
    }
    model = create_or_load_versatile_model(cli_config, {}, heads)

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
        label_names=['mlm_labels'],
        evaluation_strategy=cli_config['eval_strategy'],
        save_strategy=cli_config['eval_strategy'],
        eval_steps=cli_config['eval_frequency'],
        warmup_steps=cli_config['warmup_steps'],
        report_to=cli_config['report_to'],
        save_total_limit=5,
        prediction_loss_only=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'].with_format('torch'),
        eval_dataset=tokenized_dataset['validation'].with_format('torch'),
        compute_metrics=partial(compute_mlm_metrics, cli_config, load_metric('accuracy')),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cli_config['early_stopping_patience'])
        ],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            label_name='mlm_labels'
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
        tokenized_dataset[key],
        metric_key_prefix=key,
        ignore_keys=hidden_state_keys,
    )
    pprint.pprint(result)


def train_news_clf_argparse(parser: argparse.ArgumentParser):
    parser.add_argument('--report_to', default=None, type=str)

    parser.add_argument('--nc_data_folder', default="../data/minimal")

    parser.add_argument('--mwep_home', default='../mwep')
    parser.add_argument('--runs_folder', default='runs')
    parser.add_argument('--run_name', default=None)

    parser.add_argument('--model', default="distilbert-base-cased")
    parser.add_argument('--probing', action='store_true')
    parser.add_argument('--head_id', default='mlm-0', type=str)

    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--continue', action='store_true')
    parser.add_argument('--eval_only', action='store_true')

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

    train_mlm(create_run_folder_and_config_dict(args))
