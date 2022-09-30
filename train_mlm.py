import argparse
import pprint
from collections import OrderedDict

import wandb

from transformers import AutoTokenizer, TrainingArguments, \
    Trainer, EarlyStoppingCallback, AutoModelForMaskedLM, set_seed

import utils
from data_collator import DataCollatorForLanguageModeling
from modeling_versatile import MaskedLM
from train_news_clf import news_clf_dataset
from utils import create_run_folder_and_config_dict, create_or_load_versatile_model, train_versatile


def train_mlm(cli_config, dataset_fn=news_clf_dataset):
    wandb.init(project='entity-news', tags=['MLM'])

    # set seed to make sure that the masking happens the same way for each model instance.
    set_seed(cli_config['seed'])

    head_id = cli_config['head_id']

    tokenizer = AutoTokenizer.from_pretrained(cli_config['model'])
    tokenized_dataset = dataset_fn(cli_config, tokenizer)
    train_dataset = tokenized_dataset.remove_columns(
        [c for c in tokenized_dataset['train'].column_names if c not in ['input_ids', 'attention_mask']]
    )

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
        train_dataset=train_dataset['train'].with_format('torch'),
        eval_dataset=train_dataset['validation'].with_format('torch'),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cli_config['early_stopping_patience'])
        ],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            label_name='mlm_labels'
        ),
    )

    if cli_config['use_pretrained_mlm_weights']:
        print('Using weights from pretrained MLM model.')
        mlm_model = AutoModelForMaskedLM.from_pretrained(cli_config['model'])
        if cli_config['model'].startswith('distilbert-base'):
            relevant_weights = OrderedDict((k, v) for k, v in mlm_model.state_dict().items() if k.startswith('vocab'))
            print(f'Copying the following entries: {relevant_weights.keys()}')
            model.heads[head_id].load_state_dict(relevant_weights)
        else:
            raise ValueError(f"{cli_config['model']} not yet supported.")

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


def train_mlm_argparse(parser: argparse.ArgumentParser):
    parser = utils.base_train_argparse(parser)

    parser.add_argument('--nc_data_folder', default="../data/minimal")
    parser.add_argument('--mwep_home', default='../mwep')

    parser.add_argument('--probing', action='store_true')
    parser.add_argument('--head_id', default='mlm-0', type=str)
    parser.add_argument('--use_pretrained_mlm_weights', action='store_true')
    parser.add_argument('--seed', default=19930729, type=int)

    parser.add_argument('--eval_strategy', default='steps', type=str)
    parser.add_argument('--eval_frequency', default=500, type=int)
    parser.add_argument('--eval_metric', default='loss', type=str)
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
    parser = train_mlm_argparse(parser)
    args = parser.parse_args()

    train_mlm(create_run_folder_and_config_dict(args))
