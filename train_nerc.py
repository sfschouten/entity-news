import argparse
from functools import partial

import numpy as np
import wandb

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, \
    TrainingArguments, Trainer, EarlyStoppingCallback, \
    BatchEncoding

from data_collator import DataCollatorForTokenClassification
from modeling_versatile import TokenClassification
from utils import create_run_folder_and_config_dict, create_or_load_versatile_model, train_versatile

from sklearn import metrics


def conll2003_dataset(config, tokenizer):
    def labels(example, batch_encoding: BatchEncoding):
        words = batch_encoding.words()  # maps tokens to word indices
        labels_ = [example['ner_tags'][w] if w is not None else 0 for w in words]
        # only label first token of each word
        labels_ = [label if cw != pw else -100
                   for label, cw, pw in zip(labels_[1:], words[1:], words[0:-1])]
        batch_encoding['labels'] = labels_
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


def compute_nerc_metrics(cli_config, seq_metric, eval_pred):
    TAGS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    logits, labels = eval_pred
    np.savetxt(cli_config['run_path'] + '/logits.txt', logits)

    preds = np.argmax(logits, axis=-1)

    cf_labels = labels[labels != -100]
    cf_preds = preds[labels != -100]

    print(metrics.confusion_matrix(cf_labels, cf_preds))
    wandb.log({"nc_conf_mat": wandb.plot.confusion_matrix(
        y_true=cf_labels, preds=cf_preds, class_names=TAGS)})

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


def train_entity_recognition(cli_config, dataset_fn):
    wandb.init(project='entity-news', tags=['NERC'])

    tokenizer = AutoTokenizer.from_pretrained(cli_config['model'])

    # dataset
    dataset = dataset_fn(cli_config, tokenizer)
    dataset = dataset.rename_column('labels', 'nerc_labels')

    # load model
    head_id = cli_config['head_id']
    heads = {head_id: (1., TokenClassification)}
    model = create_or_load_versatile_model(
        cli_config,
        {
            f'{head_id}_num_labels': 9,
            f'{head_id}_attach_layer': cli_config['head_attach_layer'],
        },
        heads
    )

    training_args = TrainingArguments(
        cli_config['run_path'],
        fp16=True,
        num_train_epochs=cli_config['max_nr_epochs'],
        per_device_train_batch_size=cli_config['batch_size_train'],
        per_device_eval_batch_size=cli_config['batch_size_eval'],
        gradient_accumulation_steps=cli_config['gradient_acc_steps'],
        load_best_model_at_end=True,
        remove_unused_columns=False,
        label_names=["nerc_labels"],
        metric_for_best_model=cli_config['eval_metric'],
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
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=partial(compute_nerc_metrics, cli_config, load_metric('seqeval')),
        data_collator=DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            label_name='nerc_labels'
        ),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cli_config['early_stopping_patience'])
        ]
    )

    if cli_config['probing']:
        for param in model.base_model.parameters():
            param.requires_grad = False

    # ignore hidden states in output, prevent OOM during eval
    hidden_state_keys = [f'{key}_hidden_states' for key in heads.keys()]

    train_versatile(cli_config, trainer, eval_ignore=hidden_state_keys)

    key = 'test' if cli_config['eval_on_test'] else 'validation'
    result = trainer.evaluate(
        dataset[key],
        metric_key_prefix=f'{key}/conll',
        ignore_keys=hidden_state_keys
    )
    print(result)


def train_nerc_argparse(parser: argparse.ArgumentParser):
    parser.add_argument('--report_to', default=None, type=str)

    parser.add_argument('--runs_folder', default='runs')
    parser.add_argument('--run_name', default=None)

    parser.add_argument('--model', default="distilbert-base-cased")
    parser.add_argument('--probing', action='store_true')
    parser.add_argument('--head_attach_layer', default=-1, type=int)
    parser.add_argument('--head_id', default='nerc-0', type=str)

    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--continue', action='store_true')
    parser.add_argument('--eval_only', action='store_true')

    parser.add_argument('--eval_strategy', default='steps', type=str)
    parser.add_argument('--eval_frequency', default=500, type=int)
    parser.add_argument('--eval_metric', default='overall_f1', type=str)
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
    parser = train_nerc_argparse(parser)
    args = parser.parse_args()
    train_entity_recognition(
        create_run_folder_and_config_dict(args), conll2003_dataset
    )
