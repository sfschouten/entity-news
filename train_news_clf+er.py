import argparse
import pprint

import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer, AutoModel, TrainingArguments, EarlyStoppingCallback, \
    DataCollatorWithPadding, AutoConfig

from transformers.models.auto.auto_factory import _get_model_class

from data_collator import DataCollatorForTokenClassification

from modeling_multi_task import create_multitask_class, SequenceClassification, TokenClassification
from multitask_trainer import MultitaskTrainer, EvenMTDL
from utils import create_run_folder_and_config_dict

from train_entity_recognition import kilt_for_er_dataset, compute_er_metrics
from train_news_clf import news_clf_dataset

from sklearn import metrics

import wandb


def train_news_clf(config):
    tokenizer = AutoTokenizer.from_pretrained(config['model'])

    # collect datasets and corresponding tasks
    nc_dataset = news_clf_dataset(config, tokenizer)
    nc_class_names = nc_dataset['train'].features['labels'].names
    nc_dataset = nc_dataset.rename_column('labels', 'nc_labels')
    datasets = {
        "nc": nc_dataset['train'],
        "er": kilt_for_er_dataset(config, tokenizer).rename_column('labels', 'er_labels'),
    }
    tasks = {
        "nc": (1., SequenceClassification),
        "er": (1., TokenClassification),
    }

    base_config = AutoConfig.from_pretrained(config['model'])
    base_model_cls = _get_model_class(base_config, AutoModel._model_mapping)

    # model and configuration
    if config['checkpoint'] is not None:
        cls = create_multitask_class(base_model_cls)
        model = cls.from_pretrained(config['checkpoint'], tasks.items())
    else:
        base_config.update({
            'nc_num_labels': len(nc_class_names),
            'er_num_labels': 3,
            'er_attach_layer': config['er_attach_layer'],
        })
        cls = create_multitask_class(base_model_cls)
        model = cls(base_config, tasks.items())

    # metric
    acc_metric = load_metric('accuracy')
    seq_metric = load_metric('seqeval')

    wandb.init(project='entity-news', tags=['news_clf+er'])

    def compute_metrics(eval_pred):
        (nc_logits, er_logits), (nc_labels, er_labels) = eval_pred
        results = {}
        if not (er_labels == -1).all():
            er_metrics = compute_er_metrics(seq_metric, (er_logits, er_labels))
            results.update(er_metrics)
        if not (nc_labels == -1).all():
            preds = np.argmax(nc_logits, axis=-1)
            nc_acc = acc_metric.compute(predictions=preds, references=nc_labels)
            results.update(nc_acc)
            print(metrics.classification_report(nc_labels, preds, target_names=nc_class_names))
            print(metrics.confusion_matrix(nc_labels, preds))
            wandb.log({"nc_conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                                  y_true=nc_labels, preds=preds,
                                                                  class_names=nc_class_names)})
        return results

    # training
    training_args = TrainingArguments(
        config['run_path'],
        fp16=True,
        num_train_epochs=config['max_nr_epochs'],
        per_device_train_batch_size=config['batch_size_train'],
        per_device_eval_batch_size=config['batch_size_eval'],
        gradient_accumulation_steps=config['gradient_acc_steps'],
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        remove_unused_columns=False,
        label_names=[f"{key}_labels" for key in datasets.keys()],
        evaluation_strategy=config['eval_strategy'],
        save_strategy=config['eval_strategy'],
        eval_steps=config['eval_frequency'],
        warmup_steps=config['warmup_steps'],
        report_to=config['report_to'],
        save_total_limit=5,
    )
    trainer = MultitaskTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets,
        eval_dataset=nc_dataset['validation'],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config['early_stopping_patience'])
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

    hidden_state_keys = [f'{task_key}_hidden_states' for task_key in tasks.keys()]

    train_kwargs = {
        'ignore_keys_for_eval': hidden_state_keys  # prevent OOM during eval
    }
    if config['continue']:
        train_kwargs.update({
            'resume_from_checkpoint': config['checkpoint']
        })

    if not config['eval_only']:
        trainer.train(**train_kwargs)

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
    parser.add_argument('--er_attach_layer', default=-1, type=int)

    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--continue', action='store_true')
    parser.add_argument('--eval_only', action='store_true')

    parser.add_argument('--er_dataset_size', default=None, type=int)

    parser.add_argument('--eval_strategy', default='steps', type=str)
    parser.add_argument('--eval_frequency', default=500, type=int)

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
