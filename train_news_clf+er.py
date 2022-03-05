import argparse
import pprint

import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer, AutoModel, TrainingArguments, EarlyStoppingCallback

from modeling_multi_task import create_multitask_class, SequenceClassification, TokenClassification
from multitask_trainer import MultitaskTrainer
from utils import create_run_folder_and_config_dict

from train_entity_linker import entity_recognition_dataset
from train_news_clf import news_clf_dataset


def train_news_clf(config):

    tokenizer = AutoTokenizer.from_pretrained(config['model'])

    # collect datasets and corresponding tasks
    nc_dataset = news_clf_dataset(config, tokenizer).rename_column('labels', 'nc_labels')
    datasets = {
        "nc": nc_dataset['train'],
        "er": entity_recognition_dataset(config, tokenizer).rename_column('labels', 'er_labels'),
    }
    tasks = {
        "nc": SequenceClassification,
        "er": TokenClassification,
    }

    # model and configuration
    base_model = AutoModel.from_pretrained(config['model'])
    base_model.config.update({
        'nc_num_labels': 4,
        'er_num_labels': 3
    })
    cls = create_multitask_class(type(base_model))
    model = cls(base_model, tasks.items())

    # metric
    acc_metric = load_metric('accuracy')
    #seq_metric = load_metric('seqeval')

    def compute_metrics(eval_pred):
        (nc_logits, er_logits), (nc_labels, er_labels) = eval_pred
        metrics = {}
        if not (er_labels == -1).all():
            preds = np.argmax(er_logits, axis=-1)
            #er_result = seq_metric.compute(predictions=preds, references=er_labels)
            #metrics.update(er_result)
        if not (nc_labels == -1).all():
            preds = np.argmax(nc_logits, axis=-1)
            nc_acc = acc_metric.compute(predictions=preds, references=nc_labels)
            metrics.update(nc_acc)
        return metrics

    # training
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
        eval_steps=5,#00,
        remove_unused_columns=False,
        #label_names=['nc_labels'],
        label_names=[f"{key}_labels" for key in datasets.keys()]
    )
    trainer = MultitaskTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets,
        eval_dataset=nc_dataset['validation'].shard(num_shards=100, index=0),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config['early_stopping_patience'])
        ]
    )
    if config['continue']:
        trainer.train(resume_from_checkpoint=config['checkpoint'])
    else:
        trainer.train()

    result = trainer.evaluate(nc_dataset['test'])
    pprint.pprint(result)


if __name__ == "__main__":
    # parse cmdline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nc_data_folder', default="../data/minimal/bin")
    parser.add_argument('--mwep_home', default='../mwep')

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
    train_news_clf(
        create_run_folder_and_config_dict(args)
    )
