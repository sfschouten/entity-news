import argparse

import pandas
import torch
from numpy.typing import ArrayLike

import wandb
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, TrainingArguments, DataCollatorWithPadding

from modeling_versatile import SequenceClassification, TokenClassification
from train_ner import B
from train_news_clf import news_clf_dataset
from trainer import Trainer
from utils import create_or_load_versatile_model, create_run_folder_and_config_dict

import plotly.express as px
import numpy as np


def create_tsne(cli_config):
    # load dataset
    tokenizer = AutoTokenizer.from_pretrained(cli_config['model'])
    tokenized_dataset = news_clf_dataset(cli_config, tokenizer).rename_column('labels', 'nc_labels')
    nc_class_names = tokenized_dataset['train'].features['nc_labels'].names

    nc_head_id = cli_config['nc_head_id']
    ner_head_id = cli_config['ner_head_id']

    # model
    heads = {
        nc_head_id: (1., SequenceClassification),
        ner_head_id: (1., TokenClassification),
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

    entity_repr: ArrayLike = None
    class_clustering: ArrayLike = None
    sample_clustering: ArrayLike = None
    entity_tokens: ArrayLike = None

    def _list_of_lists_to_np_array(list_of_lists, max_len=512):
        return np.array([
            r + [0] * (max_len - len(r))  # TODO make less ad hoc
            for r in list_of_lists
        ])

    # dummy 'compute_metrics' function to collect relevant hidden representations
    def _compute_metrics(eval_result):
        nonlocal entity_repr, entity_tokens, class_clustering, sample_clustering
        preds, labels = eval_result
        nc_clf_logits, ner_logits, ner_hidden_states = preds
        ner_preds = ner_logits.argmax(-1)

        attn_mask = _list_of_lists_to_np_array(tokenized_dataset['test']['attention_mask'])
        input_ids = _list_of_lists_to_np_array(tokenized_dataset['test']['input_ids'])
        entity_i = (ner_preds == B) & attn_mask.astype(bool)
        entity_repr = ner_hidden_states[entity_i]
        entity_tokens = input_ids[entity_i]

        nr_entities = entity_i.sum(-1)
        class_clustering = np.repeat(labels, nr_entities)
        sample_clustering = np.repeat(torch.arange(len(labels)), nr_entities)

        return {}

    # create Trainer (for eval only though)
    training_args = TrainingArguments(
        cli_config['run_path'],
        fp16=True,
        label_names=['nc_labels'],
        per_device_eval_batch_size=cli_config['batch_size_eval'],
        remove_unused_columns=False,
        eval_accumulation_steps=16
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        # eval_dataset=tokenized_dataset['validation'],
        compute_metrics=_compute_metrics,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
        ),
    )
    hidden_state_keys = [f'{key}_hidden_states' for key in heads.keys()]

    def custom_outputs_to_logits(outputs, ignore_keys=()):
        hidden_states = outputs[hidden_state_keys[0]]
        token_representations = hidden_states[cli_config['ner_attach_layer']]
        outputs['ner_layer_states'] = token_representations
        return tuple(v for k, v in outputs.items() if k not in ignore_keys)

    trainer.output_to_logits_fn = custom_outputs_to_logits
    trainer.evaluate(
        tokenized_dataset['test'],
        metric_key_prefix='test',
        ignore_keys=hidden_state_keys
    )

    # create TSNE plot from representations
    tsne = TSNE(
        # n_components=3,
        perplexity=30.0,
        metric='cosine',
        square_distances=True,
    ).fit_transform(entity_repr)

    df = pandas.DataFrame()
    df['z1'] = tsne[:, 0]
    df['z2'] = tsne[:, 1]
    # df['z3'] = tsne[:, 2]
    df['sample'] = sample_clustering
    df['class'] = [nc_class_names[i] for i in class_clustering.astype(int)]
    df['token'] = tokenizer.convert_ids_to_tokens(entity_tokens)

    fig = px.scatter(df, x='z1', y='z2', color='sample', hover_data=['class', 'token'])
    wandb.log({'tsne_sample': fig})

    # fig = px.scatter(df, x='z1', y='z2', color='class', hover_data=['sample', 'token'])
    # wandb.log({'tsne_class': fig})

    # fig = px.scatter_3d(
    #     df,
    #     x='z1', y='z2', z='z3',
    #     color='class',
    #     hover_data=['sample', 'token'],
    #     size_max=10,
    # )
    # wandb.log({'tsne_class_3d': fig})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--nc_data_folder', default="../data/minimal")

    parser.add_argument('--mwep_home', default='../mwep')
    parser.add_argument('--runs_folder', default='runs')
    parser.add_argument('--run_name', default=None)

    parser.add_argument('--model', default="distilbert-base-cased")
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--ner_head_id', default='ner-0', type=str)
    parser.add_argument('--nc_head_id', default='nc-0', type=str)

    parser.add_argument('--ner_attach_layer', default=2, type=int)

    parser.add_argument('--batch_size_eval', default=16, type=int)

    args = parser.parse_args()
    create_tsne(
        create_run_folder_and_config_dict(args)
    )
