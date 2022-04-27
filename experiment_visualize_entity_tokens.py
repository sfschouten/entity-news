import argparse

import pandas
import torch
from datasets import load_dataset
from numpy.typing import ArrayLike

import dataset_mwep
import wandb
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, TrainingArguments, DataCollatorWithPadding, pipeline

from modeling_versatile import SequenceClassification, TokenClassification
from train_ner import B
from train_news_clf import news_clf_dataset
from trainer import Trainer
from utils import create_or_load_versatile_model, create_run_folder_and_config_dict

import plotly.express as px
import numpy as np
from scipy.special import softmax


def news_clf_dataset(config, tokenizer):
    # dataset processing/loading
    dataset = load_dataset(
        dataset_mwep.__file__,
        data_dir=config['nc_data_folder'],
        mwep_path=config['mwep_home'],
    )

    dataset = dataset.flatten().remove_columns(
        [
            'uri',
            'incident.extra_info.sem:hasPlace',
            'incident.extra_info.sem:hasTimeStamp',
        ]
    ).rename_column('incident.incident_type', 'labels')

    # tokenize
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples['content'], padding=False, truncation=True),
        batched=True
    )

    pipe = pipeline(
        tokenizer=tokenizer,
        model="elastic/distilbert-base-cased-finetuned-conll03-english",
        device=0
    )

    def enrich_fn(examples):
        results = pipe(examples['content'], batch_size=config['batch_size_eval'])
        ner = []
        for input_ids, result in zip(examples['input_ids'], results):
            entities = ['O'] * len(input_ids)
            for entity in result:
                if entity['index'] >= len(entities):
                    continue
                entities[entity['index']] = entity['entity']
            ner.append(entities)
        examples['ner'] = ner
        return examples
    enriched_dataset = tokenized_dataset.map(enrich_fn, batched=True, batch_size=2**10)

    return enriched_dataset.remove_columns(['content'])


def create_tsne(cli_config):
    # load dataset
    tokenizer = AutoTokenizer.from_pretrained(cli_config['model'])
    tokenized_dataset = news_clf_dataset(cli_config, tokenizer).rename_column('labels', 'nc_labels')
    nc_class_names = tokenized_dataset['train'].features['nc_labels'].names
    nc_incident_names = tokenized_dataset['train'].features['incident.wdt_id'].names

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
    entity_topics: ArrayLike = None
    entity_samples: ArrayLike = None
    entity_incidents: ArrayLike = None
    entity_tokens: ArrayLike = None
    entity_ots_preds: ArrayLike = None
    entity_validity: ArrayLike = None
    entity_confidence: ArrayLike = None

    def _list_of_lists_to_np_array(list_of_lists, max_len=512):
        return np.array([
            r + [0] * (max_len - len(r))
            for r in list_of_lists
        ])

    # dummy 'compute_metrics' function to collect relevant hidden representations
    def _compute_metrics(eval_result):
        nonlocal entity_repr, entity_tokens, entity_topics, entity_samples, entity_incidents, \
            entity_ots_preds, entity_validity, entity_confidence
        preds, labels = eval_result
        _, ner_logits, entity_idx, entity_repr = preds
        entity_idx = ~entity_idx  # undo inversion (see `custom_outputs_to_logits`)

        attn_mask = _list_of_lists_to_np_array(tokenized_dataset['test']['attention_mask'])
        mask = attn_mask[entity_idx] != 0

        entity_repr = entity_repr[mask]
        entity_confidence = softmax(ner_logits, axis=-1).max(axis=-1)[entity_idx][mask]

        input_ids = _list_of_lists_to_np_array(tokenized_dataset['test']['input_ids'])
        ots_preds = _list_of_lists_to_np_array(tokenized_dataset['test']['ner'])
        incidents = tokenized_dataset['test']['incident.wdt_id']
        entity_tokens = input_ids[entity_idx][mask]
        entity_ots_preds = ots_preds[entity_idx][mask]
        entity_validity = entity_ots_preds != 'O'

        nr_entities = entity_idx.sum(-1)
        entity_topics = np.repeat(labels, nr_entities)[mask]
        entity_samples = np.repeat(torch.arange(len(labels)), nr_entities)[mask]
        entity_incidents = np.repeat(incidents, nr_entities)[mask]

        return {}

    # create Trainer (for eval only though)
    training_args = TrainingArguments(
        cli_config['run_path'],
        fp16=True,
        label_names=['nc_labels'],
        per_device_eval_batch_size=cli_config['batch_size_eval'],
        remove_unused_columns=False,
        eval_accumulation_steps=16,  # TODO make CLI option
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=_compute_metrics,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
        ),
    )
    hidden_state_keys = [f'{key}_hidden_states' for key in heads.keys()]

    I = 1

    def custom_outputs_to_logits(outputs, ignore_keys=()):
        hidden_states = outputs[hidden_state_keys[0]]
        token_representations = hidden_states[cli_config['ner_attach_layer']]
        device = token_representations.device
        N, L, D = token_representations.shape
        M = I * N

        ner_logits = outputs['ner-0_logits']
        ner_confs, ner_preds = ner_logits.max(-1)
        entity_i = ner_preds == B

        # select entities model is most confident about
        ner_confs[~entity_i] = -1.
        _most_conf = ner_confs.view(N*L).argsort(descending=True)[:M]
        false = torch.tensor([False], device=device).expand((N*L))
        entity_j = false.scatter(0, _most_conf, ~false).view(N, L)

        assert entity_j.sum() == M
        assert entity_j.sum() < entity_i.sum()

        # store entity_j inverted, to avoid problems with padding (default Bool pad is True)
        outputs['entity_i'] = ~entity_j
        outputs['entity_repr'] = token_representations[entity_j].reshape(N, I, D)

        return tuple(v for k, v in outputs.items() if k not in ignore_keys)

    trainer.output_to_logits_fn = custom_outputs_to_logits
    trainer.evaluate(
        tokenized_dataset['test'].remove_columns(['incident.wdt_id', 'ner']),
        metric_key_prefix='test',
        ignore_keys=hidden_state_keys
    )

    # create TSNE plot from representations
    tsne = TSNE(
        perplexity=30.0,
        metric='cosine',
        square_distances=True,
    ).fit_transform(entity_repr.squeeze())

    df = pandas.DataFrame()
    df['z1'] = tsne[:, 0]
    df['z2'] = tsne[:, 1]
    df['sample'] = entity_samples
    df['incident'] = [nc_incident_names[i] for i in entity_incidents.astype(int)]
    df['class'] = [nc_class_names[i] for i in entity_topics.astype(int)]
    df['token'] = tokenizer.convert_ids_to_tokens(entity_tokens)
    df['ots_pred'] = entity_ots_preds
    df['conf'] = entity_confidence
    df['correct'] = entity_validity

    # color by incident
    fig = px.scatter(
        df, x='z1', y='z2',
        color='incident',
        symbol='class',
        hover_data=['sample', 'token', 'conf', 'ots_pred']
    )
    wandb.log({'tsne_incident': fig})

    # color by entity_type
    fig = px.scatter(
        df, x='z1', y='z2',
        color='ots_pred',
        hover_data=['sample', 'token', 'incident', 'conf']
    )
    wandb.log({'tsne_ots_pred': fig})

    # color by
    fig = px.scatter(
        df, x='z1', y='z2',
        color='correct',
        hover_data=['sample', 'token', 'incident', 'conf', 'ots_pred']
    )
    wandb.log({'tsne_correct': fig})

    # color by topic
    fig = px.scatter(df, x='z1', y='z2', color='class',
                     hover_data=['sample', 'token', 'incident', 'conf', 'ots_pred'])
    wandb.log({'tsne_class': fig})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--nc_data_folder', default="../data/minimal")
    #parser.add_argument('--nc_data_folder', default="../data/medium_plus")

    parser.add_argument('--mwep_home', default='../mwep')
    parser.add_argument('--runs_folder', default='runs')
    parser.add_argument('--run_name', default=None)

    parser.add_argument('--model', default="distilbert-base-cased")
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--ner_head_id', default='ner-0', type=str)
    parser.add_argument('--nc_head_id', default='nc-0', type=str)

    parser.add_argument('--ner_attach_layer', default=2, type=int)

    parser.add_argument('--batch_size_eval', default=32, type=int)

    args = parser.parse_args()
    create_tsne(
        create_run_folder_and_config_dict(args)
    )
