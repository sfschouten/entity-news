import argparse
import math
from collections import Counter
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import wandb
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from tqdm import tqdm

from train_news_clf import train_news_clf, train_news_clf_argparse
from experiment_visualize_entity_tokens import news_clf_dataset_with_ots_ner
from utils import create_run_folder_and_config_dict
from utils_mentions import Mention, samples_to_mentions, mentions_by_sample, calc_mention_topic_dist

VERSIONS = ['mask', 'substitute']
SUB_VARIANTS = ['random_tokens', 'random_mention', 'type_invariant', 'frequency', 'topic_shift']


def entity_poor_news_clf_dataset(cli_config, tokenizer):
    dataset = news_clf_dataset_with_ots_ner(cli_config, tokenizer)
    nr_topics = len(dataset['train'].features['labels'].names)

    def mask_entities(samples):
        input_ids = samples['input_ids']
        ner_preds = samples['ner']

        for i, (input_ids, ner_preds) in enumerate(zip(input_ids, ner_preds)):
            samples['input_ids'][i] = [
                input_id if ner == 'O' else tokenizer.mask_token_id
                for input_id, ner in zip(input_ids, ner_preds)
            ]

        return samples

    def substitute_entities(samples):
        nonlocal tokenizer
        entity_mentions = samples_to_mentions(samples)
        entity_mention_counter = Counter(entity_mentions)

        # calculate distribution over topics
        topics = samples['labels']
        mention_topic_dist = calc_mention_topic_dist(entity_mentions, topics, nr_topics)

        if cli_config['substitute_variant'] == 'random_tokens':
            vocab = list(tokenizer.vocab.values())
            unique = list(set(entity_mentions))

            def sample_fn(mention: Mention):
                entity = random.sample(unique, 1)[0]
                return random.sample(vocab, len(entity.token_ids)),
        elif cli_config['substitute_variant'] == 'random_mention':
            unique = list(set(entity_mentions))

            def sample_fn(mention: Mention):
                substitute = random.sample(unique, 1)[0]
                original_freq = entity_mention_counter[mention]
                substitute_freq = entity_mention_counter[substitute]
                original_dist = mention_topic_dist[mention]
                substitute_dist = mention_topic_dist[substitute]
                topic_shift = sum(rel_entr(original_dist, substitute_dist))
                return substitute.token_ids, {
                    'frequencies': (original_freq, substitute_freq),
                    'topic_shift': topic_shift
                }
        elif cli_config['substitute_variant'] == 'type_invariant':
            by_type = {}
            for mention in entity_mentions:
                same_type = by_type.get(mention.type, [])
                same_type.append(mention)
                by_type[mention.type] = same_type

            def sample_fn(mention):
                same_type = by_type[mention.type]
                entity = random.sample(same_type, 1)[0]
                return entity.token_ids,
        elif cli_config['substitute_variant'] == 'frequency':
            nr_most_frequent = cli_config['nr_most_frequent']
            most_frequent = entity_mention_counter.most_common(nr_most_frequent)

            # print most frequent
            print(sorted([
                tokenizer.convert_ids_to_tokens(mention.token_ids)
                for mention, _ in most_frequent
            ]))

            def sample_fn(_):
                entity, _ = random.sample(most_frequent, 1)[0]
                return entity.token_ids,
        elif cli_config['substitute_variant'] == 'topic_shift':

            return {}

            # we are looking for mentions that occur in very particular topics
            # ...
            mention_sort_idxs = [None] * nr_topics
            for t in range(nr_topics):
                mention_sort_idxs[t] = sorted(
                    mention_topic_dist.keys(),
                    key=lambda m: mention_topic_dist[m][t],
                    reverse=True
                )

            def invert_probability_vector(p):
                indxs, values = zip(*sorted(enumerate(p), key=lambda p_: p_[1]))
                _, inv = zip(*sorted(zip(indxs, reversed(values)), key=lambda p_: p_[0]))
                return inv

            C = 50  # nr of candidates
            for i, (m, p) in enumerate(mention_topic_dist.items()):
                candidates = []
                for t in range(nr_topics):
                    p_inv = invert_probability_vector(p)
                    M_d = int(p_inv[t] * C)
                    candidates.extend(mention_sort_idxs[t][:M_d])

            mention_kls = {}
            for mention_a, dist_a in tqdm(mention_topic_dist.items()):
                divergences = {}
                for mention_b, dist_b in mention_topic_dist.items():
                    divergences[mention_b] = jensenshannon(dist_a, dist_b)
                mention_kls[mention_a] = divergences

            def sample_fn(mention: Mention):
                highest_shift = sorted(mention_kls[mention].items(), key=lambda x: x[1])[
                                :-50]  # TODO make configurable
                entity, _ = random.sample(highest_shift, 1)[0]
                return entity.token_ids,
        else:
            def sample_fn(_):
                raise NotImplementedError()

        input_ids = samples['input_ids']
        metadata = {}
        entity_mentions_by_sample = mentions_by_sample(entity_mentions, len(input_ids))

        # make the substitutions
        for i, sample_mentions in enumerate(entity_mentions_by_sample):
            # create mapping from entities to substitutes
            substitutes = {mention: sample_fn(mention) for mention in set(sample_mentions)}
            sample_metadata = {}

            # go through mentions in this sample in reverse order
            for mention in sorted(sample_mentions, key=lambda m: m.token_idxs[0], reverse=True):
                start_idx = mention.token_idxs[0]

                # delete tokens
                for idx in reversed(mention.token_idxs):
                    # print(f"deleting {mention.sample_index}:{idx}")
                    if idx >= len(input_ids[mention.sample_index]):
                        print(f"WARNING: idx ({idx}) was outside of array bounds "
                              f"(len={len(input_ids[mention.sample_index])}).")
                        continue
                    del input_ids[mention.sample_index][idx]

                # insert substitute
                for t in reversed(substitutes[mention][0]):
                    input_ids[mention.sample_index].insert(start_idx, t)
                    # print(f"inserting {subst_token} at {mention.sample_index}:{start_idx}")

                if len(substitutes[mention]) > 1:
                    _metadata = substitutes[mention][1]
                    for key, value in _metadata.items():
                        values = sample_metadata.get(key, [])
                        values.append(value)
                        sample_metadata[key] = values

            for key, value in sample_metadata.items():
                values = metadata.get(key, [[]]*len(entity_mentions_by_sample))
                values[i] = value
                metadata[key] = values

                # print('----------')
            # print(f'===={i}=====')

        # trim back to max of 512 tokens
        for i, s_input_ids in enumerate(input_ids):
            if len(s_input_ids) > 512:  # TODO retrieve 512 number from config somewhere
                input_ids[i] = s_input_ids[:512]

        # create new attention mask
        mask = [[1] * len(s_input_ids) for i, s_input_ids in enumerate(input_ids)]

        return {'input_ids': input_ids, 'attention_mask': mask} | metadata

    if cli_config['experiment_version'] == 'mask':
        fn = mask_entities
    elif cli_config['experiment_version'] == 'substitute':
        fn = substitute_entities
    else:
        raise ValueError(f"Invalid version of experiment: {cli_config['experiment_version']}")

    dataset = dataset.map(fn, batched=True, batch_size=None).remove_columns(
        ['ner', 'incident.wdt_id']
    )
    return dataset


def output(df):
    if 'topic_shift' in df.columns:
        # calculate average topic_shift per sample
        df['topic_shift_avg'] = df['topic_shift'].apply(np.mean)
        df = df.drop(columns=['topic_shift'])

        print(df['metric'].corr(df['topic_shift_avg']))

    if 'frequencies' in df.columns:
        # calculate average difference in log-frequency
        df['log_frequency_shift'] = df['frequencies'].apply(
            lambda sample: list(map(lambda freqs: math.log(freqs[0]) + math.log(freqs[1]), sample))
        )
        df['log_frequency_shift_avg'] = df['log_frequency_shift'].apply(np.mean)
        df = df.drop(columns=['frequencies', 'log_frequency_shift'])

        print(df['metric'].corr(df['log_frequency_shift_avg']))


def analysis(cli_config, trainer, model, eval_dataset):
    # convert dataset to pandas dataframe
    df = pd.DataFrame(eval_dataset)
    print(df.columns)

    eval_dataset.set_format("torch")
    eval_dataset = eval_dataset.remove_columns(
        [c for c in eval_dataset.column_names if c not in ['nc_labels', 'input_ids', 'attention_mask']]
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cli_config['batch_size_eval'],
        collate_fn=trainer.data_collator
    )
    correctness = []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct = predictions == batch['labels']
        correctness.extend(list(correct))

    df['metric'] = correctness

    output(df)


def entitypoor_argparse(parser: argparse.ArgumentParser):
    parser.add_argument('--experiment_version', choices=VERSIONS)

    # if experiment_version = substitute
    parser.add_argument('--substitute_variant', choices=SUB_VARIANTS)

    # if substitute_variant = frequency
    parser.add_argument('--nr_most_frequent', default=100, type=int)

    # special option to run all experiment variants
    parser.add_argument('--run_all_variants', action='store_true')

    parser.add_argument('--do_analysis', action='store_true')

    return parser


if __name__ == "__main__":
    # parse cmdline arguments
    parser = argparse.ArgumentParser()
    parser = train_news_clf_argparse(parser)
    parser = entitypoor_argparse(parser)
    args = parser.parse_args()

    if args.run_all_variants:
        all_variants = [
            {'experiment_version': 'substitute', 'substitute_variant': x} for x in SUB_VARIANTS
        ] + [{'experiment_version': 'mask'}]

        for dict in all_variants:
            for key, value in dict.items():
                setattr(args, key, value)
            config = create_run_folder_and_config_dict(args)
            run = wandb.init(reinit=True, tags=['EntityPoor'])
            train_news_clf(config, entity_poor_news_clf_dataset)
            run.finish()
    else:
        config = create_run_folder_and_config_dict(args)
        result, trainer, model, eval_dataset = train_news_clf(config, entity_poor_news_clf_dataset)

        if config['do_analysis']:
            analysis(config, trainer, model, eval_dataset)
