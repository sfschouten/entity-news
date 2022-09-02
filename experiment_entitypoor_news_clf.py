import argparse
import math
import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

import wandb
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from tqdm import tqdm

from datasets.fingerprint import fingerprint_transform

from train_news_clf import train_news_clf, train_news_clf_argparse
from experiment_visualize_entity_tokens import news_clf_dataset_with_ots_ner
from experiment_nerc_with_analysis import draw_histwithmean
from utils import create_run_folder_and_config_dict
from utils_mentions import Mention, samples_to_mentions, mentions_by_sample, calc_mention_topic_dist

VERSIONS = ['mask', 'substitute']
SUB_VARIANTS = ['random_tokens', 'random_mention', 'type_invariant', 'frequency', 'topic_shift']


def mask_entities(samples):
    input_ids = samples['input_ids']
    ner_preds = samples['ner']

    for i, (input_ids, ner_preds) in enumerate(zip(input_ids, ner_preds)):
        samples['input_ids'][i] = [
            input_id if ner == 'O' else tokenizer.mask_token_id
            for input_id, ner in zip(input_ids, ner_preds)
        ]

    return samples

def substitute_entities(samples, nr_most_frequent=100, variant="random-mention", cli_config={}, tokenizer=None, nr_topics=0):
    entity_mentions = samples_to_mentions(samples)
    entity_mention_counter = Counter(entity_mentions)

    # calculate distribution over topics
    topics = samples['labels']
    mention_topic_dist = calc_mention_topic_dist(entity_mentions, topics, nr_topics)

    if variant == 'random_tokens':
        vocab = list(tokenizer.vocab.values())
        unique = list(set(entity_mentions))

        def sample_fn(mention: Mention):
            entity = random.sample(unique, 1)[0]
            return random.samples(vocab, len(entity.token_ids)),
    elif variant == 'random_mention':
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
    elif variant == 'type_invariant':
        by_type = {}
        for mention in entity_mentions:
            same_type = by_type.get(mention.type, [])
            same_type.append(mention)
            by_type[mention.type] = same_type

        def sample_fn(mention):
            same_type = by_type[mention.type]
            substitute = random.sample(same_type, 1)[0]
            
            original_freq = entity_mention_counter[mention]
            substitute_freq = entity_mention_counter[substitute]
            original_dist = mention_topic_dist[mention]
            substitute_dist = mention_topic_dist[substitute]
            topic_shift = sum(rel_entr(original_dist, substitute_dist))

            return substitute.token_ids, {
                'frequencies': (original_freq, substitute_freq),
                'topic_shift': topic_shift
            }
    elif variant == 'frequency':
        most_frequent = entity_mention_counter.most_common(nr_most_frequent)

        # print most frequent
        #print(sorted([
        #    tokenizer.convert_ids_to_tokens(mention.token_ids)
        #    for mention, _ in most_frequent
        #]))

        def sample_fn(_):
            entity, _ = random.sample(most_frequent, 1)[0]
        return entity.token_ids,
    elif variant == 'topic_shift':

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


def entity_poor_news_clf_dataset(cli_config, tokenizer):
    dataset = news_clf_dataset_with_ots_ner(cli_config, tokenizer)
    nr_topics = len(dataset['train'].features['labels'].names)

    if cli_config['experiment_version'] == 'mask':
        fn = mask_entities
    elif cli_config['experiment_version'] == 'substitute':
        fn = substitute_entities
    else:
        raise ValueError(f"Invalid version of experiment: {cli_config['experiment_version']}")

    dataset = dataset.map(
            fn, batched=True, batch_size=None, fn_kwargs={
                'variant': cli_config['substitute_variant'],
                'nr_most_frequent': cli_config['nr_most_frequent'], 
                'nr_topics': nr_topics, 
                'cli_config': cli_config, 
                'tokenizer': tokenizer
            }, load_from_cache_file=True
    ).remove_columns(['ner', 'incident.wdt_id'])
    return dataset


def output(df, location='.'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    import pprint
    import csv

    metrics = [k for k in df.columns if 'metric' in k]
    correct_in = 'correct' in df.columns
    
    sp_kws = {
        'squeeze': False,
        'nrows': 2 + (1 if correct_in else 0),
        'ncols': len(metrics),
    }
    sp_kws['figsize'] = (sp_kws['ncols'] * 4.8, sp_kws['nrows'] * 4.8)

    correlations = {}

    def save_correlation(key1, key2):
        correlations[key1, key2] = df[key1].corr(df[key2])

    def scatter_plot(**kwargs):
        sns.scatterplot(**kwargs, s=2, linewidth=0)

    if 'topic_shift' in df.columns:
        # calculate average topic_shift per sample
        df['topic_shift_avg'] = df['topic_shift'].apply(np.mean)

        f, axs = plt.subplots(**sp_kws)
        
        for i,metric in enumerate(metrics):
            save_correlation(metric, 'topic_shift_avg')
            
            scatter_plot(x='topic_shift_avg', y=metric, data=df, ax=axs[0, i])
            draw_histwithmean(df['topic_shift_avg'], df[metric], ax=axs[1, i])

        if correct_in:
            sns.violinplot(x='topic_shift_avg', y='correct', orient='h', inner='box', data=df, ax=axs[2, 0])

        f.savefig(os.path.join(location, 'topic_shift_avg.png'))

    if 'frequencies' in df.columns:
        # calculate average difference in log-frequency
        df['shift_of_log_frequency'] = df['frequencies'].apply(
            lambda sample: list(map(lambda freqs: math.log(freqs[0]) + math.log(freqs[1]), sample))
        )
        # calculate average difference in log-frequency
        df['log_of_frequency_shift'] = df['frequencies'].apply(
            lambda sample: list(map(lambda freqs: math.log(freqs[0] + freqs[1]), sample))
        )

        # calculate average difference in frequency
        df['log_of_absfrequency_diff'] = df['frequencies'].apply(
            lambda sample: list(map(lambda freqs: math.log(1 + abs(freqs[0] - freqs[1])), sample))
        )
        df['diff_of_log_frequency'] = df['frequencies'].apply(
            lambda sample: list(map(lambda freqs: math.log(1 + freqs[0]) - math.log(1 + freqs[1]), sample))
        )
        df['absdiff_of_log_frequency'] = df['frequencies'].apply(
            lambda sample: list(map(lambda freqs: abs(math.log(1 + freqs[0]) - math.log(1 + freqs[1])), sample))
        )


        df['max_log_frequency'] = df['frequencies'].apply(
            lambda sample: list(map(lambda freqs: math.log(max(freqs)), sample))
        )
        df['min_log_frequency'] = df['frequencies'].apply(
            lambda sample: list(map(lambda freqs: math.log(min(freqs)), sample))
        )

        for key in [k for k in df.columns if 'frequency' in k]:
            avg_key = f"{key}_avg"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                df[avg_key] = df[key].apply(np.mean)

            correct_in = 'correct' in df.columns
            f, axs = plt.subplots(**sp_kws)
            sns.despine(f, left=True, bottom=True)

            for i, metric_key in enumerate(metrics):
                print(f"\nCorrelation between {avg_key} and {metric_key}...")
                save_correlation(metric_key, avg_key)
                
                scatter_plot(x=avg_key, y=metric_key, data=df, ax=axs[0, i])
                draw_histwithmean(df[avg_key], df[metric_key], ax=axs[1, i])

            if correct_in:
                sns.violinplot(x=avg_key, y='correct', orient='h', inner='box', data=df, ax=axs[2, 0])

            f.savefig(os.path.join(location, f'{avg_key}.png'))

    pprint.pprint(correlations)
    with open(os.path.join(location, f'correlations.csv'), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['metric', 'column', 'correlation'])
        writer.writeheader()
        writer.writerows([ {'metric': m, 'column': col, 'correlation': cor} for (m, col), cor in correlations.items() ])


def analysis(cli_config, trainer, model, eval_dataset):
    # convert dataset to pandas dataframe
    df = pd.DataFrame(eval_dataset)

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
    label_probs = []
    losses = []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    model.eval()
    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs['nc-0_logits']
        probs = torch.nn.functional.softmax(logits, dim=-1)

        #_label_probs = torch.gather(probs, -1, batch['nc_labels'].unsqueeze(-1))
        #label_probs.extend(_label_probs.squeeze().tolist())
        
        loss = torch.nn.functional.cross_entropy(logits, batch['nc_labels'], reduction='none')
        losses.extend(loss.squeeze().tolist())

        predictions = torch.argmax(logits, dim=-1)
        correct = predictions == batch['nc_labels']
        correctness.extend(correct.tolist())

    #df['metric'] = label_probs
    df['metric_loss'] = losses
    df['correct'] = correctness

    output(df, cli_config['run_path'])


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
        _, trainer, model, eval_dataset = train_news_clf(config, entity_poor_news_clf_dataset)

        if config['do_analysis']:
            analysis(config, trainer, model, eval_dataset)
