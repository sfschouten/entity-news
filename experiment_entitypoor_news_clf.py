import argparse
from collections import Counter
from dataclasses import dataclass
import random
from typing import List

from train_news_clf import train_news_clf
from experiment_visualize_entity_tokens import news_clf_dataset_with_ots_ner
from utils import create_run_folder_and_config_dict


def entity_poor_news_clf_dataset(cli_config, tokenizer):
    dataset = news_clf_dataset_with_ots_ner(cli_config, tokenizer)

    # remove entities
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
        input_ids = samples['input_ids']
        ner_preds = samples['ner']
        topics = samples['labels']

        class Mention:
            def __init__(self, sample_index: int):
                self.sample_index = sample_index
                self.token_ids = []
                self.token_idxs = []

            def __hash__(self):
                return hash(tuple(self.token_ids))

            def __eq__(self, other):
                return self.token_ids == other.token_ids

            def append(self, token_id, token_idx):
                self.token_ids.append(token_id)
                self.token_idxs.append(token_idx)

        # collect entity mentions
        entity_mentions = []
        for i, (s_input_ids, s_ner_preds) in enumerate(zip(input_ids, ner_preds)):
            for j, (input_id, ner) in enumerate(zip(s_input_ids, s_ner_preds)):
                if ner.startswith('B'):
                    entity_mentions.append(Mention(i))
                    entity_mentions[-1].append(input_id, j)
                if ner.startswith('I') and entity_mentions[-1].token_idxs[-1] == j-1:
                    entity_mentions[-1].append(input_id, j)

        # count, so we know which are most frequent
        entity_mention_count = Counter(entity_mentions)
        most_frequent = entity_mention_count.most_common(50)

        # calculate distribution over topics
        entity_mention_topic_count = Counter([
            (mention, topics[mention.sample_index]) for mention in entity_mentions
        ])

        entity_mentions_by_sample = [[] for _ in range(len(input_ids))]
        for mention in entity_mentions:
            entity_mentions_by_sample[mention.sample_index].append(mention)

        # make the substitutions
        for i, sample_mentions in enumerate(entity_mentions_by_sample):
            # go through mentions in this sample in reverse order
            for mention in sorted(sample_mentions, key=lambda m: m.token_idxs[0], reverse=True):
                start_idx = mention.token_idxs[0]
                for idx in reversed(mention.token_idxs):
                    #print(f"deleting {mention.sample_index}:{idx}")
                    del input_ids[mention.sample_index][idx]

                entity, _ = random.sample(most_frequent, 1)[0]
                for t in reversed(entity.token_ids):
                    #print(f"inserting {t} at {mention.sample_index}:{start_idx}")
                    input_ids[mention.sample_index].insert(start_idx, t)
                #print('----------')
            #print(f'===={i}=====')

    if cli_config['experiment_version'] == 'mask':
        fn = mask_entities
    elif cli_config['experiment_version'] == 'substitute':
        fn = substitute_entities
    else:
        raise ValueError(f"Invalid version of experiment: {cli_config['experiment_version']}")

    dataset = dataset.map(fn, batched=True).remove_columns(['ner', 'incident.wdt_id'])
    return dataset


if __name__ == "__main__":
    # parse cmdline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--report_to', default=None, type=str)

    parser.add_argument('--nc_data_folder', default="../data/minimal")

    parser.add_argument('--mwep_home', default='../mwep')
    parser.add_argument('--runs_folder', default='runs')
    parser.add_argument('--run_name', default=None)

    parser.add_argument('--model', default="distilbert-base-cased")
    parser.add_argument('--probing', action='store_true')
    parser.add_argument('--head_id', default='nc-0', type=str)

    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--continue', action='store_true')
    parser.add_argument('--eval_only', action='store_true')

    parser.add_argument('--eval_strategy', default='steps', type=str)
    parser.add_argument('--eval_frequency', default=500, type=int)
    parser.add_argument('--eval_metric', default='accuracy', type=str)

    # hyper-parameters
    parser.add_argument('--max_nr_epochs', default=100, type=int)
    parser.add_argument('--warmup_steps', default=2000, type=int)
    parser.add_argument('--early_stopping_patience', default=5, type=int)
    parser.add_argument('--batch_size_train', default=64, type=int)
    parser.add_argument('--batch_size_eval', default=64, type=int)
    parser.add_argument('--gradient_acc_steps', default=1, type=int)

    parser.add_argument('--experiment_version', choices=['mask', 'substitute'])

    args = parser.parse_args()
    config = create_run_folder_and_config_dict(args)
    train_news_clf(config, entity_poor_news_clf_dataset)
