import argparse
from collections import Counter

from transformers import AutoTokenizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from experiment_visualize_entity_tokens import news_clf_dataset_with_ots_ner
from train_nerc import train_nerc_argparse, train_entity_recognition
from utils import create_run_folder_and_config_dict
from utils_mentions import samples_to_mentions, mentions_by_sample

TAGS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
TAG2IDX = {tag: idx for idx, tag in enumerate(TAGS)}


def mwep_silver_ner(cli_config, tokenizer):
    dataset = news_clf_dataset_with_ots_ner(cli_config, tokenizer)
    dataset = dataset.remove_columns(['labels', 'incident.wdt_id'])
    dataset = dataset.rename_column('ner', 'labels')

    def labels(sample):
        sample['labels'] = [TAG2IDX[tag] for tag in sample['labels']]
        return sample

    dataset = dataset.map(labels, batched=False)
    return dataset


def train_nerc_and_analyze(cli_config):
    # train probe
    train_entity_recognition(cli_config, mwep_silver_ner)

    # read in predictions of probe
    eval_logits = np.load(cli_config['run_path'] + '/logits.txt')

    # read in MWEP with silver-standard NERC labels
    tokenizer = AutoTokenizer.from_pretrained(cli_config['model'])
    key = 'test' if cli_config['eval_on_test'] else 'validation'
    dataset = mwep_silver_ner(cli_config, tokenizer)[key]

    correctness_vs_frequency = []

    def calc_frequencies(samples):
        entity_mentions = samples_to_mentions(samples)
        entity_mention_count = Counter(entity_mentions)
        entity_mentions_by_sample = mentions_by_sample(entity_mentions, len(samples))

        for mentions, logits in zip(entity_mentions_by_sample, eval_logits):
            for mention in mentions:
                freq = entity_mention_count[mention]
                prob = logits[TAG2IDX[mention.type]]
                correctness_vs_frequency.append((freq, prob))

    dataset.map(calc_frequencies, batched=True, batch_size=None)

    correct_vs_freq = np.array(correctness_vs_frequency)

    # report correlation between correctness of probe-prediction vs. frequency
    sns.scatterplot(data=correct_vs_freq)
    # TODO add entity-type as color

    plt.savefig(cli_config['run_path'] + '/correct_freq_scatter.png')


if __name__ == "__main__":
    # parse cmdline arguments
    parser = argparse.ArgumentParser()
    parser = train_nerc_argparse(parser)

    parser.add_argument('--mwep_home', default='../mwep')
    parser.add_argument('--nc_data_folder', default="../data/medium_plus")

    args = parser.parse_args()
    config = create_run_folder_and_config_dict(args)
    train_nerc_and_analyze(config)
