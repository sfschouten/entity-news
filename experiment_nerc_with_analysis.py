import argparse
from collections import Counter

from transformers import AutoTokenizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from experiment_visualize_entity_tokens import news_clf_dataset_with_ots_ner
from train_nerc import train_nerc_argparse, train_entity_recognition, conll2003_dataset
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

    cli_config['eval_datasets'] = ['mwep_silver_ner']

    if cli_config['logits_path'] is None:
        # train probe
        train_entity_recognition(cli_config, {
            'conll': conll2003_dataset,
            'mwep_silver_ner': mwep_silver_ner
        })
        logits_path = cli_config['run_path'] + '/logits.npy'
    else:
        logits_path = cli_config['logits_path']

    # read in predictions of probe
    eval_logits = np.load(logits_path)

    # read in MWEP with silver-standard NERC labels
    tokenizer = AutoTokenizer.from_pretrained(cli_config['model'])
    key = 'test' if cli_config['eval_on_test'] else 'validation'
    dataset = news_clf_dataset_with_ots_ner(cli_config, tokenizer)[key]

    correctness_vs_frequency = []
    types = []

    def calc_frequencies(samples):
        entity_mentions = samples_to_mentions(samples)
        entity_mention_count = Counter(entity_mentions)
        entity_mentions_by_sample = mentions_by_sample(entity_mentions, len(samples['input_ids']))

        for mentions, logits in zip(entity_mentions_by_sample, eval_logits):
            for mention in mentions:
                log_freq = np.log(entity_mention_count[mention])
                mention_acc_a = np.argmax(logits[mention.token_idxs[0]]) == TAG2IDX['B-'+mention.type]
                mention_acc_b = sum(
                    np.argmax(logits[i]) == TAG2IDX['I-'+mention.type]
                    for i in mention.token_idxs[1:]
                )
                mention_acc = mention_acc_a and mention_acc_b == len(mention.token_idxs[1:])
                correctness_vs_frequency.append([log_freq, mention_acc])
                types.append(mention.type)

    dataset.map(calc_frequencies, batched=True, batch_size=None)
    correct_vs_freq = np.array(correctness_vs_frequency).T

    # report correlation between correctness of probe-prediction vs. frequency
    log_freq, correct = correct_vs_freq[0, :], correct_vs_freq[1, :]
    sns.violinplot(x=log_freq, y=correct, orient='h', inner='box')
    plt.savefig(cli_config['run_path'] + '/correct_freq_violin.png')
    plt.clf()
    sns.violinplot(x=log_freq, y=correct, orient='h', inner='box', hue=types)
    plt.savefig(cli_config['run_path'] + '/correct_freq_type_violins.png')

    true_mean = log_freq[correct == 1].mean()
    false_mean = log_freq[correct == 0].mean()
    true_std = log_freq[correct == 1].std()
    false_std = log_freq[correct == 0].std()
    true_stderr = true_std / np.sqrt((correct == 1).sum())
    false_stderr = false_std / np.sqrt((correct == 0).sum())
    print(f"True mean: {true_mean}, False mean: {false_mean}")
    print(f"True std: {true_std}, False std: {false_std}")
    print(f"True stderr: {true_stderr}, False stderr: {false_stderr}")



if __name__ == "__main__":
    # parse cmdline arguments
    parser = argparse.ArgumentParser()
    parser = train_nerc_argparse(parser)

    parser.add_argument('--mwep_home', default="../mwep")
    parser.add_argument('--nc_data_folder', default="../data/medium_plus")

    parser.add_argument('--logits_path', default=None)

    args = parser.parse_args()
    config = create_run_folder_and_config_dict(args)
    train_nerc_and_analyze(config)
