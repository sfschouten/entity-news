import argparse
from collections import Counter

from transformers import AutoTokenizer

import numpy as np
import scipy.stats

import seaborn as sns
import matplotlib.pyplot as plt

from experiment_visualize_entity_tokens import news_clf_dataset_with_ots_ner
from train_nerc import train_nerc_argparse, train_entity_recognition, conll2003_dataset
from utils import create_run_folder_and_config_dict
from utils_mentions import samples_to_mentions, mentions_by_sample, calc_mention_topic_dist

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
    nr_topics = len(dataset.features['labels'].names)

    metrics = []
    types = []

    def calc_metrics(samples):
        entity_mentions = samples_to_mentions(samples)
        entity_mention_count = Counter(entity_mentions)
        entity_mentions_by_sample = mentions_by_sample(entity_mentions, len(samples['input_ids']))

        topics = samples['labels']
        topic_dists = calc_mention_topic_dist(entity_mentions, topics, nr_topics)

        for mentions, logits in zip(entity_mentions_by_sample, eval_logits):
            for mention in mentions:
                mention_acc_a = np.argmax(logits[mention.token_idxs[0]]) == TAG2IDX['B-'+mention.type]
                mention_acc_b = sum(
                    np.argmax(logits[i]) == TAG2IDX['I-'+mention.type]
                    for i in mention.token_idxs[1:]
                )
                mention_acc = mention_acc_a and mention_acc_b == len(mention.token_idxs[1:])
                metrics.append([
                    entity_mention_count[mention],
                    scipy.stats.entropy(topic_dists[mention]),
                    mention_acc,        # true accuracy, requiring full mention to be tagged
                    mention_acc_a       # only B-tag required
                ])
                types.append(mention.type)

    dataset.map(calc_metrics, batched=True, batch_size=None)
    metrics_arr = np.array(metrics)

    # accuracy probe vs. frequency
    #freq, topic_entropy, correct, b_correct = metrics_arr[:, 0], metrics_arr[:, 1], \
    #                                          metrics_arr[:, 2], metrics_arr[:, 3]
    freq, topic_entropy, _, correct = [
        np.squeeze(subarr) for subarr in np.split(metrics_arr, 4, axis=1)]

    # violin plot
    plt.xlim(0, 600)
    sns.violinplot(x=freq, y=correct, orient='h', inner='box')
    plt.xlabel('Frequency')
    plt.yticks(np.array([0, 1]), ['Incorrect', 'Correct'])
    plt.tight_layout()
    plt.savefig(cli_config['run_path'] + '/correct_freq_violin.png')
    plt.clf()

    #sns.violinplot(x=freq, y=correct, orient='h', inner='box', hue=types)
    #plt.savefig(cli_config['run_path'] + '/correct_freq_type_violins.png')
    #plt.clf()

    #  histogram
    draw_histwithmean(np.log(freq), correct, nr_bins=6,
                      labels_fn=lambda bins: map('{:.01f}'.format, np.exp(bins)))
    plt.xlabel('Frequency')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(cli_config['run_path'] + '/correct_freq_hist.png')
    plt.clf()

    calc_meantest(freq, correct)

    # accuracy probe vs. topic distribution entropy
    sns.violinplot(x=topic_entropy, y=correct, orient='h', inner='box')
    plt.xlabel('Entropy')
    plt.yticks(np.array([0, 1]), ['Incorrect', 'Correct'])
    plt.tight_layout()
    plt.savefig(cli_config['run_path'] + '/correct_entropy_violin.png')
    plt.clf()

    draw_histwithmean(topic_entropy, correct)
    plt.xlabel('Entropy')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(cli_config['run_path'] + '/correct_entropy_hist.png')
    plt.clf()

    calc_meantest(topic_entropy, correct)


def draw_histwithmean(variable, metric, nr_bins=7, labels_fn=lambda _: None, ax=None, confidence=.95):
    bins = np.linspace(variable.min(), variable.max() + 1e-12, nr_bins + 1)
    c = np.digitize(variable, bins)

    mean = [np.mean(metric[c == i]) for i in range(1, len(bins))]
    count = [np.sum(c == i) for i in range(1, len(bins))]

    ci = [scipy.stats.sem(metric[c == i]) 
            * scipy.stats.t.ppf((1+confidence) / 2., count[i-1]) for i in range(1, len(bins))]

    ax.bar(bins[:-1], mean, width=bins[1] - bins[0], align='edge', ec='black', yerr=ci)
    ax.set_xticks(bins, labels=labels_fn(bins), rotation=45)
    ax.margins(x=0.02)
    
    with np.printoptions(precision=5):
        print(f'bins: {bin}')
        print(f'means: {mean}')
        print(f'ci-{confidence}: {ci}')
        print(f'counts: {count}')
        print()


def calc_meantest(variable, correct):
    true_var_mean = variable[correct == 1].mean()
    false_var_mean = variable[correct == 0].mean()
    true_var_std = variable[correct == 1].std()
    false_var_std = variable[correct == 0].std()
    true_var_stderr = true_var_std / np.sqrt((correct == 1).sum())
    false_var_stderr = false_var_std / np.sqrt((correct == 0).sum())
    print("(Intermediate) results of calculations...")
    print(f"True mean: {true_var_mean}, False mean: {false_var_mean}")
    print(f"True std: {true_var_std}, False std: {false_var_std}")
    print(f"True stderr: {true_var_stderr}, False stderr: {false_var_stderr}")
    print()


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
