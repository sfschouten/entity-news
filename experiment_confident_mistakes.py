import argparse
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

from experiment_entitypoor_news_clf import entity_poor_news_clf_dataset
from experiment_visualize_entity_tokens import news_clf_dataset_with_ots_ner
from utils import create_run_folder_and_config_dict


def analyse(cli_config):

    # load dataset (valid/test split)
    tokenizer = AutoTokenizer.from_pretrained(cli_config['model'])
    dataset1 = news_clf_dataset_with_ots_ner(cli_config, tokenizer)
    cli_config['experiment_version'] = 'substitute'
    cli_config['substitute_variant'] = 'random_tokens'
    dataset2 = entity_poor_news_clf_dataset(cli_config, tokenizer)

    labels1 = dataset1['validation']['labels']
    labels2 = dataset2['validation']['labels']
    print(labels1 == labels2)
    print(sum(x==y for x,y in zip(labels1, labels2)))

    dataset = dataset1['test' if cli_config['eval_on_test'] else 'validation']

    logits_paths = cli_config['logits_paths']
    entropies_path = cli_config['entropies_path']

    eval_logits = []
    for i, arg in enumerate(logits_paths):
        print(f"loading {arg}")
        logits_ = np.loadtxt(arg)
        eval_logits.append(logits_)

    eval_logits = np.stack(eval_logits, axis=1)  # N, M, K

    eval_entropies = np.load(entropies_path)

    relevant_samples = []
    predictions = []

    def find_relevant(dataset):
        nonlocal relevant_samples
        for input_ids, label, logits, entropy in zip(dataset['input_ids'], dataset['labels'],
                                                     eval_logits, eval_entropies):
            # argmax logits
            pred = np.argmax(np.mean(logits, axis=0))
            predictions.append((pred, label))
            if pred != label:
                relevant_samples.append((input_ids, entropy))

        # sort by entropy
        relevant_samples = sorted(relevant_samples, key=lambda x: x[1])

    dataset.map(find_relevant, batched=True, batch_size=None)

    print(Counter(predictions))

    print(len(relevant_samples))


if __name__ == "__main__":
    # parse cmdline arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--runs_folder', default='runs')
    parser.add_argument('--run_name', default=None)

    parser.add_argument('--model', default="distilbert-base-cased")

    parser.add_argument('--mwep_home', default="../mwep")
    parser.add_argument('--nc_data_folder', default="../data/medium_plus")

    parser.add_argument('--eval_on_test', action='store_true')

    parser.add_argument('--logits_paths', default=None, nargs='+')
    parser.add_argument('--entropies_path', default=None)

    parser.add_argument('--batch_size_eval', default=32, type=int)

    args = parser.parse_args()
    config = create_run_folder_and_config_dict(args)
    analyse(config)
