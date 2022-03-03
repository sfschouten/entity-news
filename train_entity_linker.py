import argparse

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from utils import create_run_folder_and_config_dict

import el_wiki_dataset


def train_entity_linking(config):
    # init dataset
    dataset = load_dataset(
        el_wiki_dataset.__file__,
        split='full',
        streaming=True
    )

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    tokenized_dataset = dataset.map(
        lambda example: tokenizer(example['text'], padding='max_length', truncation=True),
    )

    # load model
    model = AutoModel.from_pretrained(config['model'])

    # encode mention with context

    # encode/embed entity

    # calculate loss

    # backprop

    pass


if __name__ == "__main__":
    # parse cmdline arguments
    parser = argparse.ArgumentParser()

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
    train_entity_linking(
        create_run_folder_and_config_dict(args)
    )
