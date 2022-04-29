import argparse

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

    # TODO add version that replaces entities with new special 'entity' token.
    # TODO add version that replaces entity with new special token indicating the type of entity.

    dataset = dataset.map(mask_entities, batched=True).remove_columns(['ner', 'incident.wdt_id'])
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

    args = parser.parse_args()
    config = create_run_folder_and_config_dict(args)
    train_news_clf(config, entity_poor_news_clf_dataset)
