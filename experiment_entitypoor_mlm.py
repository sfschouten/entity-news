import argparse

import wandb

from experiment_entitypoor_news_clf import entity_poor_news_clf_dataset, entitypoor_argparse, SUB_VARIANTS
from train_mlm import train_mlm_argparse, train_mlm
from utils import create_run_folder_and_config_dict


def entity_poor_news_data(config, tokenizer):
    dataset = entity_poor_news_clf_dataset(config, tokenizer)
    dataset.remove_columns(['labels'])
    return dataset


if __name__ == "__main__":
    # parse cmdline arguments
    parser = argparse.ArgumentParser()
    parser = train_mlm_argparse(parser)
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
            train_mlm(config, entity_poor_news_data)
            run.finish()
    else:
        config = create_run_folder_and_config_dict(args)
        train_mlm(config, entity_poor_news_data)
