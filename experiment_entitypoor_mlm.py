import argparse
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import wandb

from experiment_entitypoor_news_clf import entity_poor_news_clf_dataset, entitypoor_argparse, SUB_VARIANTS, output
from train_mlm import train_mlm_argparse, train_mlm
from utils import create_run_folder_and_config_dict


def entity_poor_news_data(config, tokenizer):
    dataset = entity_poor_news_clf_dataset(config, tokenizer)
    dataset.remove_columns(['labels'])
    return dataset


def analysis(cli_config, trainer, model, eval_dataset):
    # convert dataset to pandas dataframe
    df = pd.DataFrame(eval_dataset)

    eval_dataset = eval_dataset.remove_columns(
        [c for c in eval_dataset.column_names if c not in ['input_ids', 'attention_mask']]
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cli_config['batch_size_eval'],
        collate_fn=trainer.data_collator
    )

    losses = []
    accuracies = []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs['mlm-0_logits']
        N, M, C = logits.shape
        
        loss = torch.nn.functional.cross_entropy(logits.view(-1, C), batch['mlm_labels'].view(-1), reduction='none')
        loss = torch.mean(loss.view(N, M), dim=1)
        losses.extend(loss.squeeze().tolist())

        predictions = torch.argmax(logits, dim=-1)
        accuracy = torch.mean((predictions == batch['mlm_labels']).float(), dim=1)
        accuracies.extend(accuracy.squeeze().tolist())
    
    df['metric_loss'] = losses
    df['metric_accuracy'] = accuracies 

    output(df, location=cli_config['run_path'])


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
        result, trainer, model, eval_dataset = train_mlm(config, entity_poor_news_data)

        if config['do_analysis']:
            analysis(config, trainer, model, eval_dataset)
