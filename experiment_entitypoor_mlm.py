import argparse
import math
import os

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
    batch_size = cli_config['batch_size_eval']

    eval_dataset = eval_dataset.remove_columns(
        [c for c in eval_dataset.column_names if c not in ['input_ids', 'attention_mask']]
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=trainer.data_collator
    )
    tokenizer = trainer.data_collator.tokenizer

    losses = []
    accuracies = []
    
    # TODO retrieve constants below from config
    max_len = 512
    max_relevant = math.ceil(max_len * .15)  # 15% of tokens are masked
    l = max_relevant * batch_size
    
    logits_path = os.path.join(cli_config['run_path'], 'logits.npy')
    logits_shape = (len(eval_dataset)*max_relevant, len(tokenizer)) 
    logits = np.memmap(logits_path, dtype='float16', mode='w+', shape=logits_shape)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    for i, batch in enumerate(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        labels = batch['mlm_labels']
        _logits = outputs['mlm-0_logits']
        N, M, C = _logits.shape
        device = _logits.device

        # not all logits are relevant, only those tokens that were masked for prediction
        # we move all these tokens to the front
        relevant_idx = labels != -100
        idxs = torch.arange(M, device=device).unsqueeze(0).expand(N, -1)
        END_MARKER = torch.tensor(max_len, device=device).view(1, 1)

        # associate relevant tokens with index, and others with a number higher than all indices
        # then sort by those values and throw away tokens we know to be irrelevant
        to_sort = torch.where(relevant_idx, idxs, END_MARKER)
        idx = torch.argsort(to_sort)
        sorted_ = torch.gather(_logits, 1, idx.unsqueeze(-1).expand_as(_logits))
        sorted_ = sorted_[:, :max_relevant, :]

        # now set all remaining irrelevant logits to -100
        nr_relevant = relevant_idx.sum(-1, keepdims=True)
        not_relevant_idx = idxs[:,:77] >= nr_relevant
        sorted_[not_relevant_idx.unsqueeze(-1).expand_as(sorted_)] = -100

        # and finally write to numpy array
        _l = len(labels) * max_relevant
        logits[i*l:i*l+_l, :] = sorted_.reshape(_l, -1).cpu().numpy()

        loss = torch.nn.functional.cross_entropy(_logits.view(-1, C), batch['mlm_labels'].view(-1), reduction='none')
        loss = torch.mean(loss.view(N, M), dim=1)
        losses.extend(loss.squeeze().tolist())

        predictions = torch.argmax(_logits, dim=-1)
        accuracy = torch.mean((predictions == batch['mlm_labels']).float(), dim=1)
        accuracies.extend(accuracy.squeeze().tolist())
    
    df['metric_loss'] = losses
    df['metric_accuracy'] = accuracies 

    logits.flush()

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
