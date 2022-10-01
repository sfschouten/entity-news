from typing import Optional

import torch
import torch.nn as nn
from torch import FloatTensor

from transformers.file_utils import ModelOutput

from modeling_versatile import Head


class EntityLinkingOutput(ModelOutput):
    loss: Optional[FloatTensor] = None
    logits: FloatTensor = None


class EntityLinking(Head):

    def __init__(self, key, config):
        super().__init__(key, config)
        config_dict = config.to_dict()

        # TODO: add dropout?
        # TODO: add additional transformations before linking?
        self.attach_layer = config_dict.get(f'{key}_attach_layer', -1)

        self.nr_entities = config_dict[f'{key}_nr_entities'] + 1  # reserve one for PAD
        self.entity_embedding = nn.Embedding(
            self.nr_entities, embedding_dim=config.hidden_size, padding_idx=self.nr_entities-1)
        self.K = 111    # TODO make config parameter
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, base_outputs, *args):
        labels, return_dict = args
        labels = labels.clone()                                                     # B x N
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = base_outputs['hidden_states'][self.attach_layer]            # B x N x D
        B, N, D = hidden_states.shape
        E_ = B * N

        lbl_ignore = labels == -100                                                 # B x N
        labels[lbl_ignore] = self.nr_entities - 1
        entities = self.entity_embedding(labels)
        entities = entities.view(-1, D)                                             # E' x D

        # score against each entity in batch
        all_scores = torch.tensordot(hidden_states, entities, dims=([2], [1]))      # B x N x E'
        # separate correct entity from rest
        all_scores_sq = all_scores.view(-1, E_)                                     # B*N x E'

        lbl_scores = torch.diagonal(all_scores_sq).view(B, N, 1)                    # B x N x 1
        oth_scores = all_scores_sq.flatten()[1:] \
                         .view(E_-1, E_+1)[:, :-1].reshape(B, N, E_-1)              # B x N x E'-1
        # take negative samples that were scored the highest
        neg_scores, neg_idxs = torch.topk(oth_scores, self.K-1, dim=-1)             # B x N x K-1,  B x N x K-1

        increment = torch.eye(B*N, E_-1, device=neg_idxs.device).cumsum(-1).long()
        temp1 = neg_idxs.view(B*N, self.K-1)
        temp2 = torch.gather(increment, 1, temp1)
        neg_idxs = neg_idxs + temp2.view(B, N, self.K-1)

        neg_ignore = torch.take(lbl_ignore.view(-1), neg_idxs)
        ignore = torch.cat((neg_ignore, lbl_ignore.unsqueeze(-1)), dim=-1)

        # calculate loss
        logits = torch.cat((neg_scores, lbl_scores), dim=-1)                        # B x N x K
        target = torch.cat((torch.zeros_like(neg_scores), torch.ones_like(lbl_scores)), dim=-1)
        loss = self.loss(logits[~ignore], target[~ignore])

        if not return_dict:
            raise NotImplementedError

        return EntityLinkingOutput(
            loss=loss,
            logits=logits,
        )
