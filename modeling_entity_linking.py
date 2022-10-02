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
        self.K = 128 # TODO make config parameter
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, base_outputs, *args):
        """

        Args:
            base_outputs:
            *args:

        Returns:

        """
        labels, return_dict = args
        labels = labels.clone()                                                     # B x N
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = base_outputs['hidden_states'][self.attach_layer]            # B x N x D
        B, N, D = hidden_states.shape
        E_ = B * N

        # Change -100 to padding_idx (so we can embed PAD tokens).
        lbl_ignore = labels == -100                                                 # B x N
        labels[lbl_ignore] = self.nr_entities - 1
        entities = self.entity_embedding(labels)
        entities = entities.view(-1, D)                                             # E' x D

        # Score against each entity in batch.
        all_scores = torch.tensordot(hidden_states, entities, dims=([2], [1]))      # B x N x E'

        # Separate scores for correct entity from the rest.
        all_scores_sq = all_scores.view(-1, E_)                                     # B*N x E'
        lbl_scores = all_scores_sq.diagonal().clone().view(B, N, 1)                 # B x N x 1

        # Take entities from within batch that were scored the highest as negative samples.
        # Making sure that neither the correct entity nor any PAD tokens are part of the top-k.
        all_scores_sq.fill_diagonal_(float('-inf'))
        scr_ignore = torch.bitwise_or(
            lbl_ignore.view(B*N, 1).expand_as(all_scores_sq),
            lbl_ignore.view(1, B*N).expand_as(all_scores_sq),
        )
        all_scores_sq[scr_ignore] = float('-inf')
        # If a batch contains a lot of PAD tokens or K is chosen close to B*N some -inf might get through. However,
        # they will obtain a sigmoid activation of zero, which is also their target so should not be a problem.
        neg_scores, neg_idxs = torch.topk(all_scores, self.K-1, dim=-1)             # B x N x K-1,  B x N x K-1

        # Calculate loss.
        logits = torch.cat((neg_scores, lbl_scores), dim=-1)                        # B x N x K
        target = torch.cat((torch.zeros_like(neg_scores), torch.ones_like(lbl_scores)), dim=-1)
        ignore = lbl_ignore.unsqueeze(-1).expand_as(logits)
        loss = self.loss(logits[~ignore], target[~ignore])

        if not return_dict:
            raise NotImplementedError

        return EntityLinkingOutput(
            loss=loss,
            logits=logits,
        )
