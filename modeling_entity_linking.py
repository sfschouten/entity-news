from typing import Optional

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

from transformers.file_utils import ModelOutput

from modeling_versatile import Head


class EntityLinkingOutput(ModelOutput):
    loss: Optional[FloatTensor] = None
    logits: FloatTensor = None
    top_k_idxs: LongTensor = None


class EntityLinking(Head):

    def __init__(self, key, config):
        super().__init__(key, config)
        config_dict = config.to_dict()

        # TODO: add additional transformations before linking?
        # TODO: add dropout?

        self.attach_layer = config_dict.get(f'{key}_attach_layer', -1)

        self.nr_entities = config_dict[f'{key}_nr_entities']  # reserve one for PAD
        self.entity_embedding = nn.Embedding(
            self.nr_entities, embedding_dim=config.hidden_size, padding_idx=self.nr_entities-1)
        print(f"Embedding dimensions: {self.entity_embedding.weight.shape}")

        self.K = 128  # TODO make config parameter
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, base_outputs, *args):
        """

        Args:
            base_outputs:
            *args:

        Returns:

        """
        labels, return_dict = args
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = base_outputs['hidden_states'][self.attach_layer]            # B x N x D

        # Change -100 to padding_idx (so we can embed PAD tokens).
        pad_tokens = labels == -100                                                 # B x N
        labels = labels[~pad_tokens]                                                # T
        entities = self.entity_embedding(labels)                                    # T x D

        # Score against each entity in batch.
        hidden_states = hidden_states[~pad_tokens]                                  # T x D
        all_scores = hidden_states @ entities.T                                     # T x T
        same_labels = labels.unsqueeze(0).expand_as(all_scores) == labels.unsqueeze(1).expand_as(all_scores)

        # Separate scores for correct entity from the rest.
        lbl_scores = all_scores.diagonal().unsqueeze(-1)                            # T x 1

        # Take entities from within batch that were scored the highest as negative samples.
        # Making sure that the label entities are not part of the top-k.
        all_scores = all_scores.clone()
        all_scores.fill_diagonal_(float('-inf'))
        all_scores.masked_fill_(same_labels, float('-inf'))
        neg_scores, neg_idxs = torch.topk(all_scores, self.K-1, dim=-1)             # T x K-1,  T x K-1
        top_k_labels = torch.take(labels, neg_idxs)

        # Calculate loss.
        logits = torch.cat((lbl_scores, neg_scores), dim=-1)                        # T x K
        target = torch.cat((torch.ones_like(lbl_scores), torch.zeros_like(neg_scores)), dim=-1)
        loss = self.loss(logits[~logits.isinf()], target[~logits.isinf()])

        if not return_dict:
            raise NotImplementedError

        return EntityLinkingOutput(
            loss=loss,
            logits=logits.unsqueeze(0),
            top_k_idxs=torch.cat((labels.unsqueeze(-1), top_k_labels), -1).unsqueeze(0)
        )
