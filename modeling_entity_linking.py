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

        self.wikipedia_id_to_idx = torch.nn.Parameter(
            torch.full((61500000,), -1, dtype=torch.int32),
            requires_grad=False
        )
        self.wikipedia_id_to_idx[0] = 0

        self.entity_embedding = nn.Embedding(1, embedding_dim=config.hidden_size)

        self.K = config_dict.get(f'{key}_per_token_k', 128)
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')

    def extend_embedding(self, dataset):
        """
        Extends the embedding to include entities from `dataset` that were previously unknown.
        """
        import itertools
        wikipedia_ids = [int(id) for id in set(itertools.chain.from_iterable(dataset['nel_labels']))]
        print(f"minimum wikipedia_id: {min(wikipedia_ids)}")
        print(f"maximum wikipedia_id: {max(wikipedia_ids)}")

        wikipedia_ids = torch.tensor(wikipedia_ids)

        current_embedding = self.entity_embedding
        current_size, _ = current_embedding.weight.shape

        desired_ids = torch.LongTensor(wikipedia_ids)
        desired_idxs = torch.index_select(self.wikipedia_id_to_idx, -1, desired_ids)

        nr_newly_desired = len(desired_ids[desired_idxs == -1])
        new_size = current_size + nr_newly_desired
        print(f'There are {nr_newly_desired} newly desired wikipedia_ids.')
        if new_size > current_size:
            print(f'Extending the embedding to {new_size}')

            # Add pointers from previously unknown entities to new part of Embedding.
            self.wikipedia_id_to_idx.index_add_(
                0, desired_ids[desired_idxs == -1],
                torch.arange(start=current_size, end=new_size, dtype=torch.int32)
            )

            new_embedding = nn.Embedding(new_size, embedding_dim=self.config.hidden_size)
            new_embedding.weight.data[:current_size] = current_embedding.weight.data
            self.entity_embedding = new_embedding

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

        # convert wikipedia_id to index, and ignore PAD tokens and unknown entities
        pad_tokens = labels == -100                                                 # B x N
        labels[~pad_tokens] = torch.index_select(self.wikipedia_id_to_idx, -1, labels[~pad_tokens]).type(labels.type())
        oov_tokens = labels == -1
        labels = labels[~pad_tokens & ~oov_tokens]                                  # T
        entities = self.entity_embedding(labels)                                    # T x D

        # Score against each entity in batch.
        hidden_states = hidden_states[~pad_tokens & ~oov_tokens]                    # T x D
        all_scores = hidden_states @ entities.T                                     # T x T
        same_labels = labels.unsqueeze(0).expand_as(all_scores) == labels.unsqueeze(1).expand_as(all_scores)

        # Separate scores for correct entity from the rest.
        lbl_scores = all_scores.diagonal().unsqueeze(-1)                            # T x 1

        # Take entities from within batch that were scored the highest as negative samples.
        # Making sure that the label entities are not part of the top-k.
        all_scores = all_scores.clone()
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
