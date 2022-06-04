import torch.nn as nn

from modeling_versatile import Head


class EntityLinking(Head):

    def __init__(self, key, config):
        super().__init__(key, config)
        config_dict = config.to_dict()

        self.nr_entities = config_dict[f'{key}_nr_entities']

        NR_CLS = 3  # we use IOB
        self.er_classifier = nn.Linear(config.hidden_size, NR_CLS)

        self.entity_embedding = nn.Embedding(self.nr_entities, embedding_dim=config.hidden_size)

    def extract_kwargs(self, kwargs):
        pass

    def forward(self, base_outputs, *args):
        pass
