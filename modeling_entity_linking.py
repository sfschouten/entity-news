
from modeling_multi_task import Task

class EntityLinking(Task):

    def __init__(self, config):
        super().__init__(config)

        NR_CLS = 3  # we use IOB
        self.er_classifier = nn.Linear(config.hidden_size, NR_CLS)
        self.ed_classifier = nn.Linear(config.hidden_size, config.num_candidates)

        self.entity_embedding = nn.Embedding(config.num_entities, embedding_dim=config.hidden_size)

    def forward(self):
        pass
