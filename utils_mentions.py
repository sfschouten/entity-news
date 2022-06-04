
class Mention:
    def __init__(self, sample_index: int, type: str):
        self.sample_index = sample_index
        self.type = type
        self.token_ids = []
        self.token_idxs = []

    def __hash__(self):
        return hash(tuple(self.token_ids))

    def __eq__(self, other):
        return self.token_ids == other.token_ids

    def append(self, token_id, token_idx):
        self.token_ids.append(token_id)
        self.token_idxs.append(token_idx)


def samples_to_mentions(samples):
    """
    TODO
    Args:
        samples:

    Returns:

    """
    input_ids = samples['input_ids']
    ner_preds = samples['ner']

    entity_mentions = []
    for i, (s_input_ids, s_ner_preds) in enumerate(zip(input_ids, ner_preds)):
        for j, (input_id, ner) in enumerate(zip(s_input_ids, s_ner_preds)):
            if ner.startswith('B'):
                entity_mentions.append(Mention(i, ner.split('-')[1]))
                entity_mentions[-1].append(input_id, j)
            if ner.startswith('I') and entity_mentions[-1].token_idxs[-1] == j - 1 \
                    and ner.endswith(entity_mentions[-1].type):
                entity_mentions[-1].append(input_id, j)

    return entity_mentions


def mentions_by_sample(mentions, nr_samples):
    """
    organize mentions by sample
    Args:
        mentions:
        nr_samples:

    Returns:

    """
    entity_mentions_by_sample = [[] for _ in range(nr_samples)]
    for mention in mentions:
        entity_mentions_by_sample[mention.sample_index].append(mention)

    return entity_mentions_by_sample

