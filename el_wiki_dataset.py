import sys

from dataclasses import dataclass
from typing import Callable

import datasets
from datasets import load_dataset, DownloadManager, DatasetInfo


_DESCRIPTION = """\
Wrapper around the KILT-Wikipedia dataset for ease of use with Entity Linking.
"""


def basic_mention_extractor(mentioner, mentioned, anchor, config) -> [str, list[dict]]:
    start = anchor['start']
    end = anchor['end']
    par = anchor['paragraph_id']
    par_text = mentioner['text']['paragraph'][par]

    max_len = config.max_mention_context_length
    context_len = (max_len - (end - start)) // 2
    c_start = max(start - context_len, 0)
    c_end = min(end + context_len, len(par_text) - 1)
    text = par_text[c_start:c_end]

    mention = {
        "start_char": start - c_start,
        "end_char": end - c_start,
        "mentioned_wikipedia_id": mentioned['wikipedia_id'],
    }

    if 'mentioned_wikipedia_title' in config.optional_fields_to_add:
        mention['mentioned_wikipedia_title'] = anchor['wikipedia_title']

    if 'mentioned_categories' in config.optional_fields_to_add:
        mention['mentioned_categories'] = mentioned['categories']

    return text, [mention]


@dataclass()
class KILTWikipediaForELConfig(datasets.BuilderConfig):

    OPTIONAL_FIELDS = {
        'mentioning_wikipedia_id',
        'mentioning_wikipedia_title',
        'mentioned_wikipedia_title',
        'mentioned_categories',
    }
    optional_fields_to_add: set = None

    # The number of paragraphs of a mentioned entity's wikipedia page to include.
    nr_mentioned_wikipedia_paragraphs: int = 0

    # The extractor function to use
    mention_extractor: Callable = basic_mention_extractor

    max_mention_context_length: int = 500

    shuffle_base_dataset: bool = False
    shuffle_base_dataset_seed = None

    max_samples: int = sys.maxsize

    def __post_init__(self):
        if self.optional_fields_to_add is None:
            self.optional_fields_to_add = set()

        if not self.optional_fields_to_add.issubset(self.OPTIONAL_FIELDS):
            raise ValueError(f"invalid optional fields: "
                             f"{self.optional_fields_to_add - self.OPTIONAL_FIELDS}")


class KILTWikipediaForEL(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = KILTWikipediaForELConfig
    config: KILTWikipediaForELConfig

    def _info(self) -> DatasetInfo:
        features = {}

        # optional attributes of mentioning entity
        if 'mentioning_wikipedia_id' in self.config.optional_fields_to_add:
            features['mentioning_wikipedia_id'] = datasets.Value("string")
        if 'mentioning_wikipedia_title' in self.config.optional_fields_to_add:
            features['mentioning_wikipedia_title'] = datasets.Value("string")

        # text with mention
        features['mentioning_text'] = datasets.Value("string")

        # mentions themselves
        mentions = {
            "start_char": datasets.Value("int16"),
            "end_char": datasets.Value("int16"),
            "mentioned_wikipedia_id": datasets.Value("string"),
        }

        # optional attributes of mentioned entity
        if self.config.nr_mentioned_wikipedia_paragraphs > 0:
            mentions['mentioned_wikipedia_text'] = datasets.features.Sequence({
                "paragraph": datasets.Value("string")
            })
        if 'mentioned_wikipedia_title' in self.config.optional_fields_to_add:
            mentions['mentioned_wikipedia_title'] = datasets.Value("string")
        if 'mentioned_categories' in self.config.optional_fields_to_add:
            mentions['mentioned_categories'] = datasets.Value("string")

        features['mentions'] = datasets.features.Sequence(mentions)
        return datasets.DatasetInfo(
            features=datasets.Features(features),
            description=_DESCRIPTION,
            homepage="TODO",
        )

    def _split_generators(self, dl_manager: DownloadManager):
        self.base_dataset = load_dataset(
            'kilt_wikipedia',
            split='full',
        )

        if self.config.shuffle_base_dataset:
            self.base_dataset = self.base_dataset.shuffle(
                seed=self.config.shuffle_base_dataset_seed
            )

        return [
            datasets.SplitGenerator(name="full", gen_kwargs={})
        ]

    def _generate_examples(self):
        entity_idxs = {}        # indices of entities
        entity_mentions = {}    # where entities are mentioned
        m_idx = 0

        def stop_condition():
            return m_idx + len(entity_mentions[entity['wikipedia_id']]) >= self.config.max_samples

        def new_samples(wikipedia_id):
            """
            Checks for entities we know have mentioned the entity corresponding to `wikipedia_id',
            and yields them.
            """
            nonlocal m_idx

            entity_mentions_ = entity_mentions.pop(wikipedia_id)
            for mentioner_id, anchor_idx in entity_mentions_:
                mentioner = self.base_dataset[entity_idxs[mentioner_id]]
                mentioned = self.base_dataset[entity_idxs[wikipedia_id]]

                anchor = {key: values[anchor_idx] for key, values in mentioner['anchors'].items()}

                text, mentions_ = self.config.mention_extractor(
                    mentioner, mentioned, anchor, self.config
                )
                result = {
                    "mentioning_text": text,
                    "mentions": mentions_
                }

                if 'mentioning_wikipedia_id' in self.config.optional_fields_to_add:
                    result['mentioning_wikipedia_id'] = mentioned['wikipedia_id']
                if 'mentioning_wikipedia_title' in self.config.optional_fields_to_add:
                    result['mentioning_wikipedia_title'] = mentioner['wikipedia_title']

                yield m_idx, result
                m_idx += 1

        # First build index of:
        #   1. which entities mention each other (`entity_mentions`),
        #   2. the index in the dataset of each entity (`entity_idxs`); and
        # immediately yield mentions from entities that occur before the entity they mention.
        for e_idx, entity in enumerate(self.base_dataset):
            entity_idxs[entity['wikipedia_id']] = e_idx

            # for each mention store that the current entity mentioned it
            mentions = [t[-1] for t in zip(*entity['anchors'].values())]
            for anchor_idx, mention_wiki_id in enumerate(mentions):
                if mention_wiki_id not in entity_mentions:
                    entity_mentions[mention_wiki_id] = []

                entity_mentions[mention_wiki_id].append(
                    (entity['wikipedia_id'], anchor_idx)
                )

            if entity['wikipedia_id'] in entity_mentions:
                if stop_condition():
                    print("Reached configured maximum number of samples, stopping generation.")
                    return
                yield from new_samples(entity['wikipedia_id'])

        # Now yield mentions from entities that occurred after the entities they mentioned.
        for entity in self.base_dataset:
            if entity['wikipedia_id'] in entity_mentions:
                if stop_condition():
                    print("Reached configured maximum number of samples, stopping generation.")
                    return
                yield from new_samples(entity['wikipedia_id'])

        print(f"All possible mentions have been generated. Last mention index: {m_idx}.")


if __name__ == "__main__":
    from pprint import pprint

    # load a dataset
    dataset = load_dataset(
        __file__,
        streaming=True,
        optional_fields_to_add={
            'mentioning_wikipedia_id',
            'mentioning_wikipedia_title',
            'mentioned_wikipedia_title',
            'mentioned_categories',
        }
    )

    # print some samples
    for i, test in enumerate(dataset['full']):
        print(i)
        pprint(test)
        print()
        if i >= 9:
            break
