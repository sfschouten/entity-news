import sys

from itertools import accumulate
from dataclasses import dataclass
from typing import Callable
from bisect import bisect_left
import pickle

import datasets
from datasets import load_dataset, DownloadManager, DatasetInfo

from tqdm import tqdm

_DESCRIPTION = """\
Wrapper around the KILT-Wikipedia dataset for ease of use with Entity Linking.
"""


def basic_mention_extractor(mentioner, lookup_fn, anchor, config) -> [str, list[dict]]:
    start = anchor['start']
    end = anchor['end']
    par = anchor['paragraph_id']
    paragraphs = mentioner['text']['paragraph']
    par_text = mentioner['text']['paragraph'][par]

    max_len = config.max_mention_context_length
    context_len = (max_len - (end - start)) // 2
    text = par_text[start:end]

    # same-paragraph (local) context start/end
    lc_start = max(start - context_len, 0)
    lc_end = min(end + context_len, len(par_text))

    # list of shift values (to map within-paragraph indices to full-mention indices)
    # first collect the lengths of the text to be added from each paragraph
    shifts = [len(par_text[lc_start:])]
    s = lc_start
    l_par = r_par = par

    # left context
    l_cxt = par_text[lc_start:start]
    if lc_start == 0:
        while len(l_cxt) < context_len and l_par > 0:
            l_par -= 1
            par_text_ = paragraphs[l_par]
            # take as much text as available and still within context_len
            s = max(0, len(par_text_) - (context_len - len(l_cxt)))
            l_cxt = par_text_[s:] + l_cxt
            shifts.insert(0, len(par_text_[s:]))
    text = l_cxt + text

    # right context
    r_cxt = par_text[end:lc_end]
    if lc_end == len(par_text) - 1:
        while len(r_cxt) < context_len and r_par < len(paragraphs):
            r_par += 1
            par_text_ = paragraphs[r_par]
            # take as much text as available and still within context_len
            e = min(len(par_text_), context_len - len(r_cxt))
            r_cxt = r_cxt + par_text_[:e]
            shifts.append(len(par_text_))
    text = text + r_cxt

    # Final shifts, the shift for each paragraph is the length of the preceding paragraphs, except
    # for the first one, where the shift is negative and based on how much of the start we cut off.
    shifts = [-s] + list(accumulate(shifts))

    # retrieve mentions
    anchor_pars = mentioner['anchors']['paragraph_id']
    # search for anchors within the paragraphs we used
    i = bisect_left(anchor_pars, l_par)
    j = bisect_left(anchor_pars, r_par + 1)

    mentions = []
    for anchor_idx in range(i, j):
        anchor = {key: values[anchor_idx] for key, values in mentioner['anchors'].items()}

        p = anchor['paragraph_id'] - l_par
        start = anchor['start'] + shifts[p]
        end = anchor['end'] + shifts[p]
        if start < 0 or end >= len(text):
            continue

        mention = {
            "start_char": start, "end_char": end,
            "mentioned_wikipedia_id": anchor['wikipedia_id'],
        }

        if 'mentioned_wikipedia_title' in config.optional_fields_to_add:
            mention['mentioned_wikipedia_title'] = anchor['wikipedia_title']

        if config.mentioned_features:
            mentioned = lookup_fn(anchor['wikipedia_id'])

            if 'mentioned_categories' in config.optional_fields_to_add:
                mention['mentioned_categories'] = mentioned['categories']

            # TODO mentioned_text

        mentions.append(mention)

    return text, mentions


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

        self.mentioned_features = \
            'mentioned_categories' in self.optional_fields_to_add
        #   'mentioned_text' in self.optional_fields_to_add


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

    def _create_indices(self, _, file_destination):
        entity_idxs = {}  # indices of entities
        entity_mentions = {}  # where entities are mentioned

        # build index of:
        #   1. which entities mention each other (`entity_mentions`),
        #   2. the index in the dataset of each entity (`entity_idxs`).
        print("Building indices.")
        for e_idx, entity in tqdm(enumerate(self.base_dataset), total=len(self.base_dataset)):
            entity_idxs[entity['wikipedia_id']] = e_idx

            # for each mention store that the current entity mentioned it
            mentions = [t[-1] for t in zip(*entity['anchors'].values())]
            for anchor_idx, mention_wiki_id in enumerate(mentions):
                if mention_wiki_id not in entity_mentions:
                    entity_mentions[mention_wiki_id] = []

                entity_mentions[mention_wiki_id].append(
                    (entity['wikipedia_id'], anchor_idx)
                )

        with open(file_destination, 'wb') as file:
            pickle.dump((entity_idxs, entity_mentions), file=file)

    def _split_generators(self, dl_manager: DownloadManager):
        self.base_dataset = load_dataset(
            'kilt_wikipedia',
            split='full',
        )

        if self.config.shuffle_base_dataset:
            self.base_dataset = self.base_dataset.shuffle(
                seed=self.config.shuffle_base_dataset_seed
            )

        # abuse `download_custom` to create indices with caching
        index_file = dl_manager.download_custom(
            url_or_urls="fake://kilt4el.index",
            custom_download=self._create_indices
        )
        with open(index_file, 'rb') as file:
            self.indices = pickle.load(file)

        return [
            datasets.SplitGenerator(name="full", gen_kwargs={})
        ]

    def _generate_examples(self):
        entity_idxs, entity_mentions = self.indices
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
                    mentioner, lambda id: self.base_dataset[entity_idxs[id]], anchor, self.config
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

        # Now yield the mentions.
        for entity in self.base_dataset:
            if entity['wikipedia_id'] in entity_mentions:
                if stop_condition():
                    print("Reached configured maximum number of samples, stopping generation.")
                    return
                yield from new_samples(entity['wikipedia_id'])

        print(f"All mentions have been generated. \n"
              f"Last mention index: {m_idx}. \n")


if __name__ == "__main__":
    from pprint import pprint

    # load a dataset
    dataset = load_dataset(
        __file__,
        optional_fields_to_add={
            'mentioning_wikipedia_id',
            'mentioning_wikipedia_title',
            'mentioned_wikipedia_title',
            'mentioned_categories',
        },
        max_samples=1000
    )

    # print some samples
    for i, test in enumerate(dataset['full']):
        print(i)
        pprint(test)
        print()
        if i >= 9:
            break
