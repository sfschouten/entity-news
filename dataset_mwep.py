import sys
import os
import pickle
import random
import json
import hashlib

from dataclasses import dataclass

import datasets
from datasets import load_dataset, DownloadManager, DatasetInfo


@dataclass()
class MWEPBuilderConfig(datasets.BuilderConfig):
    mwep_path: str = None
    mwep_event_types_path: str = None

    split_level: str = 'incident'
    eval_split_size_abs: int = None
    eval_split_size_rel: float = 0.1

    def __post_init__(self):
        if self.mwep_path is None:
            raise ValueError("Loading an MWEP dataset requires you specify the path to MWEP.")

        self.mwep_cls_mod_loc = os.path.join(self.mwep_path)

        if self.mwep_event_types_path is None:
            self.mwep_event_types_path = os.path.join(self.data_dir, 'event_types.txt')

        self.mwep_type2inc_index_path = os.path.join(self.data_dir, 'json/type2inc_index.json')

        with open(self.mwep_event_types_path, 'r') as f:
            self.event_types = f.readlines()

        with open(self.mwep_type2inc_index_path, 'r') as f:
            self.type2inc = json.load(f)
            self.incidents = sorted(set([inc for incs in self.type2inc.values() for inc in incs]))


class MWEPDatasetBuilder(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = MWEPBuilderConfig
    RANDOM_SEED = 19930729

    def _info(self) -> DatasetInfo:
        return datasets.DatasetInfo(
            description="TODO",
            features=datasets.Features({
                "uri": datasets.Value("string"),
                "content": datasets.Value("string"),
                "incident": {
                    "wdt_id": datasets.ClassLabel(names=self.config.incidents),
                    "incident_type": datasets.ClassLabel(
                        names_file=self.config.mwep_event_types_path),
                    "extra_info": {
                        "sem:hasPlace": datasets.Value("string"),
                        "sem:hasTimeStamp": datasets.Value("string")
                    },
                }
            }),
            supervised_keys=None,
            homepage="TODO",
            citation="TODO"
        )

    def _split_generators(self, dl_manager: DownloadManager):
        # set seed
        random.seed(self.RANDOM_SEED)

        # add MWEP to path, so classes.py can be used to unpickle
        sys.path.append(self.config.mwep_cls_mod_loc)

        # obtain data
        if not self.config.data_dir:
            # TODO: download
            data_dir = None
            raise NotImplementedError()
        else:
            data_dir = os.path.join(self.config.data_dir, 'bin/')

        # load data
        collections_by_file = {}
        for file in sorted(os.listdir(data_dir)):
            if not file.endswith(',pilot.bin'):
                continue

            path = os.path.join(data_dir, file)
            with open(path, 'rb') as pickle_file:
                collection = pickle.load(pickle_file)
                collection.incidents = sorted(collection.incidents, key=lambda x: x.wdt_id)
                collections_by_file[file] = collection
        self.collections_by_file = collections_by_file

        # identify multi-label incidents
        id_sets_by_type = {
            file: set(
                incident.wdt_id for incident in collections_by_file[file].incidents
            ) for file in collections_by_file.keys()
        }

        multi_label_incidents = set()
        pairs_done = set()
        for a, a_set in sorted(id_sets_by_type.items()):
            for b, b_set in sorted(id_sets_by_type.items()):
                pair = frozenset([a, b])
                if pair in pairs_done or a == b:
                    continue
                pairs_done.add(frozenset([a, b]))

                overlap = a_set & b_set
                if len(overlap) > 0:
                    print(f"Between {a} and {b} there was an overlap of {len(overlap)} incidents, "
                          f"which will be filtered out.")
                    multi_label_incidents |= overlap
        print(f"Identified a total of {len(multi_label_incidents)} multi-label incidents.")

        train_idxs = set()
        valid_idxs = set()
        test_idxs = set()
        for file, collection in sorted(self.collections_by_file.items()):
            # TODO keep wikipedia pages?
            c_idxs = set(
                (file, inc_i, txt_i)
                for inc_i, inc in enumerate(collection.incidents)
                for txt_i, _ in enumerate(inc.reference_texts)
                if inc.wdt_id not in multi_label_incidents      # TODO look into multi-label?
            )

            if self.config.eval_split_size_abs is not None:
                eval_split_size = self.config.eval_split_size_abs
            elif self.config.eval_split_size_rel is not None:
                eval_split_size = int(self.config.eval_split_size_rel * len(c_idxs))
            else:
                raise ValueError('Either an absolute or relative split size must be specified.')

            print(f"For {file} we have {len(c_idxs)} articles, targeting eval split size of "
                  f"at least {eval_split_size}")

            def article_level_split():
                nonlocal c_idxs
                c_test_idxs = random.sample(c_idxs, eval_split_size)
                c_idxs -= set(c_test_idxs)
                c_valid_idxs = random.sample(c_idxs, eval_split_size)
                c_idxs -= set(c_valid_idxs)

                train_idxs.update(c_idxs)
                valid_idxs.update(c_valid_idxs)
                test_idxs.update(c_test_idxs)

            def incident_level_split():
                nonlocal c_idxs
                c_inc_idxs = list(i for i, inc in enumerate(collection.incidents)
                                  if inc.wdt_id not in multi_label_incidents)
                random.shuffle(c_inc_idxs)

                def split_off_eval():
                    c_eval_idxs = set()
                    while len(c_eval_idxs) < eval_split_size:
                        to_add = c_inc_idxs.pop()
                        c_eval_idxs.update(set(
                            (f, inc_i, txt_i) for (f, inc_i, txt_i) in c_idxs if inc_i == to_add
                        ))
                    return c_eval_idxs

                # split off validation and test sets
                v = split_off_eval()
                t = split_off_eval()
                print(f'The valid/test split sizes for this incident are: {len(v)}/{len(t)}.')
                valid_idxs.update(v)
                test_idxs.update(t)

                # add rest as training
                c_inc_idxs = set(c_inc_idxs)
                train_idxs.update(set(
                    (f, inc_i, txt_i) for (f, inc_i, txt_i) in c_idxs if inc_i in c_inc_idxs
                ))

            if self.config.split_level == 'article':
                article_level_split()
            elif self.config.split_level == 'incident':
                incident_level_split()
            elif self.config.split_level == 'none':
                return [
                    datasets.SplitGenerator(
                        name='full',
                        gen_kwargs={'idxs': c_idxs}
                    )
                ]
            else:
                raise ValueError('Invalid split_level!')

        def hashable_split_repr(split_idxs):
            return sorted(tuple(
                (
                    self.collections_by_file[file].incidents[inc_i].wdt_id,
                    self.collections_by_file[file].incidents[inc_i].reference_texts[txt_i].uri
                )
                for file, inc_i, txt_i in split_idxs
            ))

        hash_fn = lambda x: hashlib.sha256(str(x).encode("utf-8")).hexdigest()
        print(f"Hash of multi-label incidents: {hash_fn(sorted(multi_label_incidents))}")
        print(f"Hash of train-split: {hash_fn(hashable_split_repr(train_idxs))}")
        print(f"Hash of validation-split: {hash_fn(hashable_split_repr(valid_idxs))}")
        print(f"Hash of test-split: {hash_fn(hashable_split_repr(test_idxs))}\n")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'idxs': train_idxs}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={'idxs': valid_idxs}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={'idxs': test_idxs}
            ),
        ]

    def _generate_examples(self, **kwargs):
        for key in kwargs['idxs']:
            file, inc_i, txt_i = key
            inc = self.collections_by_file[file].incidents[inc_i]
            txt = inc.reference_texts[txt_i]
            yield str(key), {
                "uri": txt.uri,
                "content": txt.content,
                "incident": {
                    "wdt_id": inc.wdt_id,
                    "incident_type": inc.incident_type,
                    "extra_info": inc.extra_info
                }
            }


if __name__ == "__main__":
    # disable caching for this test code
    datasets.disable_caching()

    # load a dataset
    dataset = load_dataset(
        __file__,
        #data_dir="../data/minimal",
        #data_dir="../data/medium",
        data_dir="../data/medium_plus",
        mwep_path="../mwep",
        split_level='incident',
        eval_split_size_rel=0.1,
    )

    # print some samples
    import pprint
    test = dataset['test'].select(range(10)).flatten()
    test.map(lambda x: pprint.pprint(x))

    # tokenize and print statistics of nr of tokens
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')

    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples['content'], padding='max_length', truncation=False),
        batched=True
    ).remove_columns(['content'])

    from collections import Counter
    bins = Counter()
    BIN_SIZE = 16


    def count(example):
        bin_ = sum(example['attention_mask']) // BIN_SIZE
        bins[bin_] += 1

    tokenized_dataset.map(count)

    dataset_size = sum(dataset.num_rows.values())

    last_count = 0
    print(" bin |  #  |   cum |       %")
    print("----------------------------")
    for i in sorted(bins.keys()):
        last_count += bins[i]
        print(
            f"{BIN_SIZE*i:>4} | "
            f"{bins[i]:>3} | "
            f"{last_count:>5} | "
            f"{100*last_count/dataset_size:>6.2f}%"
        )
