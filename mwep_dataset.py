import sys
import os
import pickle
import random

from dataclasses import dataclass

import datasets
from datasets import load_dataset, DownloadManager, DatasetInfo


@dataclass()
class MWEPBuilderConfig(datasets.BuilderConfig):
    mwep_path: str = None
    mwep_event_types_path: str = None

    def __post_init__(self):
        if self.mwep_path is None:
            raise ValueError("Loading an MWEP dataset requires you specify the path to MWEP.")

        self.mwep_cls_mod_loc = os.path.join(self.mwep_path)

        if self.mwep_event_types_path is None:
            self.mwep_event_types_path = os.path.join(self.mwep_path, "config/event_types.txt")


class MWEPDatasetBuilder(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = MWEPBuilderConfig
    EVAL_SPLITS_SIZE = 500
    RANDOM_SEED = 19930729

    def _info(self) -> DatasetInfo:
        return datasets.DatasetInfo(
            description="TODO",
            features=datasets.Features({
                "uri": datasets.Value("string"),
                "content": datasets.Value("string"),
                "incident": {
                    "wdt_id": datasets.Value("string"),
                    # TODO: add back once https://github.com/huggingface/datasets/issues/3631 makes it into a release.
                    # "incident_type": datasets.ClassLabel(
                    #    names_file=self.config.mwep_event_types_path),
                    "incident_type": datasets.ClassLabel(
                        names=['Q350604', 'Q18515440', 'Q669262', 'Q7590']
                    ),
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
            data_dir = self.config.data_dir

        # load data
        self.collections_by_file = {}
        train_idxs = set()
        valid_idxs = set()
        test_idxs = set()
        for file in os.listdir(data_dir):
            if not file.endswith(',pilot.bin'):
                continue

            path = os.path.join(data_dir, file)
            with open(path, 'rb') as pickle_file:
                collection = pickle.load(pickle_file)
                collection.incidents = list(collection.incidents)
                self.collections_by_file[file] = collection

            # TODO keep wikipedia pages?
            c_idxs = set(
                (file, inc_i, txt_i)
                for inc_i, inc in enumerate(collection.incidents)
                for txt_i, _ in enumerate(inc.reference_texts)
            )
            c_test_idxs = random.sample(c_idxs, self.EVAL_SPLITS_SIZE)
            c_idxs -= set(c_test_idxs)
            c_valid_idxs = random.sample(c_idxs, self.EVAL_SPLITS_SIZE)
            c_idxs -= set(c_valid_idxs)

            train_idxs.update(c_idxs)
            valid_idxs.update(c_valid_idxs)
            test_idxs.update(c_test_idxs)

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
    datasets.set_caching_enabled(False)

    dataset = load_dataset(
        __file__,
        data_dir="/home/stefan/Projects/entity-assisted/data/minimal/bin",
        mwep_path="/home/stefan/Projects/entity-assisted/mwep",
    )

    import pprint
    test = dataset['test'].select(range(10)).flatten()
    test.map(lambda x: pprint.pprint(x))
