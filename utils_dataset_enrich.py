from typing import Callable

from datasets import Dataset
from datasets.fingerprint import hashregister, Hasher
from transformers import Pipeline


enrich_fn_hash_object_store = {}


@hashregister(type(hashregister))  # register on the `function` type
def hash_pipeline(hasher: Hasher, value):
    if value in enrich_fn_hash_object_store:
        return hasher.hash_default(enrich_fn_hash_object_store[value])
    else:
        return hasher.hash_default(value)


def enrich_dataset(dataset: Dataset, pipe: Pipeline, process_fn: Callable, pipe_column=0,
                   pipe_kwargs=None, map_kwargs=None):
    """
    Add predictions from a pipeline to a dataset.

    Uses custom `hash_pipeline` function above to ensure proper caching.

    Args:
        dataset: The dataset to enrich.
        pipe: The pipeline whose results will enrich the dataset.
        process_fn: A function for processing the results of the pipeline, this function should
            return an object that the `map` function can deal with.
        pipe_column (optional): The dataset column that will be passed to the pipeline.
        pipe_kwargs (optional): kwargs passed to the pipeline.
        map_kwargs (optional): kwargs passed to map.
    """

    if pipe_kwargs is None:
        pipe_kwargs = {}
    if map_kwargs is None:
        map_kwargs = {}

    def enrich_fn(*columns):
        results = pipe(columns[pipe_column], **pipe_kwargs)
        return process_fn(columns, results)

    # remove parameters that do not influence predictions
    del pipe_kwargs['batch_size']

    # store parameters that do influence predictions, which will be used to calculate the hash
    enrich_fn_hash_object_store[enrich_fn] = {
        'pipe_model_name_or_path': pipe.model.name_or_path,
        'tokenizer': pipe.tokenizer,
        'dataset_columns': pipe_column,
        'process_fn': process_fn,
        'pipe_kwargs': pipe_kwargs,
    }

    # By explicitly setting a set of input_columns we prevent the enrich_fn from getting wrapped,
    # which would prevent `hash_pipeline` from catching it.
    input_columns = dataset.column_names
    if isinstance(input_columns, dict):
        input_columns = next(iter(input_columns.values()))
    return dataset.map(enrich_fn, input_columns=input_columns, **map_kwargs)
