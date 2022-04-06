"""
Copyright 2021 Shahrukh khan
Copyright 2022 Stefan F. Schouten

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from itertools import cycle, islice

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import numpy as np

from trainer import Trainer


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task data loaders.
    """

    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class SizeProportionalMTDL(MultitaskDataloader):

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])


class EvenMTDL(MultitaskDataloader):

    def __len__(self):
        smallest = min(self.num_batches_dict.values())
        return 2 * smallest

    def __iter__(self):
        task_choice_cycle = islice(cycle(range(len(self.task_name_list))), len(self))
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_cycle:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])


class MultitaskTrainer(Trainer):

    def __init__(self, *args, multitask_dataloader_type=SizeProportionalMTDL, **kwargs):
        """
        Args:
            data_collator:
                dictionary of the form {'eval': ..., 'train': {'task1': ..., 'task2': ...,}}
        """
        if 'data_collator' in kwargs:
            data_collators = kwargs['data_collator']
            self.train_data_collators = data_collators['train']
            kwargs['data_collator'] = data_collators['eval']
        else:
            print("WARNING: using default collator for each task.")

        super().__init__(*args, **kwargs)
        self.multitask_dataloader_type = multitask_dataloader_type

    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.train_data_collators[task_name],
        )
        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return self.multitask_dataloader_type(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.train_dataset.items()
            }
        )
