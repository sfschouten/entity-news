from typing import TypeVar, Type, List, Dict, Optional, Tuple, Generic

import torch
import torch.nn as nn
from torch import FloatTensor
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput
from transformers.file_utils import ModelOutput


class Task(nn.Module):

    def __init__(self, key, config):
        super().__init__()
        self.key = key
        self.config = config

    def extract_kwargs(self, kwargs):
        raise NotImplementedError

    def forward(self, base_outputs, *args):
        raise NotImplementedError


class TokenClassification(Task):

    def __init__(self, key, config):
        super().__init__(key, config)
        config_dict = config.to_dict()
        self.num_labels = config_dict.get(f'{key}_num_labels', config.num_labels)
        self.attach_layer = config_dict.get(f'{key}_attach_layer', -1)

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def extract_kwargs(self, kwargs):
        labels = kwargs.pop(f'{self.key}_labels', None)
        return_dict = kwargs.get('return_dict', None)
        return labels, return_dict

    def forward(self, base_outputs, *args):
        labels, return_dict = args
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output = base_outputs[1][self.attach_layer]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + base_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=base_outputs.hidden_states,
            attentions=base_outputs.attentions,
        )


class SequenceClassification(Task):

    def __init__(self, key, config):
        super().__init__(key, config)
        config_dict = config.to_dict()
        self.num_labels = config_dict.get(f'{key}_num_labels', config.num_labels)
        self.attach_layer = config_dict.get(f'{key}_attach_layer', -1)

        self.dropout = nn.Dropout(config.dropout)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, self.num_labels)

    def extract_kwargs(self, kwargs):
        labels = kwargs.pop(f'{self.key}_labels', None)
        return_dict = kwargs.get('return_dict', None)
        return labels, return_dict

    def forward(self, base_outputs, *args):
        labels, return_dict = args
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_state = base_outputs[1][self.attach_layer]       # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]                      # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)      # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)                # (bs, dim)
        pooled_output = self.dropout(pooled_output)             # (bs, dim)
        logits = self.classifier(pooled_output)                 # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                        labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + base_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=base_outputs.hidden_states,
            attentions=base_outputs.attentions,
        )


class MultipleTasksOutput(ModelOutput):
    loss: Optional[FloatTensor] = None


T = TypeVar('T', bound=PreTrainedModel)


def create_multitask_class(model_cls: Type[T]):

    class ModelForMultipleTasks(model_cls):

        def __init__(self, transformer_model: model_cls,
                     tasks: List[Tuple[str, float, Type[Task]]]):
            config = transformer_model.config
            super().__init__(config)

            self.transformer_model = transformer_model

            self.loss_weights = {key: weight for key, (weight, _) in tasks}
            self.normalizer = sum(self.loss_weights.values())
            self.tasks = nn.ModuleDict({key: type_(key, config) for key, (_, type_) in tasks})

            self.post_init()

        def forward(self, **kwargs):
            # allow tasks to extract arguments specific to them first
            task_args = {key: task.extract_kwargs(kwargs) for key, task in self.tasks.items()}

            # now forward the underlying transformer model
            outputs = self.transformer_model(**kwargs, output_hidden_states=True)

            # finally, forward the task-specific heads, passing the arguments extracted before
            all_results = {key: task(outputs, *task_args[key]) for key, task in self.tasks.items()}

            kwargs = {
                f"{task}_{key}": values
                for task, results in all_results.items()
                for key, values in results.items()
                if key != 'loss'
            }
            return MultipleTasksOutput(
                loss=sum(
                    self.loss_weights[task] * value.view(())
                    for task, results in all_results.items()
                    for key, value in results.items()
                    if key == 'loss' and value is not None
                ),
                **kwargs
            )

        def get_position_embeddings(self) -> nn.Embedding:
            """
            Returns the position embeddings
            """
            return self.base_model.get_position_embeddings()

        def resize_position_embeddings(self, new_num_position_embeddings: int):
            """
            Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.
            Arguments:
                new_num_position_embeddings (`int`):
                    The number of new position embedding matrix. If position embeddings are learned, increasing the size
                    will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                    end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                    size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                    the size will remove vectors from the end.
            """
            self.base_model.resize_position_embeddings(new_num_position_embeddings)

    return ModelForMultipleTasks
