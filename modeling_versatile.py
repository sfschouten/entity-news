from typing import TypeVar, Type, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import FloatTensor
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import PreTrainedModel, PretrainedConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput


def _versatile_dropout(config_dict, key):
    if f'{key}_dropout' in config_dict:
        dropout = config_dict[f'{key}_dropout']
    else:
        print("Dropout for SequenceClassificationHead not specified...")
        for key in config_dict.keys():
            if 'dropout' in key:
                dropout = config_dict[key]
                print(f"falling back to '{key}' with value {dropout}")
                break
        else:
            print("falling back to no dropout")
            dropout = 0
    return dropout


class Head(nn.Module):

    def __init__(self, key, config):
        super().__init__()
        self.key = key
        self.config = config
        self.task_key, self.head_idx = key.split('-')

    def extract_kwargs(self, kwargs):
        raise NotImplementedError

    def forward(self, base_outputs, *args):
        raise NotImplementedError


class TokenClassification(Head):

    def __init__(self, key, config):
        super().__init__(key, config)
        config_dict = config.to_dict()

        self.num_labels = config_dict.get(f'{key}_num_labels', config.num_labels)
        self.attach_layer = config_dict.get(f'{key}_attach_layer', -1)

        self.dropout = nn.Dropout(_versatile_dropout(config_dict, key))
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def extract_kwargs(self, kwargs):
        labels = kwargs.pop(f'{self.task_key}_labels', None)
        return_dict = kwargs.get('return_dict', None)
        return labels, return_dict

    def forward(self, base_outputs, *args):
        labels, return_dict = args
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output = base_outputs['hidden_states'][self.attach_layer]
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


class SequenceClassification(Head):

    def __init__(self, key, config):
        super().__init__(key, config)
        config_dict = config.to_dict()
        self.num_labels = config_dict.get(f'{key}_num_labels', config.num_labels)
        self.attach_layer = config_dict.get(f'{key}_attach_layer', -1)

        self.dropout = nn.Dropout(_versatile_dropout(config_dict, key))

        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def extract_kwargs(self, kwargs):
        labels = kwargs.pop(f'{self.task_key}_labels', None)
        return_dict = kwargs.get('return_dict', None)
        return labels, return_dict

    def forward(self, base_outputs, *args):
        labels, return_dict = args
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_state = base_outputs['hidden_states'][self.attach_layer]       # (bs, seq_len, dim)
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


class VersatileOutput(ModelOutput):
    loss: Optional[FloatTensor] = None


T = TypeVar('T', bound=PreTrainedModel)


def create_versatile_class(model_cls: Type[T]):

    class VersatileModelForAnyTasks(model_cls):

        def __init__(self, config: PretrainedConfig,
                     heads: List[Tuple[str, float, Type[Head]]]):
            super().__init__(config)

            setattr(self, model_cls.base_model_prefix, model_cls(config))

            self.loss_weights = {key: weight for key, (weight, _) in heads}
            self.normalizer = sum(self.loss_weights.values())
            self.heads = nn.ModuleDict({key: type_(key, config) for key, (_, type_) in heads})

            self.post_init()

        def forward(self, **kwargs):
            # allow tasks to extract arguments specific to them first
            task_args = {key: task.extract_kwargs(kwargs) for key, task in self.heads.items()}

            # now forward the underlying transformer model
            outputs = self.base_model(**kwargs, output_hidden_states=True)

            # finally, forward the task-specific heads, passing the arguments extracted before
            all_results = {key: task(outputs, *task_args[key]) for key, task in self.heads.items()}

            kwargs = {
                f"{task}_{key}": values
                for task, results in all_results.items()
                for key, values in results.items()
                if key != 'loss'
            }
            return VersatileOutput(
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

    return VersatileModelForAnyTasks
