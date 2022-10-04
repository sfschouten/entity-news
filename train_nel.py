import argparse

import numpy as np

from datasets import load_dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer, BatchEncoding, TrainingArguments, Trainer, EarlyStoppingCallback, \
    TrainerCallback, TrainerState, TrainerControl

import wandb

import dataset_el_wiki
import utils
from utils import create_run_folder_and_config_dict, create_or_load_versatile_model, train_versatile
from data_collator import DataCollatorForTokenClassification
from modeling_entity_linking import EntityLinking


def kilt_for_el_dataset(config, tokenizer):
    def construct_iob_labels(example, batch_encoding: BatchEncoding, ):
        def warn(s_t, e_t, s_c, e_c):
            print(f"\nWARNING: "
                  f"\nstart_token={s_t}, end_token={e_t} "
                  f"\nstart_char={s_c}, end_char={e_c} "
                  f"\ntext:\n {example['mentioning_text']}")

        labels = ['O'] * len(batch_encoding['input_ids'])
        start_chars = example['mentions']['start_char']
        end_chars = example['mentions']['end_char']
        entities = example['mentions']['mentioned_wikipedia_id']
        for start_char, end_char, e in zip(start_chars, end_chars, entities):
            if start_char < 0 or end_char < 0:
                warn(-1, -1, start_char, end_char)
                continue

            sc, ec = start_char, end_char
            start_token = end_token = None
            while start_token is None and sc < end_char:
                start_token = batch_encoding.char_to_token(sc)
                sc += 1
            while end_token is None and ec > start_char:
                end_token = batch_encoding.char_to_token(ec - 1)
                ec -= 1
            if start_token is None or end_token is None:
                warn(start_token, end_token, start_char, end_char)
                continue

            for t in range(start_token, end_token + 1):
                labels[t] = e

        batch_encoding['labels'] = labels
        return batch_encoding

    kwargs = {}
    if config['nel_dataset_size']:
        kwargs['max_samples'] = config['nel_dataset_size']
    if config['nel_minimum_nr_mentions']:
        kwargs['minimum_mentions'] = config['nel_minimum_nr_mentions']
    dataset = load_dataset(
        dataset_el_wiki.__file__,
        split='full',
        **kwargs
    )

    import itertools
    label_names = list(set(itertools.chain.from_iterable(dataset.flatten()['mentions.mentioned_wikipedia_id'])))
    entity_label = ClassLabel(names=['O']+label_names)

    # tokenize
    tokenized_dataset = dataset.map(
        lambda example: construct_iob_labels(
            example,
            tokenizer(
                example['mentioning_text'],
                truncation=True
            )
        ), batched=False
    ).remove_columns(['mentioning_text', 'mentions'])

    tokenized_dataset = tokenized_dataset.map(
        lambda example: {'labels': entity_label.str2int(example['labels'])}, batched=False)

    train_eval = tokenized_dataset.train_test_split(test_size=0.01)
    valid_test = train_eval['test'].train_test_split(test_size=0.5)
    kilt_dataset = DatasetDict({
        'train': train_eval['train'],
        'validation': valid_test['train'],
        'test': valid_test['test']
    })

    return kilt_dataset, len(label_names)+1


def compute_nel_metrics(eval_pred):
    logits, labels = eval_pred                                              # B x N x K,  B x N
    B, N, K = logits.shape

    _labels1 = np.concatenate((labels, labels[:, -2:-1]), axis=-1)
    _labels2 = np.concatenate((labels[:, 0:1], labels), axis=-1)
    transitions = (_labels1 != _labels2)[:, :-1]
    entity_transitions = np.bitwise_and(transitions, labels > 0)
    entity_numbering = entity_transitions.flatten().cumsum()
    entity_numbering = entity_numbering[labels.flatten() > 0]
    entity_onehot = np.eye(np.max(entity_numbering)+1)[entity_numbering].astype(bool)[:, 1:]

    preds = logits.argmax(axis=-1)                                          # B x N
    correct = preds == K-1

    # TODO we need to know whether or not the predictions are 'O' or not

    # which entity-tokens are correct
    entity_correct = correct[labels > 0, np.newaxis] * entity_onehot
    # if at least one of the entity-tokens is correct
    entity_weak = entity_correct.any(axis=0)
    # if all the entity-tokens are correct
    entity_correct[~entity_onehot] = True   # set default value to True for `all` check.
    entity_strong = entity_correct.all(axis=0)

    nr_correct_weak = entity_weak.sum()
    nr_correct_strong = entity_strong.sum()

    nr_predict = (preds > 0).sum()  # TODO fix..
    nr_grtruth = len(entity_strong)
    precision_weak = nr_correct_weak / nr_predict if nr_predict > 0 else float('inf')
    precision_strong = nr_correct_strong / nr_predict if nr_predict > 0 else float('inf')
    recall_weak = nr_correct_weak / nr_grtruth
    recall_strong = nr_correct_strong / nr_grtruth
    f1_weak = 2 * precision_weak * recall_weak / (precision_weak + recall_weak) if nr_predict > 0 else float('nan')
    f1_strong = 2 * precision_strong * recall_strong / (precision_strong + recall_strong) if nr_predict > 0 else float('nan')

    result = {
        'Accuracy': correct.mean(),
        'Precision_W': precision_weak,
        'Recall_W': recall_weak,
        'F1_W': f1_weak,
        'Precision_S': precision_strong,
        'Recall_S': recall_strong,
        'F1_S': f1_strong,
    }
    return result


def train_entity_linking(cli_config):
    wandb.init(project='entity-news', tags=['NEL'])

    tokenizer = AutoTokenizer.from_pretrained(cli_config['model'])

    # kilt dataset
    kilt_dataset, nr_entities = kilt_for_el_dataset(cli_config, tokenizer)
    kilt_dataset = kilt_dataset.rename_column('labels', 'nel_labels')

    train_set = kilt_dataset['train']
    valid_set = kilt_dataset['validation']

    # load model
    head_id = cli_config['head_id']
    heads = {head_id: (1., EntityLinking)}
    model = create_or_load_versatile_model(
        cli_config, {
            f'{head_id}_attach_layer': cli_config['head_attach_layer'],
            f'{head_id}_nr_entities': nr_entities,
        },
        heads
    )

    training_args = TrainingArguments(
        cli_config['run_path'],
        fp16=True,
        save_total_limit=5,
        label_names=["nel_labels"],
        load_best_model_at_end=True,
        remove_unused_columns=False,
        num_train_epochs=cli_config['max_nr_epochs'],
        per_device_train_batch_size=cli_config['batch_size_train'],
        per_device_eval_batch_size=cli_config['batch_size_eval'],
        gradient_accumulation_steps=cli_config['gradient_acc_steps'],
        metric_for_best_model=cli_config['eval_metric'],
        evaluation_strategy=cli_config['eval_strategy'],
        save_strategy=cli_config['eval_strategy'],
        eval_steps=cli_config['eval_frequency'],
        logging_steps=cli_config['eval_frequency'],
        warmup_steps=cli_config['warmup_steps'],
        report_to=cli_config['report_to'],
    )

    class UnfreezeCallback(TrainerCallback):
        def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if state.epoch == cli_config['frozen_until']:
                for param in model.base_model.parameters():
                    param.requires_grad = True

    callbacks = []
    if cli_config['early_stopping_patience']:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=cli_config['early_stopping_patience'],
        ))
    if cli_config['frozen_until']:
        callbacks.append(UnfreezeCallback())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        compute_metrics=compute_nel_metrics,
        data_collator=DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            label_name='nel_labels'
        ),
        callbacks=callbacks,
    )

    if cli_config['probing'] or cli_config['frozen_until']:
        for param in model.base_model.parameters():
            param.requires_grad = False

    # ignore hidden states in output, prevent OOM during eval
    hidden_state_keys = [f'{key}_hidden_states' for key in heads.keys()]

    train_versatile(cli_config, trainer, eval_ignore=hidden_state_keys)

    key = 'test' if cli_config['eval_on_test'] else 'validation'
    for test_set in cli_config['test_dataset']:
        if test_set == 'kilt':
            print(f"Evaluating on KILT wikipedia {key} split.")
            result = trainer.evaluate(
                kilt_dataset[key],
                metric_key_prefix=f'{key}_kilt',
                ignore_keys=hidden_state_keys,
            )
            print(result)


def train_entity_linking_argparse(parser: argparse.ArgumentParser):
    parser = utils.base_train_argparse(parser)

    parser.add_argument('--probing', action='store_true', help="If the base model's weights should remain frozen.")
    parser.add_argument('--frozen_until', default=4, type=int, help="Keep base model's weights frozen until this "
                                                                    "epoch.")

    parser.add_argument('--head_id', default='nel-0', type=str)
    parser.add_argument('--head_attach_layer', default=-1, type=int)

    # dataset
    parser.add_argument('--train_dataset', choices=['kilt'], default='kilt')
    parser.add_argument('--valid_dataset', choices=['kilt'], default='kilt')
    parser.add_argument('--test_dataset', choices=['kilt'], default=['kilt'], nargs='*')

    parser.add_argument('--nel_dataset_size', default=None, type=int)
    parser.add_argument('--nel_minimum_nr_mentions', default=0, type=int)

    parser.add_argument('--eval_strategy', default='steps', type=str)
    parser.add_argument('--eval_frequency', default=500, type=int)
    parser.add_argument('--eval_metric', default='Accuracy', type=str)
    parser.add_argument('--eval_on_test', action='store_true')

    # hyper-parameters
    parser.add_argument('--max_nr_epochs', default=100, type=int)
    parser.add_argument('--warmup_steps', default=2000, type=int)
    parser.add_argument('--early_stopping_patience', default=0, type=int)
    parser.add_argument('--batch_size_train', default=64, type=int)
    parser.add_argument('--batch_size_eval', default=64, type=int)
    parser.add_argument('--gradient_acc_steps', default=1, type=int)

    return parser


if __name__ == "__main__":
    # parse cmdline arguments
    _parser = argparse.ArgumentParser()
    _parser = train_entity_linking_argparse(_parser)
    _args = _parser.parse_args()

    cli_config = create_run_folder_and_config_dict(_args)
    train_entity_linking(cli_config)
