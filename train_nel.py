import argparse

import datasets
import numpy as np
import scipy.sparse as sp_sparse
import pandas as pd
import transformers

from datasets import load_dataset, DatasetDict, ClassLabel, Dataset
from transformers import AutoTokenizer, BatchEncoding, TrainingArguments, Trainer, EarlyStoppingCallback, \
    TrainerCallback, TrainerState, TrainerControl, AdamW

import wandb

import dataset_el_wiki
import utils
from utils import create_run_folder_and_config_dict, create_or_load_versatile_model, train_versatile
from data_collator import DataCollatorForTokenClassification
from modeling_entity_linking import EntityLinking

TASK_KEY = 'nel'
LABELS_KEY = f'{TASK_KEY}_labels'


def kilt_for_el_dataset(config, tokenizer, base_dataset, skip_labels=False):
    def construct_labels(example, batch_encoding: BatchEncoding):
        def warn(s_t, e_t, s_c, e_c):
            print(f"\nWARNING: "
                  f"\nstart_token={s_t}, end_token={e_t} "
                  f"\nstart_char={s_c}, end_char={e_c} "
                  f"\ntext:\n {example['mentioning_text']}")

        labels = [0] * len(batch_encoding['input_ids'])
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
                labels[t] = int(e)

        batch_encoding[LABELS_KEY] = labels
        return batch_encoding

    # tokenize
    if not skip_labels:
        tokenized_dataset = base_dataset.map(
            lambda example: construct_labels(
                example,
                tokenizer(example['mentioning_text'], truncation=True)
            ), batched=False
        ).remove_columns(['mentioning_text', 'mentions'])
    else:
        tokenized_dataset = base_dataset.map(
            lambda example: tokenizer(example['mentioning_text'], truncation=True)
        ).remove_columns(['mentioning_text', 'mentions'])

    return tokenized_dataset


def full_kilt(config, tokenizer):
    kwargs = {}
    if config['full_kilt_dataset_size']:
        kwargs['max_samples'] = config['full_kilt_dataset_size']
    if config['full_kilt_max_context_length']:
        kwargs['max_mention_context_length'] = config['full_kilt_max_context_length']
    if config['full_kilt_minimum_nr_mentions']:
        kwargs['minimum_mentions'] = config['full_kilt_minimum_nr_mentions']
    dataset = load_dataset(dataset_el_wiki.__file__, split='full', **kwargs)
    dataset = kilt_for_el_dataset(config, tokenizer, dataset)

    lengths = dataset.map(lambda sample: {'length': len(sample['input_ids'])}, batched=False)['length']
    print(f'Longest sample is {np.max(lengths)} tokens long.')

    train_eval = dataset.train_test_split(test_size=0.01)
    valid_test = train_eval['test'].train_test_split(test_size=0.5)
    kilt_dataset = DatasetDict({
        'train': train_eval['train'],
        'validation': valid_test['train'],
        'test': valid_test['test']
    })
    return kilt_dataset


def kilt_aidayago(config, tokenizer):
    def process_split(split):
        df = pd.DataFrame(split)

        df[['article_id', 'mention_id']] = df.apply(
            lambda row: row['id'].split('_')[1].split(':'),
            result_type='expand', axis=1,
        )

        START_TOKEN, END_TOKEN = "[START_ENT] ", "[END_ENT] "

        def aggregate(article):
            text = next(iter(article.input)).replace(START_TOKEN, '').replace(END_TOKEN, '')
            start = article.input.apply(lambda x: x.find(START_TOKEN))
            end = article.input.apply(lambda x: x.find(END_TOKEN) - len(START_TOKEN))
            wikipedia_id = article.output.apply(lambda y: y[0]['provenance'][0]['wikipedia_id'] if y else None)
            wikipedia_title = article.output.apply(lambda y: y[0]['answer'] if y else None)
            frame = pd.DataFrame({
                'article_id': next(iter(article.article_id)),
                'text': text,
                'start': start,
                'end': end,
            })
            frame['mentions'] = [{
                'start': list(article.input.apply(lambda x: x.find(START_TOKEN))),
                'end': list(article.input.apply(lambda x: x.find(END_TOKEN) - len(START_TOKEN))),
                'paragraph_id': [0] * len(article.input),
                'wikipedia_id': list(wikipedia_id),
                'wikipedia_title': list(wikipedia_title),
            }] * len(article.input)
            return frame

        grouped = df.groupby(by='article_id').apply(aggregate)

        def apply_mention_extractor(x):
            text, mentions = dataset_el_wiki.mention_extractor(
                mention_start_c=x['start'],
                mention_end_c=x['end'],
                mention_par_idx=0,
                mentioning_page_paragraphs=[x['text']],
                mentioning_page_mentions=x['mentions'],
                mentioned_page_features_to_add={'mentioned_wikipedia_title'}
            )
            mentions = utils.list_of_dicts_to_dict_of_lists(mentions)
            return text, mentions

        result = grouped.apply(apply_mention_extractor, result_type='expand', axis=1)
        result.columns = ['mentioning_text', 'mentions']

        split = Dataset.from_pandas(
            result,
            features=datasets.Features({
                "mentioning_text": datasets.Value("string"),
                "mentions": datasets.features.Sequence({
                    "start_char": datasets.Value("int16"),
                    "end_char": datasets.Value("int16"),
                    "mentioned_wikipedia_id": datasets.Value("string"),
                    "mentioned_wikipedia_title": datasets.Value("string")
                })
            })
        )
        return split

    aidayago = load_dataset("kilt_tasks", name="aidayago2")
    aidayago = DatasetDict({
        key: kilt_for_el_dataset(config, tokenizer, process_split(split), skip_labels=(key == 'test'))
        for key, split in aidayago.items()
    })
    return aidayago


def compute_nel_metrics(eval_pred):
    (logits, top_k_idxs), labels = eval_pred
    _, _, K = logits.shape
    logits = logits[logits != -100].reshape(-1, K)
    top_k_idxs = top_k_idxs[top_k_idxs >= 0].reshape(-1, K)

    entity_mask = labels[labels >= 0] > 0                                                       # Et
    other_mask = labels[labels >= 0] == 0                                                       # Ot
    entity_logits = logits[entity_mask].reshape(-1, K)                                          # Et x K
    other_logits = logits[other_mask].reshape(-1, K)                                            # Ot x K

    # = Group tokens by entity =
    # first find transitions between sequences of tokens with the same label
    _labels1 = np.concatenate((labels, labels[:, -2:-1]), axis=-1)
    _labels2 = np.concatenate((labels[:, 0:1], labels), axis=-1)
    transitions = (_labels1 != _labels2)[:, :-1]
    # only keep transitions into sequences of tokens labelled as an entity (label > 0)
    entity_transitions = np.bitwise_and(transitions, labels > 0)
    # use cumsum to number entities and only keep tokens labelled as entity
    entity_numbering = entity_transitions.flatten().cumsum()
    entity_numbering = entity_numbering[labels.flatten() > 0]
    entity_onehot = sp_sparse.eye(np.max(entity_numbering)+1, format='csr')[entity_numbering].astype(bool)[:, 1:]
    _, E = entity_onehot.shape

    # predicted entities (as indices into top-k)
    entity_preds = entity_logits.argmax(axis=-1)                                                # Et
    other_preds = other_logits.argmax(axis=-1)                                                  # Ot
    # predicted entities (as entity indices)
    entity_preds_ = np.take_along_axis(top_k_idxs[entity_mask].reshape(-1, K), entity_preds[:, None], axis=1).squeeze()
    other_preds_ = np.take_along_axis(top_k_idxs[other_mask].reshape(-1, K), other_preds[:, None], axis=1).squeeze()
    entity_correct = entity_preds == 0                                                          # Et

    # how many entities we predicted in total
    nr_predict = (entity_preds_ > 0).sum() + (other_preds_ > 0).sum()

    # which entity-tokens are correct
    entity_correct = entity_onehot.multiply(entity_correct[:, None])                            # Et x E
    # if at least one of the entity-tokens of the entities is correct
    entity_weak = entity_correct.sum(axis=0) > 0                                                # E
    # if all the entity-tokens of the entities are correct
    entity_strong = entity_correct.sum(axis=0) == entity_onehot.sum(axis=0)                     # E

    def metrics(nr_correct):
        nr_grtruth = E
        precision = nr_correct / nr_predict if nr_predict > 0 else float('inf')
        recall = nr_correct / nr_grtruth
        f1 = 2 * precision * recall / (precision + recall) if nr_predict > 0 else float('inf')
        return precision, recall, f1

    p_weak, r_weak, f1_weak = metrics(entity_weak.sum())
    p_strong, r_strong, f1_strong = metrics(entity_strong.sum())

    result = {
        'Precision_W': p_weak,      'Recall_W': r_weak,     'F1_W': f1_weak,
        'Precision_S': p_strong,    'Recall_S': r_strong,   'F1_S': f1_strong,
    }
    return result


DATASET_LOADERS = {
    'full_kilt': full_kilt,
    'aidayago': kilt_aidayago,
}


def train_entity_linking(cli_config):
    wandb.init(project='entity-news', tags=['NEL'])

    tokenizer = AutoTokenizer.from_pretrained(cli_config['model'])

    datasets = {}
    necessary_datasets = {cli_config[option] for option in ['train_dataset', 'valid_dataset']} \
                       | {key for key in cli_config['test_dataset']}
    for key in necessary_datasets:
        loader = DATASET_LOADERS[key]
        datasets[key] = loader(cli_config, tokenizer)

    train_set = datasets[cli_config['train_dataset']]['train']
    valid_set = datasets[cli_config['valid_dataset']]['validation']

    # load model
    head_id = cli_config['head_id']
    heads = {head_id: (1., EntityLinking)}
    model = create_or_load_versatile_model(
        cli_config, {
            f'{head_id}_attach_layer': cli_config['head_attach_layer'],
        },
        heads
    )

    # Initialize entity embeddings.
    model.heads[cli_config['head_id']].extend_embedding(train_set)
    # Optionally, initialize embeddings for valid/test set as well (metrics will no longer be InKB)
    # TODO make configurable
    # model.heads[cli_config['head_id']].extend_embedding(valid_set)

    training_args = TrainingArguments(
        cli_config['run_path'],
        fp16=True,
        save_total_limit=5,
        label_names=[LABELS_KEY],
        load_best_model_at_end=True,
        remove_unused_columns=False,
        lr_scheduler_type='constant_with_warmup',
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

    grouped_parameters = [
        {
            "params": model.heads.parameters(),
            "initial_lr": 0.01,
        },
        {
            "params": model.base_model.parameters(),
            "initial_lr": 5e-5,
        }
    ]

    optimizer = AdamW(params=grouped_parameters)
    lr_scheduler = transformers.get_constant_schedule_with_warmup(
        optimizer, cli_config['warmup_steps'], last_epoch=cli_config['max_nr_epochs'])

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
            label_name=LABELS_KEY
        ),
        callbacks=callbacks,
        optimizers=(optimizer, lr_scheduler)
    )

    if cli_config['probing'] or cli_config['frozen_until']:
        for param in model.base_model.parameters():
            param.requires_grad = False

    # ignore hidden states in output, prevent OOM during eval
    hidden_state_keys = [f'{key}_hidden_states' for key in heads.keys()]

    train_versatile(cli_config, trainer, eval_ignore=hidden_state_keys)

    key = 'test' if cli_config['eval_on_test'] else 'validation'
    for test_set in cli_config['test_dataset']:
        if test_set == 'full_kilt':
            print(f"Evaluating on KILT wikipedia {key} split.")
            result = trainer.evaluate(
                datasets[test_set][key],
                metric_key_prefix=f'{key}_kilt',
                ignore_keys=hidden_state_keys,
            )
            print(result)
        elif test_set == 'aidayago' and key != 'test':
            pass  #TODO


def train_entity_linking_argparse(parser: argparse.ArgumentParser):
    parser = utils.base_train_argparse(parser)

    parser.add_argument('--probing', action='store_true', help="If the base model's weights should remain frozen.")
    parser.add_argument('--frozen_until', default=4, type=int, help="Keep base model's weights frozen until this "
                                                                    "epoch.")

    parser.add_argument('--head_id', default='nel-0', type=str)
    parser.add_argument('--head_attach_layer', default=-1, type=int)

    # dataset
    DATASETS = ['full_kilt', 'aidayago']
    parser.add_argument('--train_dataset', choices=DATASETS, default='full_kilt')
    parser.add_argument('--valid_dataset', choices=DATASETS, default='full_kilt')
    parser.add_argument('--test_dataset', choices=DATASETS, default=['aidayago'], nargs='*')

    parser.add_argument('--full_kilt_dataset_size', default=None, type=int)
    parser.add_argument('--full_kilt_max_context_length', default=300, type=int,
                        help="Maximum number of characters in a mention's context.")
    parser.add_argument('--full_kilt_minimum_nr_mentions', default=10, type=int)

    parser.add_argument('--eval_strategy', default='steps', type=str)
    parser.add_argument('--eval_frequency', default=500, type=int)
    parser.add_argument('--eval_metric', default='F1_W', type=str)
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
