import argparse
from functools import partial

import numpy as np
import wandb

from datasets import load_dataset, load_metric, DatasetDict
from transformers import AutoTokenizer, BatchEncoding, \
    TrainingArguments, Trainer, EarlyStoppingCallback

from data_collator import DataCollatorForTokenClassification
from modeling_versatile import TokenClassification
from utils import create_run_folder_and_config_dict, create_or_load_versatile_model, train_versatile
import dataset_el_wiki

from sklearn import metrics

I, O, B = 0, 1, 2


def kilt_for_er_dataset(config, tokenizer):
    def construct_iob_labels(example, batch_encoding: BatchEncoding):
        def warn(s_t, e_t, s_c, e_c):
            print(f"\nWARNING: "
                  f"\nstart_token={s_t}, end_token={e_t} "
                  f"\nstart_char={s_c}, end_char={e_c} "
                  f"\ntext:\n {example['mentioning_text']}")

        labels = [O] * len(batch_encoding['input_ids'])
        start_chars = example['mentions']['start_char']
        end_chars = example['mentions']['end_char']
        for start_char, end_char in zip(start_chars, end_chars):
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
            labels[start_token] = B
            for t in range(start_token + 1, end_token + 1):
                labels[t] = I

        batch_encoding['labels'] = labels
        return batch_encoding

    kwargs = {}
    if config['ner_dataset_size']:
        kwargs['max_samples'] = config['ner_dataset_size']
    dataset = load_dataset(
        dataset_el_wiki.__file__,
        split='full',
        **kwargs
    )

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

    train_eval = tokenized_dataset.train_test_split(test_size=0.01)
    valid_test = train_eval['test'].train_test_split(test_size=0.5)
    kilt_dataset = DatasetDict({
        'train': train_eval['train'],
        'validation': valid_test['train'],
        'test': valid_test['test']
    })

    return kilt_dataset


def conll2003_dataset(config, tokenizer):
    TAGS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    def labels(example, batch_encoding: BatchEncoding):
        words = batch_encoding.words()  # maps tokens to word indices
        labels = [TAGS[example['ner_tags'][w]][0] if w is not None else 'O' for w in words]
        labels = [{'I': I, 'O': O, 'B': B}[lbl] for lbl in labels]
        # If a word got split up in tokenizer, and had a 'B' tag,
        # replace second token's tag with 'I'.
        labels = [labels[0]] + [
            lbl if not (prev == lbl == B) else I
            for lbl, prev in zip(labels[1:], labels[0:-1])
        ]
        batch_encoding['labels'] = labels
        return batch_encoding

    dataset = load_dataset("conll2003")
    dataset = dataset.map(
        lambda example: labels(
            example,
            tokenizer(
                example['tokens'],
                is_split_into_words=True,
                truncation=True
            )
        ), batched=False,
    ).remove_columns(['tokens', 'id', 'pos_tags', 'chunk_tags', 'ner_tags'])
    return dataset


def compute_ner_metrics(seq_metric, eval_pred):
    def swap4lbl(ndarray):
        return [
            [{I: 'I', O: 'O', B: 'B'}[x.item()] for x in row if x.item() != -100]
            for row in ndarray
        ]

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    cf_labels = labels[labels != -100]
    cf_preds = preds[labels != -100]
    print()
    print(metrics.confusion_matrix(cf_labels, cf_preds))
    wandb.log({"nc_conf_mat": wandb.plot.confusion_matrix(
        y_true=cf_labels, preds=cf_preds, class_names=["I", "O", "B"])})

    labels = swap4lbl(labels)
    preds = swap4lbl(preds)
    preds = [pred[:len(lbl)] for pred, lbl in zip(preds, labels)]
    er_result = seq_metric.compute(predictions=preds, references=labels, scheme='IOB2')
    return er_result


def train_entity_recognition(cli_config):
    wandb.init(project='entity-news', tags=['NER'])

    tokenizer = AutoTokenizer.from_pretrained(cli_config['model'])

    any_dataset = [cli_config['train_dataset'], cli_config['valid_dataset']] \
                + cli_config['test_dataset']

    # kilt dataset
    if 'kilt' in any_dataset:
        kilt_dataset = kilt_for_er_dataset(cli_config, tokenizer)\
            .rename_column('labels', 'ner_labels')

    # conll2003 dataset
    if 'conll' in any_dataset:
        conll_dataset = conll2003_dataset(cli_config, tokenizer)\
            .rename_column('labels', 'ner_labels')

    if cli_config['train_dataset'] == 'kilt':
        train_set = kilt_dataset['train']
    else:
        train_set = conll_dataset['train']

    if cli_config['valid_dataset'] == 'kilt':
        valid_set = kilt_dataset['validation']
    else:
        valid_set = conll_dataset['validation']

    # load model
    head_id = cli_config['head_id']
    heads = {head_id: (1., TokenClassification)}
    model = create_or_load_versatile_model(
        cli_config, {
            f'{head_id}_num_labels': 3,
            f'{head_id}_attach_layer': cli_config['head_attach_layer'],
        },
        heads
    )

    training_args = TrainingArguments(
        cli_config['run_path'],
        fp16=True,
        num_train_epochs=cli_config['max_nr_epochs'],
        per_device_train_batch_size=cli_config['batch_size_train'],
        per_device_eval_batch_size=cli_config['batch_size_eval'],
        gradient_accumulation_steps=cli_config['gradient_acc_steps'],
        load_best_model_at_end=True,
        remove_unused_columns=False,
        label_names=["ner_labels"],
        metric_for_best_model=cli_config['eval_metric'],
        evaluation_strategy=cli_config['eval_strategy'],
        save_strategy=cli_config['eval_strategy'],
        eval_steps=cli_config['eval_frequency'],
        warmup_steps=cli_config['warmup_steps'],
        report_to=cli_config['report_to'],
        save_total_limit=5,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        compute_metrics=partial(compute_ner_metrics, load_metric('seqeval')),
        data_collator=DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            label_name='ner_labels'
        ),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cli_config['early_stopping_patience'])
        ]
    )

    if cli_config['probing']:
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
        if test_set == 'conll':
            print("Evaluating on CoNLL2003")
            result = trainer.evaluate(
                conll_dataset[key],
                metric_key_prefix=f'{key}_conll',
                ignore_keys=hidden_state_keys,
            )
            print(result)
        if test_set == 'mwep_w/ots-nerc':
            pass


def sanity_check_kilt_iob_labels(cli_config):
    tokenizer = AutoTokenizer.from_pretrained(cli_config['model'])

    # kilt dataset
    kilt_dataset = kilt_for_er_dataset(cli_config, tokenizer)
    kilt_data = kilt_dataset.select(list(range(10)))

    for sample in kilt_data:
        labels = sample['labels']
        labels = [{I: 'I', O: 'O', B: 'B'}[l] for l in labels]

        token_ids = sample['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        print(list(zip(tokens, labels)))

    exit()


if __name__ == "__main__":
    # parse cmdline arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--report_to', default=None, type=str)

    parser.add_argument('--runs_folder', default='runs')
    parser.add_argument('--run_name', default=None)

    parser.add_argument('--model', default="distilbert-base-cased")
    parser.add_argument('--probing', action='store_true')
    parser.add_argument('--head_attach_layer', default=-1, type=int)
    parser.add_argument('--head_id', default='ner-0', type=str)

    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--continue', action='store_true')
    parser.add_argument('--eval_only', action='store_true')

    # dataset
    parser.add_argument('--train_dataset', choices=['kilt', 'conll'], default='kilt')
    parser.add_argument('--valid_dataset', choices=['kilt', 'conll'], default='kilt')
    parser.add_argument('--test_dataset', choices=['kilt', 'conll'], default=['kilt', 'conll'],
                        nargs='*')

    parser.add_argument('--ner_dataset_size', default=None, type=int)

    parser.add_argument('--eval_strategy', default='steps', type=str)
    parser.add_argument('--eval_frequency', default=500, type=int)
    parser.add_argument('--eval_metric', default='overall_f1', type=str)
    parser.add_argument('--eval_on_test', action='store_true')

    # hyper-parameters
    parser.add_argument('--max_nr_epochs', default=100, type=int)
    parser.add_argument('--warmup_steps', default=2000, type=int)
    parser.add_argument('--early_stopping_patience', default=5, type=int)
    parser.add_argument('--batch_size_train', default=64, type=int)
    parser.add_argument('--batch_size_eval', default=64, type=int)
    parser.add_argument('--gradient_acc_steps', default=1, type=int)

    parser.add_argument('--do_sanity_checks', action='store_true')

    args = parser.parse_args()

    cli_config = create_run_folder_and_config_dict(args)

    if args.do_sanity_checks:
        sanity_check_kilt_iob_labels(cli_config)

    train_entity_recognition(cli_config)
