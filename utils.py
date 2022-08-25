import argparse
import datetime
import os
import json

from transformers import AutoConfig, AutoModel
from transformers.models.auto.auto_factory import _get_model_class

from modeling_versatile import create_versatile_class


def create_run_folder_and_config_dict(args):
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
    run_path = os.path.join(args.runs_folder, run_name)
    os.makedirs(run_path)

    # config dict
    config = {**vars(args), 'run_path': run_path}
    with open(os.path.join(run_path, 'config.json'), 'w') as fp:
        json.dump(config, fp)

    return config


def create_or_load_versatile_model(cli_config, config_additions, heads):
    base_config = AutoConfig.from_pretrained(cli_config['model'])
    base_model_cls = _get_model_class(base_config, AutoModel._model_mapping)
    cls = create_versatile_class(base_model_cls)

    if cli_config['checkpoint'] is not None:
        print(f"Loading checkpoint {cli_config['checkpoint']} for base of versatile model.")
        config = AutoConfig.from_pretrained(cli_config['checkpoint'])
        config.update(config_additions)
        model = cls.from_pretrained(cli_config['checkpoint'], heads.items(), config=config)
    elif cli_config['no_pretrained']:
        print(f"Randomly initializing the base of the versatile model.")
        base_config.update(config_additions)
        model = cls(base_config, heads.items())
    else:
        print(f"Using pretrained weights for the base of the versatile model.")
        base_config.update(config_additions)
        model = cls.from_pretrained(cli_config['model'], heads.items(), config=base_config)

    return model


def train_versatile(cli_config, trainer, eval_ignore=()):
    train_kwargs = {
        'ignore_keys_for_eval': eval_ignore
    }
    if cli_config['continue']:
        train_kwargs.update({
            'resume_from_checkpoint': cli_config['checkpoint']
        })

    if not cli_config['eval_only']:
        trainer.train(**train_kwargs)


def base_train_argparse(parser: argparse.ArgumentParser):
    parser.add_argument('--report_to', default=None, type=str)

    parser.add_argument('--runs_folder', default='runs')
    parser.add_argument('--run_name', default=None)

    parser.add_argument('--model', default="distilbert-base-cased")

    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--continue', action='store_true')
    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--eval_only', action='store_true')

    return parser
