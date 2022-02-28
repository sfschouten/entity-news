import datetime
import os
import json


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