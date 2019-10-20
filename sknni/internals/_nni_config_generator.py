""" This module generates various files needed by NNI command line """

import os
import json
import yaml

from absl import logging

def _generate_nni_config(nni_config, out_dir):
    # all we got to do is to update this dictionary
    # with missing items
    nni_config.update({
        'searchSpacePath' : 'search_space.json'
    })

    nni_config['trial'].update({
        'command' : 'python -m sknni.cli run-classification-experiment',
        'codeDir' : '.'
    })

    path_to_config_file = os.path.join(out_dir, 'config.yml')
    with open(path_to_config_file, 'w+') as outfile:
        yaml.dump(nni_config, outfile)

def _generate_search_space(search_space, out_dir):
    # we build the final dictionary
    # and automatically add the name of the estimator as the prefix
    # so that we can use the same names across various estimators
    complete_dict = {}
    for param in search_space:
        if param.values() is None:
            continue
        param_key = list(param.keys())[0]
        param_values = list(param.values())[0]
        for k,v in param_values.items():
            complete_dict.update({
                f'{param_key}_{k}' : v
            })

    path_to_search_space_file = os.path.join(out_dir, 'search_space.json')
    with open(path_to_search_space_file, 'w+') as outfile:
        json.dump(complete_dict, outfile)



def generate(experiment_spec, out_dir):
    logging.debug(f"Writing search_space.json to {out_dir} ..")
    _generate_search_space(experiment_spec['nniConfigSearchSpace'], out_dir)
    logging.debug(f"Writing config.yml to {out_dir} ..")
    _generate_nni_config(experiment_spec['nniConfig'], out_dir)

