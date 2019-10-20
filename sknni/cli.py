import os
import nni
import sys
import glob
import shutil

import yaml
import click

from absl import logging

from sknni.internals import get_class
from sknni.internals import PipelineBuilder
from sknni.internals import nni_config_generator

def _read_experiment_spec(spec):
    with open(spec) as file:
        experiment_spec = yaml.load(file, Loader=yaml.FullLoader)
    return experiment_spec

@click.command()
@click.option('--spec', type=click.Path(exists=True), required=True, help='Path to the experiment specification')
@click.option('--output-dir', type=click.Path(), required=True, help='Path to the output directory')
def generate_experiment(spec, output_dir):
    """ Generate Microsoft NNI Experiment """
    experiment_spec = _read_experiment_spec(spec)
    output_dir = os.path.join(output_dir, experiment_spec['nniConfig']['experimentName'])
    os.makedirs(output_dir, exist_ok=True)
    # copy the spec file as well in the experiment
    shutil.copy(spec, dst=output_dir)
    # generate the nni config and search space
    nni_config_generator(experiment_spec, output_dir)

    print("Done ! You are all set with your experiment.")
    print()
    print(f"Example cmd to run your experiment -")
    print("="*80)
    print(f"nnictl create --config {output_dir}/config.yml")
    print("="*80)


@click.command(hidden=True)
def run_classification_experiment():
    """ Run the experiment """
    config_files = glob.glob(os.path.join(os.getcwd(), "*.nni.yml"))

    if len(config_files) == 0:
        logging.error("Path to the nni specification is invalid !")
        raise ValueError("Could not find nni spec !")

    # read the spec
    experiment_spec = _read_experiment_spec(config_files[0])

    # get the datasource
    params = {}
    if 'params' in experiment_spec['dataSource'].keys():
        params = experiment_spec['dataSource']['params']

    datasource = get_class(experiment_spec['dataSource']['reader'])()

    (X_train,y_train), (X_test, y_test) = datasource(**params)

    # get the next nni parameters
    nni_hparams = nni.get_next_parameter()

    logging.info(f"Received from NNI -> {nni_hparams}")
    pipeline = PipelineBuilder(experiment_spec)(nni_hparams)

    # fit
    pipeline.fit(X_train, y_train)

    # score
    score = pipeline.score(X_test, y_test)
    logging.info(f"Final score from the pipeline is {score}")

    # report the score
    nni.report_final_result(score)

@click.group()
@click.option('--verbose/--no-verbose', default=False, required=False, help='if verbose then debug level is printed')
def cli(verbose):
    if verbose:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.INFO)

cli.add_command(generate_experiment)
cli.add_command(run_classification_experiment)

if __name__ == "__main__":
    cli()
