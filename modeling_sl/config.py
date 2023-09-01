import os

from experiment_config import Config, get_and_run_config_command


config = Config("SemRepFactConfig")


@config.parameter(group="Experiment", types=str)
def name(val):
    """
    What to name this experiment.
    """
    assert val is not None
    assert val != ''


@config.parameter(group="Experiment", types=str)
def logdir(val):
    """
    Base directory where to save experiment logs, model checkpoints,
    predictions, etc.
    Actual experiment directory will be {logdir}/{name}/version_{version}/.
    """
    assert val is not None
    assert val != ''


@config.parameter(group="Experiment", default=0, types=int)
def version(val):
    """
    The version number of this experiment.
    """
    assert val is not None
    assert val >= 0


@config.parameter(group="Experiment", default=0, types=int)
def random_seed(val):
    assert val is not None


@config.parameter(group="Data", types=str)
def datadir(val):
    """
    Directory containing .ann files and containing
    annotations and .json files containing sentences.
    OR
    A directory containing .tar files with train, dev, and test splits
    that can be loaded by webdataset.
    """
    assert os.path.isdir(val)


@config.parameter(group="Data", default="all", types=(str,list))
def tasks_to_load(val):
    """
    Can be either "all" or a list of strings specifying the tasks.
    """
    if isinstance(val, list):
        for item in val:
            assert isinstance(item, str)


@config.parameter(group="Data", default=-1, types=int)
def num_examples(val):
    assert val == -1 or val > 0


if __name__ == "__main__":
    get_and_run_config_command(config)
