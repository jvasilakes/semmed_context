import os

from experiment_config import Config, get_and_run_config_command

from data.encoders import ENCODER_REGISTRY
from models import MODEL_REGISTRY, ENTITY_POOLER_REGISTRY


config = Config("SemRepFactConfig")


@config.parameter(group="Experiment", types=str)
def name(val):
    assert len(val) > 0


@config.parameter(group="Experiment", types=str)
def logdir(val):
    pass


@config.parameter(group="Experiment", default=0, types=int)
def version(val):
    assert val >= 0


@config.parameter(group="Experiment", default=0, types=int)
def random_seed(val):
    pass


@config.parameter(group="Data", types=str)
def datadir(val):
    """
    Directory containing .ann files containing
    annotations and .json files containing sentences.
    """
    assert os.path.isdir(val)


@config.parameter(group="Data", default="all", types=(str, list))
def tasks_to_load(val):
    """
    Can be either "all" or a list of strings specifying the tasks.
    """
    if isinstance(val, list):
        for item in val:
            assert isinstance(item, str)


@config.parameter(group="Data", default="default", types=str)
def encoder_type(val):
    """
    See ENCODER_REGISTRY in data/encoders.py for valid values.
    """
    assert val in ENCODER_REGISTRY.keys()


@config.parameter(group="Data", default="bert-base-uncased", types=str)
def bert_model_name_or_path(val):
    pass


@config.parameter(group="Data", default=256, types=int)
def max_seq_length(val):
    assert val > 1


@config.parameter(group="Data", default=-1, types=int)
def num_examples(val):
    assert val == -1 or val > 0


@config.parameter(group="Model", default="default", types=str)
def model_name(val):
    """
    See MODEL_REGISTRY is models.py for valid values.
    """
    assert val in MODEL_REGISTRY.keys()


@config.parameter(group="Model", default="bert-base-uncased", types=str)
def bert_model_name_or_path(val):
    pass


@config.parameter(group="Model", default=None, types=(str, type(None)))
def entity_pool_fn(val):
    assert val is None or val in ENTITY_POOLER_REGISTRY.keys()


@config.parameter(group="Training", default=1, types=int)
def epochs(val):
    assert val > 0


@config.parameter(group="Training", default=1e-3, types=float)
def lr(val):
    assert val > 0.0


@config.parameter(group="Training", default=8, types=int)
def batch_size(val):
    assert val > 0


@config.parameter(group="Training", default=0.0, types=float)
def weight_decay(val):
    assert val >= 0.0


@config.parameter(group="Training", default=0.0, types=float)
def dropout_prob(val):
    assert val >= 0.0


@config.on_load
def validate_parameters():
    assert config.Data.bert_model_name_or_path == config.Model.bert_model_name_or_path  # noqa
    assert config.Data.encoder_type == config.Model.model_name


if __name__ == "__main__":
    get_and_run_config_command(config)
