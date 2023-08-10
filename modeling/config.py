import os

from experiment_config import Config, get_and_run_config_command

from data.encoders import ENCODER_REGISTRY
from models import MODEL_REGISTRY, ENTITY_POOLER_REGISTRY, LOSS_REGISTRY


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


@config.parameter(group="Data", default="all", types=(str, list))
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


@config.parameter(group="Data.Encoder", default="default",
                  types=(str, type(None)))
def encoder_type(val):
    """
    See ENCODER_REGISTRY in data/encoders.py for valid values.
    """
    assert val is None or val in ENCODER_REGISTRY.keys()


@config.parameter(group="Data.Encoder", default="bert-base-uncased", types=str)
def bert_model_name_or_path(val):
    pass


@config.parameter(group="Data.Encoder", default=256, types=int)
def max_seq_length(val):
    assert val > 1


@config.parameter(group="Data.Encoder", default={}, types=dict)
def init_kwargs(val):
    """
    A dict of optional keyword arguments to pass to the encoder.
    """
    pass


@config.parameter(group="Model", default="default", types=str)
def model_name(val):
    """
    See MODEL_REGISTRY in models.py for valid values.
    """
    assert val in MODEL_REGISTRY.keys()


@config.parameter(group="Model", default="bert-base-uncased", types=str)  # noqa
def bert_model_name_or_path(val):
    pass


@config.parameter(group="Model", default="first", types=(str, type(None)))
def entity_pool_fn(val):
    """
    How to pool the subject and object markers.
    See ENTITY_POOLER_REGISTRY in models.py for valid values.
    """
    assert val is None or val in ENTITY_POOLER_REGISTRY.keys()


@config.parameter(group="Model", default=True, types=bool)
def project_entities(val):
    """
    If True, h_p = f([PLM(h_s); PLM(h_o)]))
        dim(h_p) == bert_config.hidden_size
    If False, h_p = [PLM(h_s); PLM(h_o)]))
        dim(h_p) == 2 * bert_config.hidden_size
    """
    pass


@config.parameter(group="Model", default="max", types=(str, type(None)))
def levitated_pool_fn(val):
    """
    How to pool the levitated markers, if applicable.
    See ENTITY_POOLER_REGISTRY in models.py for valid values.
    """
    assert val is None or val in ENTITY_POOLER_REGISTRY.keys()


@config.parameter(group="Model.TaskEncoder", default="identity", types=str)  # noqa
def encoder_type(val):
    """
    identity: No task-specific encoder between final BERT layer and
              entity poolers.
    bert-layer: Use another BertLayer for each task.
    bert-attention: Use a BertAttention mechanism for each task.
    """
    assert val in ["identity", "bert-layer", "bert-attention"]


@config.parameter(group="Model.TaskEncoder", default={}, types=dict)  # noqa
def init_kwargs(val):
    """
    Optional keyword arguments to pass to the task encoder.
    """
    pass


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


@config.parameter(group="Training", default=1, types=int)
def accumulate_grad_batches(val):
    assert val >= 1


@config.parameter(group="Training", default=0.0, types=float)
def dropout_prob(val):
    assert val >= 0.0


@config.parameter(group="Training", default="cross_entropy", types=str)
def loss_fn(val):
    """
    See LOSS_REGISTRY in models.py for valid values.
    """
    assert val in LOSS_REGISTRY.keys()


@config.on_load
def validate_parameters():
    assert config.Data.Encoder.bert_model_name_or_path == config.Model.bert_model_name_or_path  # noqa
    if config.Data.Encoder.encoder_type.value is not None:
        # Use in to check since we can have derivative models,
        # e.g., levitated_marker vs. levitated_marker_attentions
        assert config.Data.Encoder.encoder_type.value in config.Model.model_name.value  # noqa
    if "attention" not in config.Model.model_name.value:
        assert "attention" not in config.Model.levitated_pool_fn.value


if __name__ == "__main__":
    get_and_run_config_command(config)
