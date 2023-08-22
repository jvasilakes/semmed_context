import os

from experiment_config import Config, get_and_run_config_command


config = Config("VAEConfig")


@config.parameter(group="Experiment", types=str)
def name(val):
    assert val != ''


@config.parameter(group="Experiment", types=str)
def checkpoint_dir(val):
    pass


@config.parameter(group="Experiment", default=0, types=int)
def random_seed(val):
    pass


@config.parameter(group="Data", types=str)
def datadir(val):
    assert os.path.isdir(val)


@config.parameter(group="Data", types=str)
def dataset_name(val):
    assert val != ''


@config.parameter(group="Data", default="all", types=(str,list))
def tasks_to_load(val):
    pass


@config.parameter(group="Data", default=-1, types=int)
def num_examples(val):
    assert val == -1 or val > 0


@config.parameter(group="Data", default="lstm", types=str)
def encoder_type(val):
    pass


@config.parameter(group="Model", default=256, types=int)
def embedding_dim(val):
    assert val > 0


@config.parameter(group="Model", default=256, types=int)
def hidden_dim(val):
    assert val > 0


@config.parameter(group="Model", default=2, types=int)
def num_rnn_layers(val):
    assert val >= 2


@config.parameter(group="Model", default=False, types=bool)
def bow_encoder(val):
    pass


@config.parameter(group="Model", default=True, types=bool)
def bidirectional_encoder(val):
    pass


@config.parameter(group="Model", default=0.0, types=float)
def encoder_dropout(val):
    assert 0.0 <= val <= 1.0


@config.parameter(group="Model", default=0.0, types=float)
def decoder_dropout(val):
    assert 0.0 <= val <= 1.0


@config.parameter(group="Model", default={"total": 32}, types=dict)
def latent_dims(val):
    """
    The dimensionality of each latent space, including the total.
    Will default to the number of labels if not specified.
    """
    for (name, dim) in val.items():
        assert isinstance(name, str)
        assert name != ''
        assert isinstance(dim, int)
        assert dim > 0


@config.parameter(group="Training", default=1, types=int)
def epochs(val):
    assert val > 0


@config.parameter(group="Training", default=32, types=int)
def batch_size(val):
    assert val > 0


@config.parameter(group="Training", default=1e-4, types=float)
def learn_rate(val):
    assert val > 0.0


@config.parameter(group="Training", default=0.5, types=float)
def teacher_forcing_prob(val):
    assert 0.0 <= val <= 1.0


@config.parameter(group="Training", default={"default": 1.0}, types=dict)
def lambdas(val):
    """
    The weight of the KL term for each latent space.
    """
    for (name, lam) in val.items():
        assert isinstance(name, str)
        assert name != ''
        assert isinstance(lam, float)


@config.parameter(group="Training", default=False, types=bool)
def adversarial_loss(val):
    pass


@config.parameter(group="Training", default=False, types=bool)
def mi_loss(val):
    pass


if __name__ == "__main__":
    get_and_run_config_command(config)
