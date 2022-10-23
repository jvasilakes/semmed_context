import os
import argparse

from experiment_config import BaseConfig, parameter


class ExperimentConfig(BaseConfig):

    @parameter("Logging", type=str, default="model_checkpoints")
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @parameter("Logging", type=str, default="logs")
    def logdir(self):
        return self._logdir

    @parameter("Data", type=str)
    def datafile(self):
        assert os.path.isfile(self._datafile), f"Couldn't find datafile at {datafile}"  # noqa
        return self._datafile

    @parameter("Data", type=int, default=128)
    def max_seq_length(self):
        assert self._max_seq_length > 0
        return self._max_seq_length

    @parameter("Model", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")  # noqa
    def bert_model_name_or_path(self):
        return self._bert_model_name_or_path

    @parameter("Model", type=float, default=0.1)
    def dropout_prob(self):
        assert 0.0 <= self._dropout_prob <= 1.0
        return self._dropout_prob

    @parameter("Training", type=int, default=7)
    def epochs(self):
        assert self._epochs > 0
        return self._epochs

    @parameter("Training", type=float, default=1e-5)
    def lr(self):
        assert self._lr > 0
        return self._lr

    @parameter("Training", type=float, default=0.01)
    def weight_decay(self):
        assert self._weight_decay >= 0.0
        return self._weight_decay

    @parameter("Training", type=int, default=16)
    def batch_size(self):
        assert self._batch_size > 0
        return self._batch_size

    @parameter("Training", type=int, default=0)
    def random_state(self):
        return self._random_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outpath", type=str,
                        help="Where to save the default config file.")
    args = parser.parse_args()
    ExperimentConfig.save_default(args.outpath)
