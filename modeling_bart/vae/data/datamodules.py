import random
import webdataset as wds
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from .datasets import SemRepFactDataset, ConceptNetDataset


class SemRepFactDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._ran_setup = False

    def setup(self, stage=None):
        self.dataset = SemRepFactDataset.from_config(self.config)
        self.label_spec = self.dataset.label_spec
        self.batch_size = self.config.Training.batch_size.value
        self.tokenizer = self.dataset.encoder.tokenizer
        self.rng = random.Random(self.config.Experiment.random_seed.value)
        random.seed(self.config.Experiment.random_seed.value)
        self._ran_setup = True

    def train_dataloader(self):
        if getattr(self.dataset.train, "__len__", None) is not None:
            random.shuffle(self.dataset.train)
        return wds.WebLoader(self.dataset.train, batch_size=self.batch_size,
                             num_workers=4).shuffle(1000, rng=self.rng)

    def val_dataloader(self):
        return wds.WebLoader(self.dataset.val, batch_size=self.batch_size,
                             num_workers=4)

    def test_dataloader(self):
        return wds.WebLoader(self.dataset.test, batch_size=self.batch_size,
                             num_workers=4)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__


class ConceptNetDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._ran_setup = False

    def setup(self, stage=None):
        self.dataset = ConceptNetDataset.from_config(self.config)
        self.label_spec = self.dataset.label_spec
        self.batch_size = self.config.Training.batch_size.value
        self.tokenizer = self.dataset.encoder.tokenizer
        random.seed(self.config.Experiment.random_seed.value)
        self._ran_setup = True

    def train_dataloader(self):
        return DataLoader(self.dataset.train, batch_size=self.batch_size,
                          shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset.val, batch_size=self.batch_size,
                          shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset.test, batch_size=self.batch_size,
                          shuffle=False, num_workers=4)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__
