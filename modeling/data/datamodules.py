import webdataset as wds
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from .datasets import SemRepFactDataset, SemRepFactWebDataset


class SemRepFactDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._ran_setup = False

    def setup(self):
        self.dataset = SemRepFactDataset.from_config(self.config)

        train_size = int(len(self.dataset) * 0.8)
        evalu_size = len(self.dataset) - train_size
        val_size = int(evalu_size / 2)
        test_size = evalu_size - val_size
        self.train, self.val, self.test = random_split(
            self.dataset, [train_size, val_size, test_size])

        self.label_spec = self.dataset.label_spec
        self.batch_size = self.config.Training.batch_size.value
        self.tokenizer = self.dataset.encoder.tokenizer
        self._ran_setup = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size,
                          shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size,
                          shuffle=False, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size,
                          shuffle=False, num_workers=4)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__


class SemRepFactWebDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._ran_setup = False

    def setup(self):
        self.dataset = SemRepFactWebDataset.from_config(self.config)
        self.label_spec = self.dataset.label_spec
        self.batch_size = self.config.Training.batch_size.value
        self.tokenizer = self.dataset.encoder.tokenizer
        self._ran_setup = True

    def train_dataloader(self):
        return wds.WebLoader(self.dataset.train, batch_size=self.batch_size,
                             num_workers=4).shuffle(1000)

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
