import webdataset as wds
import pytorch_lightning as pl

from .datasets import SemRepFactDataset


class SemRepFactDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._ran_setup = False

    def setup(self):
        self.dataset = SemRepFactDataset.from_config(self.config)
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
