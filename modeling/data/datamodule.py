import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from .datasets import SemRepFactDataset


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
        self.train, evalu = random_split(self.dataset, [train_size, evalu_size])
        self.val, self.test = random_split(evalu, [val_size, test_size])
        self.label_spec = self.dataset.label_spec
        self.tokenizer = self.dataset.encoder.tokenizer
        self._ran_setup = True

    def train_dataloader(self) -> DataLoader:
        batch_size = self.config.Training.batch_size.value
        return DataLoader(self.train, batch_size=batch_size,
                          shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        batch_size = self.config.Training.batch_size.value
        return DataLoader(self.val, batch_size=batch_size,
                          shuffle=False, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        batch_size = self.config.Training.batch_size.value
        return DataLoader(self.test, batch_size=batch_size,
                          shuffle=False, num_workers=4)
