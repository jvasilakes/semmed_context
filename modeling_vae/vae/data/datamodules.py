import random
import webdataset as wds
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, default_collate

from .datasets import SemRepFactDataset, ConceptNetDataset


DATAMODULE_REGISTRY = {}


def register_datamodule(name):
    def add_to_registry(cls):
        DATAMODULE_REGISTRY[name] = cls
        return cls
    return add_to_registry


def min_pad_batch(batch, key):
    max_length = max([len(ex["json"][key]["input_ids"])
                      for ex in batch])
    for ex in batch:
        input_ids = ex["json"][key]["input_ids"]
        pad_length = max_length - len(input_ids)
        if pad_length == 0:
            continue
        id_pad = torch.zeros(pad_length, dtype=torch.long)
        ex["json"][key]["input_ids"] = torch.cat([input_ids, id_pad])


def min_pad_collate(batch):
    keys = ["encoded"]
    if "target_encoded" in batch[0]["json"].keys():
        keys.append("target_encoded")
    for key in keys:
        min_pad_batch(batch, key)
    collated = default_collate(batch)
    return collated


@register_datamodule("semrep-fact")
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
        self.rng = random.Random(self.config.Experiment.random_seed.value)
        random.seed(self.config.Experiment.random_seed.value)
        self._ran_setup = True

    def train_dataloader(self):
        if getattr(self.dataset.train, "__len__", None) is not None:
            random.shuffle(self.dataset.train)
        return wds.WebLoader(
            self.dataset.train, batch_size=self.batch_size,
            num_workers=4, collate_fn=min_pad_collate).shuffle(1000, rng=self.rng)  # noqa

    def val_dataloader(self):
        return wds.WebLoader(self.dataset.val, batch_size=self.batch_size,
                             num_workers=4, collate_fn=min_pad_collate)

    def test_dataloader(self):
        return wds.WebLoader(self.dataset.test, batch_size=self.batch_size,
                             num_workers=4, collate_fn=min_pad_collate)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__


@register_datamodule("conceptnet")
class ConceptNetDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._ran_setup = False

    def setup(self):
        self.dataset = ConceptNetDataset.from_config(self.config)
        self.label_spec = self.dataset.label_spec
        self.batch_size = self.config.Training.batch_size.value
        self.tokenizer = self.dataset.encoder.tokenizer
        random.seed(self.config.Experiment.random_seed.value)
        self._ran_setup = True

    def train_dataloader(self):
        return DataLoader(self.dataset.train, batch_size=self.batch_size,
                          shuffle=True, num_workers=4,
                          collate_fn=min_pad_collate)

    def val_dataloader(self):
        return DataLoader(self.dataset.val, batch_size=self.batch_size,
                          shuffle=False, num_workers=4,
                          collate_fn=min_pad_collate)

    def test_dataloader(self):
        return DataLoader(self.dataset.test, batch_size=self.batch_size,
                          shuffle=False, num_workers=4,
                          collate_fn=min_pad_collate)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__
