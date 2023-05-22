import os
import warnings
from glob import glob
from collections import Counter
from tqdm import tqdm

import torch
import numpy as np
import webdataset as wds
from torch.utils.data import Dataset

import pybrat

from .encoders import ENCODER_REGISTRY

torch.multiprocessing.set_sharing_strategy('file_system')


class BaseSemRepFactDataset(object):

    LABEL_ENCODINGS = {
        "Certainty": {"Uncertain": 0,  # L1 or L2
                      "Certain": 1},   # L3
        "Polarity": {"Negative": 0,
                     "Positive": 1},
        "Factuality": {"Fact": 0,
                       "Counterfact": 1,
                       "Probable": 2,
                       # Uncommitted is omitted
                       "Possible": 3,
                       "Doubtful": 4}
    }

    def __init__(self,
                 datadir,
                 encoder=None,
                 tasks_to_load="all",
                 num_examples=-1):
        assert os.path.isdir(datadir), f"{datadir} is not a directory."
        self.datadir = datadir
        if encoder is None:
            self.encoder = lambda example: example
        else:
            self.encoder = encoder
        self.tasks_to_load = tasks_to_load
        self.num_examples = num_examples

        if isinstance(tasks_to_load, str):
            if tasks_to_load == "all":
                tasks_to_load = list(self.LABEL_ENCODINGS.keys())
            else:
                tasks_to_load = [tasks_to_load]
        assert isinstance(tasks_to_load, (list, tuple))
        tasks_set = set(tasks_to_load)
        valid_tasks_set = set(self.LABEL_ENCODINGS)
        unknown_tasks = tasks_set.difference(valid_tasks_set)
        assert len(unknown_tasks) == 0, f"Unknown tasks: '{unknown_tasks}'"
        self.tasks_to_load = tasks_to_load

    @property
    def label_spec(self):
        return {task: len(labs) for (task, labs)
                in self.LABEL_ENCODINGS.items()
                if task in self.tasks_to_load}

    def load(self):
        raise NotImplementedError()


class SemRepFactWebDataset(BaseSemRepFactDataset):

    @classmethod
    def from_config(cls, config):
        encoder_type = config.Data.Encoder.encoder_type.value
        if encoder_type is None:
            encoder = None
        else:
            encoder = ENCODER_REGISTRY[encoder_type].from_config(config)
        if config.Data.num_examples.value != -1:
            warnings.warn("Data.num_examples was set but is ignored by SemRepFactWebDataset")  # noqa
        return cls(config.Data.datadir.value, encoder=encoder,
                   tasks_to_load=config.Data.tasks_to_load.value)

    def __init__(self,
                 datadir,
                 encoder,
                 tasks_to_load="all"):
        super().__init__(datadir, encoder, tasks_to_load=tasks_to_load)
        self.load()

    def load(self):
        train_path = os.path.join(self.datadir, "train.tar")
        val_path = os.path.join(self.datadir, "val.tar")
        test_path = os.path.join(self.datadir, "test.tar")
        self.train = wds.WebDataset(train_path).shuffle(1000).decode()
        self.train = self.train.map(self.encoder).map(self.tasks_filter)
        self.val = wds.WebDataset(val_path).decode().map(self.encoder)
        self.val = self.val.map(self.encoder).map(self.tasks_filter)
        self.test = wds.WebDataset(test_path).decode().map(self.encoder)
        self.test = self.test.map(self.encoder).map(self.tasks_filter)

    def tasks_filter(self, sample):
        if self.tasks_to_load == "all":
            return sample
        sample["json"]["labels"] = {
            k: v for (k, v) in sample["json"]["labels"].items()
            if k in self.tasks_to_load}
        return sample


class SemRepFactDataset(BaseSemRepFactDataset, Dataset):

    @classmethod
    def from_config(cls, config):
        encoder_type = config.Data.Encoder.encoder_type.value
        if encoder_type is None:
            encoder = None
        else:
            encoder = ENCODER_REGISTRY[encoder_type].from_config(config)
        return cls(datadir=config.Data.datadir.value,
                   encoder=encoder,
                   tasks_to_load=config.Data.tasks_to_load.value,
                   num_examples=config.Data.num_examples.value)

    def __init__(self,
                 datadir,
                 encoder,
                 tasks_to_load="all",
                 num_examples=-1):
        BaseSemRepFactDataset.__init__(
            self, datadir, encoder,
            tasks_to_load=tasks_to_load,
            num_examples=num_examples)
        Dataset.__init__(self)
        # To function as a standard Dataset, we need random access
        # to the data, so we exhaust the load() generator.
        self.data = list(self.load())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # The encoder caches the result
        return self.encoder(self.data[idx])

    def load(self):
        num_processed = 0
        annglob = os.path.join(self.datadir, "*.ann")
        annfiles = glob(annglob)
        for annfile in tqdm(annfiles):
            file_id = os.path.splitext(annfile)[0]
            jsonfile = f"{file_id}.json"
            assert os.path.isfile(jsonfile), f"{jsonfile} not found!"
            anns = pybrat.BratAnnotations.from_file(annfile)
            anntxt = pybrat.BratText.from_files(sentences=jsonfile)
            for event in anns.events:
                # Uncommitted Factuality have no certainty/polarity labels.
                if event.attributes["Factuality"].value == "Uncommitted":
                    continue
                sentence = anntxt.sentences(annotations=event)[0]
                pred, subj, obj = event.spans
                subj = (subj.text,
                        subj.start_index - sentence["start_char"],
                        subj.end_index - sentence["start_char"])
                obj = (obj.text,
                       obj.start_index - sentence["start_char"],
                       obj.end_index - sentence["start_char"])
                label_dict = {task: event.attributes[task].value
                              for task in self.LABEL_ENCODINGS}
                for task in label_dict:
                    if task == "Certainty":
                        if label_dict[task] in ["L1", "L2"]:
                            label_dict[task] = "Uncertain"
                        elif label_dict[task] == "L3":
                            label_dict[task] = "Certain"
                        else:
                            raise ValueError(f"Unsupported Certainty label '{label_dict[task]}'")  # noqa
                label_dict = {task: self.LABEL_ENCODINGS[task][str_value]
                              for (task, str_value) in label_dict.items()}
                example = {"__key__": f"sample{num_processed:06d}",
                           "__url__": annfile,
                           "json": {"text": sentence["_text"],
                                    "subject": subj,
                                    "predicate": pred.type,
                                    "object": obj,
                                    "labels": label_dict}
                           }
                yield example
                num_processed += 1
                if num_processed == self.num_examples:
                    return

    def summarize(self):
        n = len(self.data)
        pred_counts = Counter([ex["json"]["predicate"] for ex in self.data])
        sorted_pred_counts = sorted(pred_counts.items(),
                                    key=lambda x: x[0])
        pred_counts_str = '\n  '.join(
            [f"{key}: {val} ({(val/n)*100:.2f}%)"
             for (key, val) in sorted_pred_counts])
        count_strings = {}
        for task in self.label_spec:
            task_counts = Counter([ex["json"]["labels"][task]
                                   for ex in self.data])
            task_counts_str = ', '.join(
                [f"{key}: {val} ({(val/n)*100:.2f}%)"
                 for (key, val) in task_counts.items()])
            count_strings[task] = task_counts_str

        print(f"N: {n}")
        print(f"Predicates:\n  {pred_counts_str}")
        print("Labels")
        for (task, tc) in count_strings.items():
            print(f"  {task}: {tc}")
        print()

    def to_tar(self, outdir):
        """
        Used for converting a dataset to webdataset tar format.
        """
        train_path = os.path.join(outdir, "train.tar")
        train_sink = wds.TarWriter(train_path)

        val_path = os.path.join(outdir, "val.tar")
        val_sink = wds.TarWriter(val_path)

        test_path = os.path.join(outdir, "test.tar")
        test_sink = wds.TarWriter(test_path)

        sinks = [train_sink, val_sink, test_sink]
        for (i, example) in enumerate(self.data):
            sink = np.random.choice(sinks, p=[0.8, 0.1, 0.1])
            sink.write(example)
        sink.close()
