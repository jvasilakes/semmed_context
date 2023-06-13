import os
from copy import deepcopy
from glob import glob
from collections import defaultdict
from tqdm import tqdm

import torch
import numpy as np
import webdataset as wds
import pybrat

from .encoders import ENCODER_REGISTRY

torch.multiprocessing.set_sharing_strategy('file_system')


class SemRepFactDataset(object):

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
                       "Doubtful": 4},
        "Predicate": {"AFFECTS": 0,
                      "ASSOCIATED_WITH": 1,
                      "AUGMENTS": 2,
                      "CAUSES": 3,
                      "COEXISTS_WITH": 4,
                      "DISRUPTS": 5,
                      "INHIBITS": 6,
                      "INTERACTS_WITH": 7,
                      "PREDISPOSES": 8,
                      "PREVENTS": 9,
                      "PRODUCES": 10,
                      "STIMULATES": 11,
                      "TREATS": 12}
        # "COMPLICATES" and "MANIFESTATION_OF" are omitted
        # due to very low frequencies (0.3% and 0.2%).
    }

    @property
    def INVERSE_LABEL_ENCODINGS(self):
        try:
            return getattr(self, "_inverse_label_encodings")
        except AttributeError:
            self._inverse_label_encodings = {
                task: {enc_label: str_label for (str_label, enc_label)
                       in self.LABEL_ENCODINGS[task].items()}
                for task in self.LABEL_ENCODINGS}
            return self._inverse_label_encodings

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
                 encoder=None,
                 tasks_to_load="all",
                 num_examples=-1):
        super().__init__()
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
        self.train, self.val, self.test = self.load()

    @property
    def label_spec(self):
        return {task: len(labs) for (task, labs)
                in self.LABEL_ENCODINGS.items()
                if task in self.tasks_to_load}

    def load(self):
        split_names = ["train", "val", "test"]
        split_files = [os.path.join(self.datadir, f"{split}.tar.gz")
                       for split in split_names]
        is_split = all([os.path.isfile(split_file)
                        for split_file in split_files])

        # splits will be three nested lists of [train, val, test]
        if is_split is True:
            splits = self.load_tar(self.datadir)
        else:
            all_examples = self.load_ann(self.datadir)
            splits = self.split(all_examples)
        return splits

    def load_tar(self, tardir):
        """
        Load the dataset already preprocessed and split
        into train, val, and test.
        """
        train_path = os.path.join(tardir, "train.tar.gz")
        val_path = os.path.join(tardir, "val.tar.gz")
        test_path = os.path.join(tardir, "test.tar.gz")
        train = wds.WebDataset(train_path).shuffle(1000).decode()
        train = train.map(self.encoder).map(self.tasks_filter)
        train = train.map(self.transform_labels)
        val = wds.WebDataset(val_path).decode()
        val = val.map(self.encoder).map(self.tasks_filter)
        val = val.map(self.transform_labels)
        test = wds.WebDataset(test_path).decode()
        test = test.map(self.encoder).map(self.tasks_filter)
        test = test.map(self.transform_labels)
        if self.num_examples > -1:
            # We have to call list, otherwise slice will return
            # different examples each epoch.
            train = list(train.slice(self.num_examples))
            val = list(val.slice(self.num_examples))
            test = list(test.slice(self.num_examples))
        return train, val, test

    def transform_labels(self, example):
        excp = deepcopy(example)
        label_dict = excp["json"].pop("labels")
        excp["json"]["labels"] = {task: self.LABEL_ENCODINGS[task][val]
                                  for (task, val) in label_dict.items()}
        return excp

    def inverse_transform_labels(self, example):
        excp = deepcopy(example)
        label_dict = excp["json"].pop("labels")
        excp["json"]["labels"] = {
            task: self.INVERSE_LABEL_ENCODINGS[task][val]
            for (task, val) in label_dict.items()}
        return excp

    def tasks_filter(self, sample):
        if self.tasks_to_load == "all":
            return sample
        sample["json"]["labels"] = {
            k: v for (k, v) in sample["json"]["labels"].items()
            if k in self.tasks_to_load}
        return sample

    def load_ann(self, anndir):
        """
        Load the dataset from .ann and .json files directly.
        """
        examples = []
        num_processed = 0
        annglob = os.path.join(anndir, "*.ann")
        annfiles = glob(annglob)
        assert len(annfiles) > 0, f"No examples found at {annglob}"
        for annfile in tqdm(annfiles):
            file_id = os.path.splitext(annfile)[0]
            jsonfile = f"{file_id}.json"
            assert os.path.isfile(jsonfile), f"{jsonfile} not found!"
            anns = pybrat.BratAnnotations.from_file(annfile)
            anntxt = pybrat.BratText.from_files(sentences=jsonfile)
            for event in anns.events:
                sentence = anntxt.sentences(annotations=event)[0]
                pred, subj, obj = event.spans
                subj = (subj.text,
                        subj.start_index - sentence["start_char"],
                        subj.end_index - sentence["start_char"])
                obj = (obj.text,
                       obj.start_index - sentence["start_char"],
                       obj.end_index - sentence["start_char"])
                label_dict = {task: event.attributes[task].value
                              for task in self.LABEL_ENCODINGS
                              if task in event.attributes}
                label_dict["Predicate"] = pred.type
                # Filter the labels to match the LABEL_ENCODINGS.
                keep_example = True
                for task in label_dict:
                    if task == "Certainty":
                        if label_dict[task] in ["L1", "L2"]:
                            label_dict[task] = "Uncertain"
                        elif label_dict[task] == "L3":
                            label_dict[task] = "Certain"
                        else:
                            raise ValueError(f"Unsupported Certainty label '{label_dict[task]}'")  # noqa
                    if label_dict[task] not in self.LABEL_ENCODINGS[task]:
                        keep_example = False
                example = {"__key__": f"sample{num_processed:06d}",
                           "__url__": annfile,
                           "json": {"text": sentence["_text"],
                                    "subject": subj,
                                    "object": obj,
                                    "labels": label_dict}
                           }
                if keep_example is True:
                    examples.append(self.transform_labels(example))
                    num_processed += 1
                if num_processed == self.num_examples:
                    return examples
        return examples

    def split(self, examples):
        train = []
        val = []
        test = []
        splits = [train, val, test]
        for (i, example) in enumerate(examples):
            split_idx = np.random.choice(range(3), p=[0.8, 0.1, 0.1])
            splits[split_idx].append(example)
        return splits

    def save(self, outdir):
        """
        Used for converting a dataset to webdataset tar format.
        """
        os.makedirs(outdir, exist_ok=False)
        train_path = os.path.join(outdir, "train.tar.gz")
        train_sink = wds.TarWriter(train_path, compress=True)

        val_path = os.path.join(outdir, "val.tar.gz")
        val_sink = wds.TarWriter(val_path, compress=True)

        test_path = os.path.join(outdir, "test.tar.gz")
        test_sink = wds.TarWriter(test_path, compress=True)

        splits = [self.train, self.val, self.test]
        sinks = [train_sink, val_sink, test_sink]
        for (split, sink) in zip(splits, sinks):
            for example in split:
                sink.write(self.inverse_transform_labels(example))
            sink.close()

    def summarize(self):
        for split in ["train", "val", "test"]:
            print(split.upper())
            print("=" * len(split))
            split_data = getattr(self, split)

            n = 0
            task_counts = defaultdict(lambda: defaultdict(int))
            for example in tqdm(split_data):
                n += 1
                example = self.inverse_transform_labels(example)
                for (task, val) in example["json"]["labels"].items():
                    task_counts[task][val] += 1

            count_strings = {}
            for (task, counts) in task_counts.items():
                sorted_counts = sorted(counts.items(), key=lambda x: x[0])
                counts_str = '\n      '.join(
                    [f"{key}: {val} ({(val/n)*100:.2f}%)"
                     for (key, val) in sorted_counts])
                count_strings[task] = counts_str

            print(f"  N: {n}")
            print("  Labels")
            for (task, tc) in count_strings.items():
                print(f"    {task}:\n      {tc}")
            print()
