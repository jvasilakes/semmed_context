import os
from glob import glob
from collections import Counter

from torch.utils.data import Dataset

import pybrat

from .encoders import ENCODER_REGISTRY


class SemRepFactDataset(Dataset):
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

    @classmethod
    def from_config(cls, config):
        return cls(datadir=config.Data.datadir.value,
                   tasks_to_load=config.Data.tasks_to_load.value,
                   encoder_type=config.Data.encoder_type.value,
                   bert_model_name_or_path=config.Data.bert_model_name_or_path.value,
                   max_seq_length=config.Data.max_seq_length.value,
                   num_examples=config.Data.num_examples.value)

    def __init__(self,
                 datadir,
                 tasks_to_load="all",
                 encoder_type="default",
                 bert_model_name_or_path="bert-base-uncased",
                 max_seq_length=256,
                 num_examples=-1):
        super().__init__()
        assert os.path.isdir(datadir), f"{datadir} is not a directory."
        self.datadir = datadir
        self.tasks_to_load = tasks_to_load
        self.encoder_type = encoder_type
        self.bert_model_name_or_path = bert_model_name_or_path
        self.max_seq_length = max_seq_length
        self.num_examples = num_examples

        if self.tasks_to_load == "all":
            self.tasks_to_load = list(self.LABEL_ENCODINGS.keys())
        else:
            assert isinstance(self.tasks_to_load, (list, tuple))
        try:
            self.encoder = ENCODER_REGISTRY[encoder_type]()
        except KeyError:
            raise KeyError(f"Unknown encoder type: '{encoder_type}'")
        self.data = self.load()
        self.data = self.encoder(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def label_spec(self):
        return {task: len(labs) for (task, labs)
                in self.LABEL_ENCODINGS.items()
                if task in self.tasks_to_load}

    def load(self):
        examples = []
        annglob = os.path.join(self.datadir, "*.ann")
        annfiles = glob(annglob)
        for annfile in annfiles:
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
                example = {"text": sentence["_text"],
                           "subject": subj,
                           "predicate": pred.type,
                           "object": obj,
                           "labels": label_dict,
                           }
                examples.append(example)
                if len(examples) == self.num_examples:
                    return examples
        return examples

    def summarize(self):
        n = len(self.data)
        pred_counts = Counter([ex["predicate"] for ex in self.data])
        sorted_pred_counts = sorted(pred_counts.items(),
                                    key=lambda x: x[0])
        pred_counts_str = '\n             '.join(
            [f"{key}: {val} ({(val/n)*100:.2f}%)"
             for (key, val) in sorted_pred_counts])
        count_strings = {}
        for task in self.label_spec:
            task_counts = Counter([ex[task] for ex in self.data])
            task_counts_str = ', '.join(
                [f"{key}: {val} ({(val/n)*100:.2f}%)"
                 for (key, val) in task_counts.items()])
            count_strings[task] = task_counts_str

        print(f"N: {n}")
        print(f"Predicates: {pred_counts_str}")
        print("Labels")
        for (task, tasks_counts) in count_strings.items():
            print(f"  {task}: {task_counts}")
        print()
