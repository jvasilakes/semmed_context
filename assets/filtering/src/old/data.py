import re

import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SemMedDataset(Dataset):
    """
    Dataset class for the several input components
    """
    PREDICATES = ['COEXISTS_WITH', 'COMPLICATES', 'MANIFESTATION_OF',
                  'PREVENTS', 'PRODUCES', 'TREATS',
                  'INTERACTS_WITH', 'STIMULATES',
                  'INHIBITS', 'CAUSES', 'PREDISPOSES',
                  'ASSOCIATED_WITH', 'DISRUPTS', 'AUGMENTS',
                  'AFFECTS']

    def __init__(self, datafile, bert_model_name_or_path="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",  # noqa
                 max_seq_length=128):
        self.datafile = datafile
        self.bert_model_name_or_path = bert_model_name_or_path
        self.max_seq_length = max_seq_length

        self.raw_data = self._load_data(self.datafile)
        self.encoded_data = {}  # will populate this cache on the fly
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.bert_model_name_or_path)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        try:
            encoded = self.encoded_data[idx]
        except KeyError:
            encoded = self.encode(self.raw_data.iloc[idx])
            self.encoded_data[idx] = encoded
        metadata = dict(self.raw_data.iloc[idx])
        encoded["metadata"] = metadata
        return encoded

    def _load_data(self, datafile):
        """
        Read data
        Keep only predicates of interest
        Downsample positive (most common) class
        """
        df = pd.read_csv(datafile)
        df = df[df.PREDICATE.isin(self.PREDICATES)]
        lab0 = df[df.LABEL == 0]
        lab1_downsampled = df[df.LABEL == 1].sample(n=len(lab0), replace=False)
        downsampled = pd.concat([lab0, lab1_downsampled])
        return downsampled

    def encode(self, datum):
        sentence = self.normalize_text(datum.SENTENCE)
        triple = ' '.join(datum[["SUBJECT_TEXT", "PREDICATE", "OBJECT_TEXT"]])
        encoded_input = self.tokenizer(sentence, triple, return_tensors="pt",
                                       max_length=self.max_seq_length,
                                       padding="max_length",
                                       truncation="only_first")
        # For whatever reason, default collate fn in DataLoader adds an extra
        # dimension if the input already has a batch dimension, so we remove it
        for key in encoded_input.keys():
            if torch.is_tensor(encoded_input[key]):
                encoded_input[key] = encoded_input[key].squeeze()
        encoded_label = torch.tensor(datum.LABEL, dtype=torch.long)
        return {"model_input": encoded_input,
                "label": encoded_label}

    def normalize_text(self, text):
        """
        Remove non-alpha characters
        Normalize whitespace
        """
        pattern = re.compile(r'[\W_]+')
        text = pattern.sub(' ', text)
        text = ' '.join(text.split())
        text = text.lower()
        return text
