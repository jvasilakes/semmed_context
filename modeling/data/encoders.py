import torch
import numpy as np
from transformers import AutoTokenizer


ENCODER_REGISTRY = {}


def register_encoder(name):
    def add_to_registry(cls):
        ENCODER_REGISTRY[name] = cls
        return cls
    return add_to_registry


class Encoder(object):

    def __init__(self, bert_model_name_or_path="bert-base-uncased",
                 max_seq_length=256):
        self.bert_model_name_or_path = bert_model_name_or_path
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.bert_model_name_or_path)

    def __call__(self, examples):
        return [self.encode_single_example(example)
                for example in examples]

    def encode_single_example(self, example):
        raise NotImplementedError


@register_encoder("default")
class DefaultEncoder(Encoder):

    def encode_single_example(self, example):
        encoded = self.tokenizer(example["text"], return_offsets_mapping=True,
                                 max_length=self.max_seq_length,
                                 padding="max_length", truncation=True)

        subj_start, subj_end = example["subject"][1:]
        obj_start, obj_end = example["object"][1:]
        prev_end = 0
        subj_idxs = [None, None]
        obj_idxs = [None, None]
        for (token_i, (start, end)) in enumerate(encoded["offset_mapping"]):
            if prev_end <= subj_start and start >= subj_start:
                subj_idxs[0] = token_i
            elif prev_end <= subj_end and start >= subj_end:
                subj_idxs[1] = token_i
            if prev_end <= obj_start and start >= obj_start:
                obj_idxs[0] = token_i
            elif prev_end <= obj_end and start >= obj_end:
                obj_idxs[1] = token_i
            if all(subj_idxs + obj_idxs):  # None casts to False
                break
            prev_end = end
        del encoded["offset_mapping"]  # no longer needed
        if not all(subj_idxs + obj_idxs):
            raise ValueError("Can't find subject or object. Try increasing max_seq_length")  # noqa
        all_idxs = subj_idxs + obj_idxs
        example["encoded"] = {k: torch.as_tensor(v)
                              for (k, v) in encoded.items()}
        example["subject_idxs"] = torch.as_tensor(all_idxs[:2])
        example["object_idxs"] = torch.as_tensor(all_idxs[2:])
        # Just keep the text, we don't need the character offsets anymore.
        example["subject"] = example["subject"][0]
        example["object"] = example["object"][0]
        return example


@register_encoder("solid_marker")
class SolidMarkerEncoder(Encoder):

    def encode_single_example(self, example):
        encoded = self.tokenizer(example["text"], return_offsets_mapping=True,
                                 max_length=self.max_seq_length,
                                 padding="max_length", truncation=True)

        subj_start, subj_end = example["subject"][1:]
        obj_start, obj_end = example["object"][1:]
        prev_end = 0
        subj_idxs = [None, None]
        obj_idxs = [None, None]
        for (token_i, (start, end)) in enumerate(encoded["offset_mapping"]):
            if prev_end <= subj_start and start >= subj_start:
                subj_idxs[0] = token_i
            elif prev_end <= subj_end and start >= subj_end:
                subj_idxs[1] = token_i
            if prev_end <= obj_start and start >= obj_start:
                obj_idxs[0] = token_i
            elif prev_end <= obj_end and start >= obj_end:
                obj_idxs[1] = token_i
            if all(subj_idxs + obj_idxs):  # None casts to False
                break
            prev_end = end
        del encoded["offset_mapping"]  # no longer needed
        if not all(subj_idxs + obj_idxs):
            raise ValueError("Can't find subject or object. Try increasing max_seq_length")  # noqa
        #           [S],         [/S]         [O]          [/O]
        markers = ["[unused0]", "[unused1]", "[unused2]", "[unused3]"]
        marker_ids = self.tokenizer.convert_tokens_to_ids(markers)
        all_idxs = subj_idxs + obj_idxs
        insert_order = np.argsort(all_idxs)
        for (n, i) in enumerate(insert_order):
            encoded["input_ids"].insert(all_idxs[i] + n, marker_ids[i])
            all_idxs[i] += n
        encoded["input_ids"] = encoded["input_ids"][:self.max_seq_length]
        example["encoded"] = {k: torch.as_tensor(v)
                              for (k, v) in encoded.items()}
        example["subject_idxs"] = torch.as_tensor(all_idxs[:2])
        example["object_idxs"] = torch.as_tensor(all_idxs[2:])
        # Just keep the text, we don't need the character offsets anymore.
        example["subject"] = example["subject"][0]
        example["object"] = example["object"][0]
        return example


if __name__ == "__main__":
    print("Available Encoders")
    print("------------------")
    for (k, v) in ENCODER_REGISTRY.items():
        print(f"  '{k}': {v.__name__}")
