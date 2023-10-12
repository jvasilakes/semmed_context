import os
import warnings

import torch
import numpy as np
from transformers import AutoTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right


os.environ["TOKENIZERS_PARALLELISM"] = "true"

ENCODER_REGISTRY = {}


def register_encoder(name):
    def add_to_registry(cls):
        ENCODER_REGISTRY[name] = cls
        return cls
    return add_to_registry


class Encoder(object):

    @classmethod
    def from_config(cls, config):
        return cls(config.Data.Encoder.bart_model_name_or_path.value,
                   config.Data.Encoder.max_seq_length.value)

    def __init__(self, bart_model_name_or_path="facebook/bart-large",
                 max_seq_length=256, cache=True):
        self.bart_model_name_or_path = bart_model_name_or_path
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.bart_model_name_or_path)
        # Whether to cache encoded examples
        self.cache = cache
        self._cache = {}

    def __call__(self, examples):
        if isinstance(examples, list):
            return [self._encode_and_cache(example)
                    for example in examples if example is not None]
        else:
            # There is just a single example
            return self._encode_and_cache(examples)

    def encode_single_example(self, example):
        raise NotImplementedError

    def _encode_and_cache(self, example):
        key = example["__key__"]
        try:
            return self._cache[key]
        except KeyError:
            encoded = self.encode_single_example(example)
            if self.cache is True:
                self._cache[key] = encoded
            return encoded


@register_encoder("default")
class DefaultEncoder(Encoder):

    def encode_single_example(self, example):
        data = example["json"]
        encoded = self.tokenizer(data["text"], return_offsets_mapping=True,
                                 max_length=self.max_seq_length,
                                 padding="max_length", truncation=True)

        subj_start, subj_end = data["subject"][1:]
        obj_start, obj_end = data["object"][1:]
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
            warnings.warn("Can't find subject or object. Try increasing max_seq_length")  # noqa
            return None
        all_idxs = subj_idxs + obj_idxs
        data["encoded"] = {k: torch.as_tensor(v)
                           for (k, v) in encoded.items()}
        data["subject_idxs"] = torch.as_tensor(all_idxs[:2])
        data["object_idxs"] = torch.as_tensor(all_idxs[2:])
        # Just keep the text, we don't need the character offsets anymore.
        data["subject"] = data["subject"][0]
        data["object"] = data["object"][0]
        return example


@register_encoder("solid-marker")
class SolidMarkerEncoder(Encoder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.special_tokens_map = {"subj_start_token": 1629,  # "$",
                                   "subj_end_token": 45946,   # "=$",
                                   "obj_start_token": 1039,   # "@",
                                   "obj_end_token": 49436}    # "@#"

    def encode_single_example(self, example):
        data = example["json"]
        encoded = self.tokenizer(data["text"], return_offsets_mapping=True,
                                 max_length=self.max_seq_length,
                                 padding="max_length", truncation=True,
                                 return_tensors="pt")

        subj_start, subj_end = data["subject"][1:]
        obj_start, obj_end = data["object"][1:]
        prev_end = 0
        subj_idxs = [None, None]
        obj_idxs = [None, None]
        for (token_i, (start, end)) in enumerate(encoded["offset_mapping"][0]):
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
            warnings.warn("Can't find subject or object. Try increasing max_seq_length")  # noqa
            return None

        pack_on = np.array([subj_idxs, obj_idxs])
        insert_order = np.argsort(pack_on, axis=0)[:, 0]
        packed_encoded, new_entity_idxs = self.add_solid_markers(
            encoded, [pack_on[insert_order]])

        # The transformers tokenizer adds a batch dimension, which we need to
        # remove to ensure collation works properly later.
        packed_encoded = {key: t.squeeze()
                          for (key, t) in packed_encoded.items()}
        data["encoded"] = packed_encoded

        target_predicate = self.get_predicate_text(data)
        target_encoded = self.tokenizer(
            target_predicate, max_length=20,
            padding="max_length", truncation=True, return_token_type_ids=False,
            return_attention_mask=False)
        data["encoded"]["labels"] = torch.as_tensor(
            target_encoded["input_ids"])
        # 2 is the decoder_start_token_id for BART
        data["encoded"]["decoder_input_ids"] = shift_tokens_right(
            data["encoded"]["labels"].unsqueeze(0),
            self.tokenizer.pad_token_id, 2).squeeze()

        new_subject_idxs = new_entity_idxs[0][insert_order[0]]
        new_object_idxs = new_entity_idxs[0][insert_order[1]]
        data["subject_idxs"] = torch.as_tensor(new_subject_idxs).unsqueeze(0)
        data["object_idxs"] = torch.as_tensor(new_object_idxs).unsqueeze(0)

        # Just keep the text, we don't need the character offsets anymore.
        data["subject"] = data["subject"][0]
        data["object"] = data["object"][0]

        return example

    def add_solid_markers(self, tokenizer_output, spans_to_mark):
        """
        Given a batch of tokenizer_output, add solid markers around the tokens
        at indices specified in spans_to_mark.

        .. code-block:: python

            >>> from transformers import AutoTokenizer
            >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            >>> enc = tokenizer("I like donuts")
            >>> spans_to_mark = [(1, 1)]
            >>> print(add_solid_markers(input_ids, spans_to_mark))
            ... {"input_ids": [[101, 1, 1045, 2, 2066, 2123, 16446, 102]], ...}

        :param dict tokenizer_output: the encoded input to modify.
        :param List[tuple] spans_to_mark: List of (start, end) indices in
            input_ids to add solid markers around.
        """
        batch_size = tokenizer_output["input_ids"].size(0)
        if batch_size != len(spans_to_mark):
            raise ValueError(f"len(spans_to_mark) ({len(spans_to_mark)}) != batch_size ({batch_size})")  # noqa

        new_texts = []
        all_new_entity_idxs = []
        for (ids, spans) in zip(tokenizer_output["input_ids"], spans_to_mark):
            start_marker_id = self.special_tokens_map["subj_start_token"]
            end_marker_id = self.special_tokens_map["subj_end_token"]
            # Make sure we process spans from left to right in the input.
            spans = sorted(spans, key=lambda s: s[0])
            new_entity_idxs = []
            for (i, (start, end)) in enumerate(spans):
                offset = i * 2  # offset index by 1 for each start/end marker.
                # Subtract 1 from end because start, end is a range.
                start, end = start + offset, (end - 1) + offset
                # We want to return the updated ranges, so add the 1 back in
                # and add two more for the markers themselves.
                new_entity_idxs.append((start, end + 3))
                ids = torch.cat((ids[:start],
                                 torch.LongTensor([start_marker_id]),
                                 ids[start:end+1],
                                 torch.LongTensor([end_marker_id]),
                                 ids[end+1:(self.max_seq_length-2)])
                                )
                start_marker_id = self.special_tokens_map["obj_start_token"]
                end_marker_id = self.special_tokens_map["obj_end_token"]
            if self.tokenizer.eos_token_id not in ids:
                # Just in case the sentence was truncated, we may have
                # cut out the eos token. Add it back in.
                ids[-1] = self.tokenizer.eos_token_id
            new_texts.append(self.tokenizer.decode(ids))
            all_new_entity_idxs.append(new_entity_idxs)
        # It's not ideal, but to get the correct token_type_ids and
        # attention_masks it is easiest to just re-run the solid-marked
        # input through the tokenizer.
        new_encodings = self.tokenizer(
            new_texts, max_length=self.max_seq_length, padding="max_length",
            truncation=True, add_special_tokens=False,
            return_tensors="pt")
        return new_encodings, all_new_entity_idxs

    def get_predicate_text(self, data):
        subj = data["subject"][0]
        obj = data["object"][0]

        use_infinitive_verb = False
        unc_val = data["labels"]["Certainty"]
        unc_text = None
        if unc_val == "Uncertain":
            use_infinitive_verb = True
            unc_text = "might"
        pol_val = data["labels"]["Polarity"]
        pol_text = None
        if pol_val == "Negative":
            use_infinitive_verb = True
            pol_text = "not"
            if unc_val == "Certain":
                pol_text = "does not"

        pred_tokens = data["labels"]["Predicate"].split('_')
        if use_infinitive_verb is True:
            pred_tokens[0] = pred_tokens[0].rstrip('S')
        if pred_tokens[0].endswith("ED"):
            pred_tokens[0] = pred_tokens[0].rstrip('D')
            if use_infinitive_verb is False:
                pred_tokens[0] = pred_tokens[0] + 'S'
        pred = ' '.join(pred_tokens)

        txts = [subj, unc_text, pol_text, pred, obj]
        pred_text = ' '.join([txt for txt in txts if txt is not None])
        return pred_text.lower()


if __name__ == "__main__":
    print("Available Encoders")
    print("------------------")
    for (k, v) in ENCODER_REGISTRY.items():
        print(f"  '{k}': {v.__name__}")
