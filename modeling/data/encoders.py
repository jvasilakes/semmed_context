import os
import string
import warnings

import torch
import numpy as np
from transformers import AutoTokenizer
from spacy.lang.en.stop_words import STOP_WORDS


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
        return cls(config.Data.Encoder.bert_model_name_or_path.value,
                   config.Data.Encoder.max_seq_length.value)

    def __init__(self, bert_model_name_or_path="bert-base-uncased",
                 max_seq_length=256):
        self.bert_model_name_or_path = bert_model_name_or_path
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.bert_model_name_or_path)
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


@register_encoder("solid_marker")
class SolidMarkerEncoder(Encoder):

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

        entity_marker_ids = []
        new_texts = []
        all_new_entity_idxs = []
        for (ids, spans) in zip(tokenizer_output["input_ids"], spans_to_mark):
            start_marker_id, end_marker_id = 1, 2
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
                                 ids[end+1:])
                                )
                entity_marker_ids.extend([start_marker_id, end_marker_id])
                start_marker_id = end_marker_id + 1
                end_marker_id = start_marker_id + 1
            new_texts.append(self.tokenizer.decode(ids))
            all_new_entity_idxs.append(new_entity_idxs)
        # It's not ideal, but to get the correct token_type_ids and
        # attention_masks it is easiest to just re-run the solid-marked
        # input through the tokenizer.
        # _add_tokens is necessary to not split the markers when re-tokenizing
        entity_marker_tokens = self.tokenizer.convert_ids_to_tokens(entity_marker_ids)  # noqa
        self.tokenizer._add_tokens(entity_marker_tokens, special_tokens=True)
        new_encodings = self.tokenizer(
            new_texts, max_length=self.max_seq_length, padding="max_length",
            truncation=True, add_special_tokens=False,
            return_tensors="pt")
        return new_encodings, all_new_entity_idxs


@register_encoder("levitated_marker")
class LevitatedMarkerEncoder(SolidMarkerEncoder):

    @classmethod
    def from_config(cls, config):
        return cls(config.Data.Encoder.bert_model_name_or_path.value,
                   config.Data.Encoder.max_seq_length.value,
                   **config.Data.Encoder.init_kwargs.value)

    def __init__(self,
                 bert_model_name_or_path="bert-base-uncased",
                 max_seq_length=256,
                 levitated_window_size=4,
                 max_num_markers=40,
                 ignore_stopwords=False
                 ):
        super().__init__(bert_model_name_or_path, max_seq_length)
        self.levitated_window_size = levitated_window_size
        self.max_num_markers = max_num_markers
        self.ignore_stopwords = ignore_stopwords

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

        spans_to_mark = self.get_levitated_spans(
            packed_encoded["input_ids"][0], new_entity_idxs,
            ignore_stopwords=self.ignore_stopwords)
        spans_before_entity = [(s, e) for (s, e) in spans_to_mark
                               if e < new_entity_idxs[0][0][0]]
        spans_between = [(s, e) for (s, e) in spans_to_mark
                         if s >= new_entity_idxs[0][0][-1]
                         and e < new_entity_idxs[0][-1][0]]
        spans_after_entity = [(s, e) for (s, e) in spans_to_mark
                              if s >= new_entity_idxs[0][-1][-1]]
        spans_in_window = spans_before_entity[-self.levitated_window_size:] + \
            spans_between + spans_after_entity[:self.levitated_window_size]
        spans_in_window = spans_in_window[:self.max_num_markers]
        packed_levitated, lev_marker_idxs = self.add_levitated_markers(
            packed_encoded, [spans_in_window],
            start_marker_id=5, end_marker_id=6)
        # The transformers tokenizer adds a batch dimension, which we need to
        # remove to ensure collation works properly later.
        packed_levitated = {key: t.squeeze()
                            for (key, t) in packed_levitated.items()}

        data["encoded"] = packed_levitated
        # Take the 0 index of new_entity_idxs to remove batch dimension.
        new_subject_idxs = new_entity_idxs[0][insert_order[0]]
        new_object_idxs = new_entity_idxs[0][insert_order[1]]
        data["subject_idxs"] = torch.as_tensor(new_subject_idxs).unsqueeze(0)
        data["object_idxs"] = torch.as_tensor(new_object_idxs).unsqueeze(0)
        # Just keep the text, we don't need the character offsets anymore.
        data["subject"] = data["subject"][0]
        data["object"] = data["object"][0]
        pad_amount = self.max_num_markers - len(spans_in_window)
        padded_lev_spans = torch.as_tensor(
            lev_marker_idxs[0] + [[0, 0] for _ in range(pad_amount)])
        data["levitated_idxs"] = padded_lev_spans
        return example

    def get_levitated_spans(self, input_ids, entity_idxs, ignore_stopwords=None):  # noqa

        entity_starts_ends = torch.stack(
            [torch.as_tensor(idxs) for idxs in entity_idxs[0]])
        # From the beginning of the first-occurring entity to the end of
        # the second-occurring entity.
        entity_idx_range = torch.cat([torch.arange(*start_end)
                                      for start_end in entity_starts_ends])
        seq_len_no_pad = (input_ids != 0).sum()
        # Start at 1 to remove the [CLS] token.
        # Subtract 1 from seq_len_no_pad to remove the [SEP] token.
        lev_idx_range = torch.arange(1, seq_len_no_pad - 1)
        combined = torch.cat((entity_idx_range, lev_idx_range))
        # This computes the set difference
        # That is, we only want the levitated indices that are
        # not part of an entity.
        uniques, counts = combined.unique(return_counts=True)
        levitated_idxs = uniques[counts == 1].tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        token_spans = self.collapse_wordpiece_spans(
            tokens, levitated_idxs,
            ignore_stopwords=ignore_stopwords)
        return token_spans

    def collapse_wordpiece_spans(self, tokens, token_idxs, ignore_stopwords=False):  # noqa
        out_token_spans = []
        current_token_span = []
        # Always ignore punctuation
        ignore_tokens = set(string.punctuation)
        if ignore_stopwords is True:
            ignore_tokens = STOP_WORDS.union(set(string.punctuation))
        for idx in token_idxs:
            wp = tokens[idx]
            if wp.startswith("##"):
                current_token_span.append(idx)
            else:
                if current_token_span != []:
                    collapsed_token = ''.join(
                        [tokens[i].strip('#') for i in current_token_span])
                    if collapsed_token not in ignore_tokens:
                        out_token_spans.append((current_token_span[0],
                                                current_token_span[-1]+1))
                current_token_span = [idx]
        if current_token_span != []:
            collapsed_token = ''.join(
                [tokens[i].strip('#') for i in current_token_span])
            if collapsed_token not in ignore_tokens:
                out_token_spans.append((current_token_span[0],
                                        current_token_span[-1]+1))
        return out_token_spans

    def add_levitated_markers(self, tokenizer_output, spans_to_mark,
                              start_marker_id=1, end_marker_id=2):
        """
        :param dict tokenizer_output: The output of an AutoTokenizer.
        :param List[List[tuple] spans_to_mark: List of sets of tuples of
            (start, end) token indices corresponding to
            tokenizer_output["input_ids"], for which levitated markers will be
            created. The number of sets of tuples len(spans_to_mark) must
            equal the batch size inferred from the tokenizer_output.
        :param transformers.PreTrainedTokenizer tokenizer:
            The tokenizer which return tokenizer_output
        :param int start_marker_id: (Optional) Default is 1 = "[unused0]".
        :param int end_marker_id: (Optional) Default is 2 = "[unused1]".

        E.g.,

        .. code-block:: python

            >>> from transformers import AutoTokenizer, AutoModel
            >>> model_name = "bert-base-uncased"
            >>> tokenizer = AutoTokenizer.from_pretrained(model_name)
            >>> model = AutoModel.from_pretrained(model_name)
            >>> text = "I like donuts"
            >>> encoded = tokenizer(text, max_length=8, padding="max_length",
            ...                     return_tensors="pt")
            >>> spans_to_mark = [(1, 1), (3, 4)]  # start/end of 'I', 'donuts'.
            >>> levitated = add_levitated_markers(encoded, spans_to_mark)
        """
        new_encodings = {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "attention_mask": []
        }
        input_ids = tokenizer_output["input_ids"]

        batch_size, max_seq_length = input_ids.size()
        if len(spans_to_mark) != batch_size:
            raise ValueError(f"len(spans_to_mark) ({len(spans_to_mark)}) != batch_size ({batch_size})")  # noqa
        # max_num_markers * 2 because each consists of a start and end token.
        max_levitated_seq_length = max_seq_length + (self.max_num_markers * 2)

        # token_type_ids are just more zeros for every example,
        # so we'll get that out of the way first.
        token_type_ids = tokenizer_output["token_type_ids"]
        new_encodings["token_type_ids"] = torch.hstack(
            (token_type_ids,
             torch.zeros(batch_size, self.max_num_markers * 2,
                         dtype=torch.long)))

        levitated_marker_idxs = [[] for _ in range(batch_size)]
        for example_idx in range(batch_size):
            # Create the empty input_ids, position_ids, and attention_mask
            # which we'll populate below.
            ex_input_ids = input_ids[example_idx]
            # 0 is [PAD]
            num_word_tokens = len([tid for tid in ex_input_ids if tid != 0])
            # By initializing with zeros, any unused marker slots will default
            # to a [PAD] token.
            new_input_ids = torch.cat(
                (ex_input_ids, torch.zeros(self.max_num_markers * 2,
                                           dtype=torch.long)))

            new_position_ids = torch.arange(max_levitated_seq_length, dtype=torch.long)  # noqa

            attention_mask = tokenizer_output["attention_mask"][example_idx]
            new_attention_mask = torch.hstack(
                (attention_mask,
                 torch.zeros(self.max_num_markers * 2,
                             dtype=torch.long))).unsqueeze(0)
            new_attention_mask = new_attention_mask.repeat(
                max_levitated_seq_length, 1)
            # Ensure [PAD] attends to nothing
            new_attention_mask[num_word_tokens:, :] = 0

            marker_start_idxs = list(range(max_seq_length,
                                           max_levitated_seq_length - 1, 2))
            for (marker_start, span) in zip(marker_start_idxs, spans_to_mark[example_idx]):  # noqa
                marker_end = marker_start + 1
                levitated_marker_idxs[example_idx].append(
                    (marker_start, marker_end))

                # Add the marker input_ids
                new_input_ids[marker_start] = start_marker_id
                new_input_ids[marker_end] = end_marker_id

                # Add the position IDs.
                # Levitated markers share position IDs with their spans.
                new_position_ids[marker_start] = span[0]
                new_position_ids[marker_end] = span[1] - 1

                # Populate the attention mask
                # Markers can attend to their partners
                new_attention_mask[marker_start, [marker_start, marker_end]] = 1  # noqa
                new_attention_mask[marker_end, [marker_start, marker_end]] = 1
                # Markers can attend to the text
                new_attention_mask[marker_start, :num_word_tokens] = 1
                new_attention_mask[marker_end, :num_word_tokens] = 1

            new_encodings["input_ids"].append(new_input_ids)
            new_encodings["position_ids"].append(new_position_ids)
            new_encodings["attention_mask"].append(new_attention_mask.unsqueeze(0))  # noqa

        new_encodings["input_ids"] = torch.vstack(new_encodings["input_ids"])
        new_encodings["position_ids"] = torch.vstack(new_encodings["position_ids"])  # noqa
        new_encodings["attention_mask"] = torch.vstack(new_encodings["attention_mask"])  # noqa
        return new_encodings, levitated_marker_idxs


def visualize_attention_matrix(tokens, attn_mat):
    max_token_len = max([len(tok) for tok in tokens])
    header = f"{'':<{max_token_len+2}}"
    for tok in tokens:
        header += f"{tok:<{len(tok)+2}}"
    print(header)
    print('-' * len(header))

    for (i, row) in enumerate(attn_mat):
        row_str = f"{tokens[i]:<{max_token_len+1}}| "
        for (j, item) in enumerate(row):
            row_str += f"{item.item():<{len(tokens[j])+2}}"
        print(row_str)


if __name__ == "__main__":
    print("Available Encoders")
    print("------------------")
    for (k, v) in ENCODER_REGISTRY.items():
        print(f"  '{k}': {v.__name__}")
