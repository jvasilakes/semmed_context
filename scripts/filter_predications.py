import os
import re
import sys
import argparse
from glob import glob
from collections.abc import MutableMapping

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from pybrat import BratAnnotations, BratText

sys.path.append("assets/filtering/src/")
from model import PubMedBERT  # noqa


DEVICE = "cuda" if torch.cuda.is_available else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("anndir", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("outdir", type=str)
    parser.add_argument("-n", type=int, default=-1)
    return parser.parse_args()


def main(args):
    os.makedirs(args.outdir, exist_ok=False)

    print("loading model...", end='', flush=True)
    model = PubMedBERT()
    model.load_state_dict(torch.load(args.model_path))
    model.to(DEVICE)
    model.eval()
    print(f"model loaded to {DEVICE}")
    print("loading tokenizer...", end='', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model.weight_path)
    print("done")

    total_events = 0
    total_filtered_events = 0
    annglob = glob(os.path.join(args.anndir, "*.ann"))
    for (i, annfile) in tqdm(enumerate(annglob), total=len(annglob)):
        try:
            if i == args.n:
                break
            pmidpath = os.path.splitext(annfile)[0]
            txtpath = f"{pmidpath}.txt"
            sentpath = f"{pmidpath}.json"
            if not os.path.isfile(txtpath):
                txtpath = None
            if not os.path.isfile(sentpath):
                sentpath = None
            if (txtpath, sentpath) == (None, None):
                raise OSError(f"Could not find text or sentence files for {annfile}")  # noqa
            anns = BratAnnotations.from_file(annfile)
            anntxt = BratText.from_files(text=txtpath, sentences=sentpath)

            predications = prepare_predications_for_model(anns, anntxt)
            if len(predications) == 0:
                continue
            total_events += len(predications)
            keep_idxs = apply_filtering_model(model, tokenizer, predications)
            filtered_anns = subset_events(anns, keep_idxs)
            num_filtered = len(filtered_anns.events)
            total_filtered_events += num_filtered
            if num_filtered > 0:
                pmid = os.path.basename(pmidpath)
                filtered_anns.save_brat(args.outdir, f"{pmid}.ann")
                # Symlink .txt and .json files from anndir
                src_txtpath = os.path.join(args.anndir, f"{pmid}.txt")
                if os.path.isfile(src_txtpath):
                    src_txtpath = os.path.abspath(src_txtpath)
                    dest_txtpath = os.path.abspath(
                        os.path.join(args.outdir, f"{pmid}.txt"))
                    os.symlink(src_txtpath, dest_txtpath)
                src_sentpath = os.path.join(args.anndir, f"{pmid}.json")
                if os.path.isfile(src_sentpath):
                    src_sentpath = os.path.abspath(src_sentpath)
                    dest_sentpath = os.path.abspath(
                        os.path.join(args.outdir, f"{pmid}.json"))
                    os.symlink(src_sentpath, dest_sentpath)
        except KeyboardInterrupt:
            print("Stopping early")
            break

    ratio = 100 * (total_filtered_events / total_events)
    print(f"Events output: {total_filtered_events}/{total_events} ({ratio:.0f}%)")  # noqa


def prepare_predications_for_model(anns, anntxt):
    examples = []
    for event in anns.events:
        assert len(event.spans) == 3, f"Found event with < 3 spans {event} in {annfilepath}"  # noqa
        try:
            sentence = anntxt.sentences(annotations=[event])[0]["_text"]
        except IndexError:
            continue
        sentence = normalize_text(sentence)
        pred, subj, obj = event.spans
        triple_text = ' '.join((subj.text, pred.type, obj.text))
        example = [sentence, triple_text]
        examples.append(example)

    return examples


def normalize_text(text):
    pattern = re.compile(r'[\W_]+')
    text = pattern.sub(' ', text)
    text = ' '.join(text.split())
    text = text.lower()
    return text


def apply_filtering_model(model, tokenizer, examples):
    sentences = []
    triples = []
    for ex in examples:
        sentences.append(ex[0])
        triples.append(ex[1])
    encoded = tokenizer(sentences, triples, return_tensors="pt",
                        padding=True, truncation=True, verbose=False)
    encoded = send_to_device(encoded, DEVICE)
    outputs = model(**encoded)  # (len(examples), 2)
    keep_idxs = torch.where(outputs.logits.argmax(dim=1) == 1)[0]
    return keep_idxs.tolist()


def subset_events(anns, keep_idxs):
    filtered_events = [anns.events[i] for i in keep_idxs]
    return BratAnnotations.from_events(filtered_events)


def send_to_device(collection, device):
    if torch.is_tensor(collection):
        if collection.device != device:
            return collection.to(device)
    if isinstance(collection, (dict, MutableMapping)):
        for key in collection.keys():
            collection[key] = send_to_device(collection[key], device)
    elif isinstance(collection, (list, tuple, set)):
        for i in range(len(collection)):
            collection[i] = send_to_device(collection[i], device)
    return collection


if __name__ == "__main__":
    args = parse_args()
    main(args)
