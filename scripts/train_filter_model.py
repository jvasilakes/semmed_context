import io
import os
import sys
import json
import random
import tarfile
import argparse
from glob import glob
from collections import defaultdict
from collections.abc import MutableMapping

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix

sys.path.append("assets/filtering/src")
from model import PubMedBERT  # noqa


DEVICE = "cuda" if torch.cuda.is_available else "cpu"

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 3
LR = 5e-6
CV_FOLDS = 20
RANDOM_SEED = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str,
                        help="Directory containing .json example files.")
    parser.add_argument("--labelfile", type=str, default=None,
                        help="CSV file containing binary labels.")
    parser.add_argument("--task", type=str, required=True,
                        help="Task to train on. Must occur in json examples.")
    parser.add_argument("--logdir", type=str, required=True,
                        help="Where to save checkpoints, etc.")
    parser.add_argument("--train_full", action="store_true", default=False,
                        help="""Train a model on the full dataset.""")
    parser.add_argument("--eval", action="store_true", default=False,
                        help="""Evaluate existing model on labelfile.
                                Model checkpoint taken from logdir.""")
    parser.add_argument("--reannotate", action="store_true", default=False,
                        help="""Re-annotate a dataset using predictions from
                                the trained model checkpoint in logdir.""")
    return parser.parse_args()


def main(args):
    set_seed(RANDOM_SEED)
    model = PubMedBERT()
    tokenizer = AutoTokenizer.from_pretrained(model.weight_path)
    raw_data = list(load_raw_examples(args.datadir))
    keep_label = None
    if args.reannotate is True:
        if args.task == "Polarity":
            keep_label = "Negative"
        elif args.task == "Certainty":
            keep_label = "Uncertain"
    examples = load_data(raw_data, args.task, labelfile=args.labelfile,
                         keep_label=keep_label)
    dataset = BertDataset(examples, tokenizer, args.task)

    if args.eval is True:
        model = PubMedBERT()
        model.to(DEVICE)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        model_ckpt = os.path.join(args.logdir, "checkpoints/best_model.pth")
        model.load_state_dict(torch.load(model_ckpt))
        avg_loss, avg_acc = run_evaluation(model, dataloader, verbose=True)
        print(f"Loss: {avg_loss:.4f}  Accuracy: {avg_acc:.4f}")
    elif args.reannotate is True:
        model = PubMedBERT()
        model.to(DEVICE)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        model_ckpt = os.path.join(args.logdir, "checkpoints/best_model.pth")
        model.load_state_dict(torch.load(model_ckpt))
        if args.task == "Polarity":
            label_set = {"Positive", "Negative"}
        elif args.task == "Certainty":
            label_set = {"Certain", "Uncertain"}
        else:
            raise ValueError(f"Unsupported task {args.task}")
        reannotate(model, dataloader, raw_data, args.task, args.datadir,
                   label_set)
    elif args.train_full is True:
        model = PubMedBERT()
        model.to(DEVICE)
        os.makedirs(args.logdir, exist_ok=False)
        hparams = {"epochs": EPOCHS,
                   "learning_rate": LR,
                   "batch_size": BATCH_SIZE,
                   "cv_folds": CV_FOLDS,
                   "random_seed": RANDOM_SEED}
        print(hparams)
        hparams_file = os.path.join(args.logdir, "hparams.json")
        with open(hparams_file, 'w') as outF:
            json.dump(hparams, outF)
        ckpt_dir = os.path.join(args.logdir, "checkpoints")
        os.makedirs(ckpt_dir)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        logfile = os.path.join(
            ckpt_dir, "best_model_metadata_full.json")
        loss, acc = run_train(model, dataloader, dataloader, logfile)
        ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Train loss: {loss:.4f}")
    else:
        os.makedirs(args.logdir, exist_ok=False)
        hparams = {"epochs": EPOCHS,
                   "learning_rate": LR,
                   "batch_size": BATCH_SIZE,
                   "cv_folds": CV_FOLDS,
                   "random_seed": RANDOM_SEED}
        print(hparams)
        hparams_file = os.path.join(args.logdir, "hparams.json")
        with open(hparams_file, 'w') as outF:
            json.dump(hparams, outF)
        ckpt_dir = os.path.join(args.logdir, "checkpoints")
        os.makedirs(ckpt_dir)
        all_idxs = torch.randperm(len(dataset)).tolist()
        losses = []
        accuracies = []
        for (fold, val_idxs) in enumerate(np.array_split(all_idxs, CV_FOLDS)):
            train_idxs = list(set(all_idxs).difference(set(val_idxs)))
            train_ds = Subset(dataset, train_idxs)
            val_ds = Subset(dataset, val_idxs)
            model = PubMedBERT()
            model.to(DEVICE)
            print(f"Fold {fold}")
            print("===========")
            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True)
            val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE,
                                shuffle=False)
            logfile = os.path.join(
                ckpt_dir, f"best_model_metadata_fold{fold}.json")
            loss, acc = run_train(model, train_dl, val_dl, logfile)
            losses.append(loss)
            accuracies.append(acc)
            print()

        avg_loss = torch.as_tensor(losses).mean().item()
        avg_acc = torch.as_tensor(accuracies).mean().item()
        metrics_str = f"Average val loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}"  # noqa
        print(metrics_str)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)


def run_train(model, train_dataloader, val_dataloader, logfile):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = torch.inf
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            opt.zero_grad()
            batch = send_to_device(batch, DEVICE)
            output = model(**batch["encoded"], labels=batch["correct"])
            output.loss.backward()
            opt.step()
            epoch_losses.append(output.loss.detach().cpu())
            avg_train_loss = torch.as_tensor(epoch_losses).mean()
            pbar.set_description(f"{epoch} loss: {avg_train_loss:.4f}")
        train_loss, train_acc = run_evaluation(model, train_dataloader)
        val_loss, val_acc = run_evaluation(model, val_dataloader)
        print(f"Validation loss: {val_loss:.4f} Accuracy: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            metadata = {"epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc}
            with open(logfile, 'w') as outF:
                json.dump(metadata, outF)
            # Bold text
            print("\033[1m Saved new best model\033[0m")
    return best_val_loss, best_val_acc


def run_evaluation(model, dataloader, verbose=False):
    model.eval()
    all_preds = []
    all_labs = []
    losses = []
    for batch in tqdm(dataloader):
        batch = send_to_device(batch, DEVICE)
        output = model(**batch["encoded"], labels=batch["correct"])
        losses.append(output.loss.detach().cpu().item())
        preds = output.logits.argmax(1)
        labs = batch["correct"]
        all_preds.append(preds)
        all_labs.append(labs)

    all_preds = torch.cat(all_preds).detach().cpu().numpy()
    all_labs = torch.cat(all_labs).detach().cpu().numpy()
    if verbose is True:
        cm = confusion_matrix(all_labs, all_preds)
        print("Confusion Matrix")
        print_confusion_matrix(cm)
    accuracy = (all_preds == all_labs).mean()
    return torch.as_tensor(losses).mean().item(), accuracy.item()


def reannotate(model, dataloader, raw_examples, task, datadir, label_set):
    raw_examples = dict(raw_examples)
    model.eval()
    preds_by_eid = {}
    for batch in tqdm(dataloader, desc="Reannotating"):
        batch = send_to_device(batch, DEVICE)
        output = model(**batch["encoded"])
        preds = output.logits.argmax(1)
        preds = preds.detach().cpu().tolist()
        preds_by_eid.update(dict(zip(batch["__key__"], preds)))

    if len(label_set) != 2:
        raise ValueError(f"Only binary labels are supported! Got '{label_set}'.")  # noqa
    os.rename(datadir, f"{datadir}.orig")
    reannotated = 0
    with tarfile.open(datadir, "w:gz") as tarF:
        for (eid, example) in raw_examples.items():
            try:
                pred = preds_by_eid[eid]
            except KeyError:
                pred = -1
            if pred == -1:  # not predicted
                raw_examples[eid]["predicate"] = raw_examples[eid]["labels"]["Predicate"]  # noqa
                raw_examples[eid]["correct"] = pred
            elif pred == 0:  # incorrect
                reannotated += 1
                bad_label = raw_examples[eid]["labels"][task]
                fixed_label = list(label_set - set([bad_label]))[0]
                raw_examples[eid]["labels"][task] = fixed_label
            data = json.dumps(raw_examples[eid]).encode()
            info = tarfile.TarInfo(name=f"{eid}.json")
            info.size = len(data)
            tarF.addfile(info, io.BytesIO(data))
    print(f"Reannotated {reannotated} samples.")


def load_raw_examples(datadir):
    if tarfile.is_tarfile(datadir):
        tarf = tarfile.open(datadir)
        for member in tqdm(tarf, desc="Loading data"):
            eid = os.path.splitext(member.name)[0]
            example = json.load(tarf.extractfile(member))
            yield eid, example
    else:
        example_files = glob(os.path.join(datadir, "*.json"))
        for f in example_files:
            eid = os.path.basename(os.path.splitext(f)[0])
            example = json.load(f)
            yield eid, example


def load_data(raw_examples, task, labelfile=None, keep_label=None):
    labels = defaultdict(lambda: -1)
    if labelfile is not None:
        labels = pd.read_csv(labelfile, index_col=0, header=None)
        labels = pd.get_dummies(labels[1])
        labels = labels.drop(columns='n').to_dict()['y']
    examples = {}
    for (eid, example) in raw_examples:
        if keep_label is not None:
            if example["labels"][task] != keep_label:
                continue
        example["predicate"] = example["labels"]["Predicate"]
        example["correct"] = labels[eid]
        examples[eid] = example
    return examples


class BertDataset(Dataset):

    def __init__(self, examples, tokenizer, task):
        self.tokenizer = tokenizer
        self.task = task
        self.max_seq_length = 256
        self.examples = self.tokenize_examples(examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def tokenize_examples(self, examples):
        data = []
        for (ex_id, ex) in tqdm(examples.items(), desc="Tokenizing"):
            text = ex["text"]
            label_text = self._label_text[self.task][ex["labels"][self.task]]
            pred_text = ' '.join([t.strip('S')
                                  for t in ex["predicate"].split('_')])
            predication = ' '.join([ex["subject"][0],
                                    label_text,
                                    pred_text,
                                    ex["object"][0]
                                    ])
            encoded = self.tokenizer(
                text, predication, max_length=self.max_seq_length,
                truncation=True, padding="max_length", return_tensors="pt")
            encoded = {k: v.squeeze() for (k, v) in encoded.items()}
            datum = {"__key__": ex_id, "encoded": encoded,
                     "correct": ex["correct"]}
            data.append(datum)
        return data

    @property
    def _label_text(self):
        return {"Polarity": {"Positive": "does",
                             "Negative": "does not"},
                "Certainty": {"Certain": "does",
                              "Uncertain": "might"}
                }


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
    else:
        try:
            collection = torch.as_tensor(collection)
        except TypeError:  # torch does not support this type as a tensor
            pass  # keep the collection as it is.
    return collection


def print_confusion_matrix(cm):
    num_labs = cm.shape[0]
    row_data = []
    max_count_len = 0
    for i in range(num_labs):
        count_strs = [str(c) for c in cm[i]]
        max_row_str_len = max([len(sc) for sc in count_strs])
        if max_row_str_len > max_count_len:
            max_count_len = max_row_str_len
        row_data.append(count_strs)

    header = f"T↓/P→  {(' '*max_count_len).join(str(i) for i in range(num_labs))}"  # noqa
    print("\u0332".join(header + "  "))
    for i in range(num_labs):
        row_strs = [f"{s:<{max_count_len}}" for s in row_data[i]]
        print(f"{i}    │ {' '.join(row_strs)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
