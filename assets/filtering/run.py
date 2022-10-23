import os
import csv
import argparse
from collections.abc import MutableMapping

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import Subset, DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, confusion_matrix, \
                            precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.data import SemMedDataset
from src.config import ExperimentConfig
from src.model import BertPredicationFilter


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str,
                        help="path to config file for this experiment.")
    parser.add_argument("--run-train", action="store_true", default=False)
    parser.add_argument("--run-eval", action="store_true", default=False)
    parser.add_argument("--eval-split", type=str, default=None,
                        choices=["train", "dev", "test"],
                        help="If --run-eval, which split to evaluate on.")
    parser.add_argument("--run-predict", action="store_true", default=False)
    parser.add_argument("--predict-datafile", type=str, default=None,
                        help="""If --run-predict, a CSV file of examples
                                to predict.""")
    return parser.parse_args()


def main(args):
    config = ExperimentConfig.from_yaml_file(args.config_file)
    print(config)

    if args.run_train is True:
        os.makedirs(config.checkpoint_dir, exist_ok=False)
        os.makedirs(config.logdir, exist_ok=False)

    print(f"Running on {DEVICE}")
    torch.random.manual_seed(config.random_state)
    np.random.seed(config.random_state)

    full_dataset = SemMedDataset(
        config.datafile,
        bert_model_name_or_path=config.bert_model_name_or_path,
        max_seq_length=config.max_seq_length)

    data_idxs = list(range(len(full_dataset)))
    train_idxs, other_idxs = train_test_split(data_idxs, train_size=0.8,
                                              random_state=config.random_state)
    dev_idxs, test_idxs = train_test_split(other_idxs, test_size=0.5,
                                           random_state=config.random_state)

    train_dataset = Subset(full_dataset, train_idxs)
    dev_dataset = Subset(full_dataset, dev_idxs)
    test_dataset = Subset(full_dataset, test_idxs)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False)

    if args.run_train is True:
        model = BertPredicationFilter(
            bert_model_name_or_path=config.bert_model_name_or_path,
            dropout_prob=config.dropout_prob)
        model = model.to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=config.lr,
                          weight_decay=config.weight_decay)
        train_losses = run_train(model, optimizer, config.epochs,
                                 train_loader, dev_loader,
                                 checkpoint_dir=config.checkpoint_dir)
        plot_losses(train_losses, config.logdir)

    if args.run_eval is True:
        loader_map = {"train": train_loader,
                      "dev": dev_loader,
                      "test": test_loader}
        eval_loader = loader_map[args.eval_split]
        print("Loading model checkpoint")
        model_path = os.path.join(config.checkpoint_dir, "model.pt")
        model = torch.load(model_path)
        model = model.to(DEVICE)
        model.eval()
        eval_results = run_eval(model, eval_loader)
        formatted = format_results(eval_results)
        print(formatted)
        results_path = os.path.join(
            config.logdir, f"{args.eval_split}_results.txt")
        with open(results_path, 'w') as outF:
            outF.write(formatted)

    if args.run_predict is True:
        pred_dataset = SemMedDataset(
            args.predict_datafile,
            bert_model_name_or_path=config.bert_model_name_or_path,
            max_seq_length=config.max_seq_length)
        pred_loader = DataLoader(pred_dataset, batch_size=config.batch_size,
                                 shuffle=False)
        model_path = os.path.join(config.checkpoint_dir, "model.pt")
        model = torch.load(model_path)
        run_predict(model, pred_loader, config.logdir)


def run_train(model, optimizer, epochs, train_loader, dev_loader,
              checkpoint_dir="model_checkpoints"):
    losses = {"train": [],
              "dev": []}
    #best_val_loss_so_far = None
    best_val_f1_so_far = None

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=150,
        num_training_steps=epochs*len(train_loader))

    template = "({0}) loss: {1:.4f}"
    for epoch in range(epochs):
        epoch_losses = []
        pbar = tqdm(total=len(train_loader))
        for (i, batch) in enumerate(train_loader):
            batch = send_to_device(batch, DEVICE)
            optimizer.zero_grad()
            outputs = model(**batch["model_input"])
            loss = model.loss_fn(outputs, batch["label"])
            epoch_losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            loss.backward()
            optimizer.step()
            scheduler.step()

            avg_loss = np.mean(epoch_losses)
            desc = template.format(epoch, avg_loss)
            pbar.set_description(desc)
            pbar.update()

        losses["train"].append(epoch_losses)

        val_results = run_eval(model, dev_loader)
        val_loss = val_results["loss"]
        losses["dev"].append(val_loss)
        val_acc = val_results["accuracy"]
        print(f"(Val) loss: {val_loss:.4f} acc: {val_acc:.4f}")

        #if best_val_loss_so_far is None or val_loss < best_val_loss_so_far:
        #    print(f"Best val loss {best_val_loss_so_far} ==> {val_loss}")
        #    best_val_loss_so_far = val_loss
        #    outpath = os.path.join(checkpoint_dir, "model.pt")
        #    torch.save(model, outpath)

        val_f1 = val_results["f1"]
        if best_val_f1_so_far is None or val_f1 > best_val_f1_so_far:
            print(f"Best val f1 {best_val_f1_so_far} ==> {val_f1}")
            best_val_f1_so_far = val_f1
            outpath = os.path.join(checkpoint_dir, "model.pt")
            torch.save(model, outpath)

    return losses


def run_eval(model, dataloader):
    """
    Run inference and evaluation model on dataloader
    """
    losses = []
    predictions = []
    gold_labels = []

    model.eval()
    for batch in tqdm(dataloader, desc="Eval"):
        batch = send_to_device(batch, DEVICE)
        outputs = model(**batch["model_input"])
        loss = model.loss_fn(outputs, batch["label"])
        losses.append(loss.item())
        preds = model.predict_from_logits(outputs)
        predictions.extend(preds.tolist())
        gold_labels.extend(batch["label"].tolist())

    p, r, f1, _ = precision_recall_fscore_support(gold_labels, predictions,
                                                  average="macro")
    results = {"loss": np.mean(losses),
               "accuracy": accuracy_score(gold_labels, predictions),
               "precision": p,
               "recall": r,
               "f1": f1,
               "confusion_matrix": confusion_matrix(gold_labels, predictions)}
    return results


def run_predict(model, dataloader, logdir):
    model.eval()
    predictions = []
    fields = ["PREDICATE", "SUBJECT_TEXT", "OBJECT_TEXT", "SENTENCE"]
    for batch in tqdm(dataloader, desc="Predict"):
        batch = send_to_device(batch, DEVICE)
        outputs = model(**batch["model_input"])
        preds = model.predict_from_logits(outputs)

        uncollated_metadata = uncollate(batch["metadata"])
        for (raw_example, pred) in zip(uncollated_metadata, preds):
            row = []
            for field in fields:
                row.append(raw_example[field])
            row.append(pred.item())
            predictions.append(row)

    outpath = os.path.join(logdir, "predictions.csv")
    with open(outpath, 'w') as outF:
        writer = csv.writer(outF, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fields + ["LABEL"])
        for row in predictions:
            writer.writerow(row)
    print(f"Predictions saved to {outpath}")


def plot_losses(losses_by_split, outdir):
    train_avg_epoch_losses = np.mean(losses_by_split["train"], axis=1)
    epoch_x = np.arange(0, len(losses_by_split["train"]))
    train_losses_by_epoch = np.array(losses_by_split["train"])
    flat_losses = train_losses_by_epoch.flatten()
    flat_x = np.linspace(0, len(train_losses_by_epoch)-1, len(flat_losses))
    val_avg_epoch_losses = np.array(losses_by_split["dev"])
    plt.plot(epoch_x, train_avg_epoch_losses, label="(train) epoch loss")
    plt.plot(flat_x, flat_losses, label="(train) batch loss", alpha=0.3)
    plt.plot(epoch_x, val_avg_epoch_losses, label="(dev) epoch loss")
    plt.xticks(epoch_x)
    plt.legend()
    outpath = os.path.join(outdir, "losses.pdf")
    plt.savefig(outpath)


def format_results(results_dict):
    loss_str = f"Loss: {results_dict['loss']:.4f}"
    accuracy_str = f"Accuracy: {results_dict['accuracy']:.4f}"
    precision_str = f"Precision: {results_dict['precision']:.4f}"
    recall_str = f"Recall: {results_dict['recall']:.4f}"
    f1_str = f"F1: {results_dict['f1']:.4f}"
    prf_str = '  '.join([precision_str, recall_str, f1_str])
    conf_mat_str = f"  | {'0':<3} {'1':<3}\n"
    conf_mat_str += '-' * (len(conf_mat_str) - 1)
    conf_mat_str += '\n'
    for (label, row) in enumerate(results_dict["confusion_matrix"]):
        conf_mat_str += f"{label:<1} | {row[0]:<3} {row[1]:<3}\n"
    conf_mat_str = conf_mat_str.rstrip()
    conf_mat_str = "Confusion Matrix\nrows=True, cols=Predicted\n" + conf_mat_str  # noqa
    formatted = '\n\n'.join([loss_str, accuracy_str, prf_str, conf_mat_str])
    return "RESULTS\n=======\n" + formatted


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


def uncollate(batch):
    """
    Modified from
    https://lightning-flash.readthedocs.io/en/stable/_modules/flash/core/data/batch.html#default_uncollate  # noqa

    This function is used to uncollate a batch into samples.
    The following conditions are used:

    - if the ``batch`` is a ``dict``, the result will be a list of dicts
    - if the ``batch`` is list-like, the result is guaranteed to be a list

    Args:
        batch: The batch of outputs to be uncollated.

    Returns:
        The uncollated list of predictions.

    Raises:
        ValueError: If input ``dict`` values are not all list-like.
        ValueError: If input ``dict`` values are not all the same length.
        ValueError: If the input is not a ``dict`` or list-like.
    """
    def _is_list_like_excluding_str(x):
        if isinstance(x, str):
            return False
        try:
            iter(x)
        except TypeError:
            return False
        return True

    if isinstance(batch, dict):
        if any(not _is_list_like_excluding_str(sub_batch)
               for sub_batch in batch.values()):
            raise ValueError("When uncollating a dict, all sub-batches (values) are expected to be list-like.")  # noqa
        uncollated_vals = [uncollate(val) for val in batch.values()]
        if len(set([len(v) for v in uncollated_vals])) > 1:
            uncollated_keys_vals = [(key, uncollate(val))
                                    for (key, val) in batch.items()]
            print([(k, len(v)) for (k, v) in uncollated_keys_vals])
            raise ValueError("When uncollating a dict, all sub-batches (values) are expected to have the same length.")  # noqa
        elements = list(zip(*uncollated_vals))
        return [dict(zip(batch.keys(), element)) for element in elements]
    if isinstance(batch, (list, tuple, torch.Tensor)):
        return list(batch)
    raise ValueError(
        "The batch of outputs to be uncollated is expected to be a `dict` or list-like "  # noqa
        f"(e.g. `Tensor`, `list`, `tuple`, etc.), but got input of type: {type(batch)}"  # noqa
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
