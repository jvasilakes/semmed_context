import json
import argparse
from collections import defaultdict

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torchtext.data.metrics import bleu_score
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_file", type=str,
                        help="path/to/predictions.jsonl")
    return parser.parse_args()


def main(args):
    raw_preds = [json.loads(line) for line in open(args.predictions_file)]
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

    preds_by_task = defaultdict(list)
    labels_by_task = defaultdict(list)
    for example in raw_preds:
        preds_by_task["Reconstruction"].append(
            tokenizer.tokenize(example["predictions"]))
        labels_by_task["Reconstruction"].append([
            tokenizer.tokenize(example["targets"])])
        for (task, d) in example["tasks"].items():
            preds_by_task[task].append(d["predictions"])
            labels_by_task[task].append(d["labels"])

    results = {}
    for (task, preds) in preds_by_task.items():
        labels = labels_by_task[task]
        if task == "Reconstruction":
            bleu = bleu_score(preds, labels)
            results[task] = {"bleu": bleu}
        else:
            labeldim = len(set(labels))
            average = None
            if labeldim > 2:
                average = "macro"
            p, r, f, _ = precision_recall_fscore_support(
                labels, preds, average=average)
            metrics = {"precision": p,
                       "recall": r,
                       "f1": f}
            results[task] = metrics

    format_results_as_markdown(results)


def format_results_as_markdown(results_dict):
    for (task, metrics) in results_dict.items():
        print(task)
        for (name, val) in metrics.items():
            if isinstance(val, np.ndarray):
                val_str = ' '.join([f"{v:.4f}" for v in val])
                print(f"  {name:<10}: {val_str}")
            else:
                print(f"  {name:<10}: {val:.4f}")
        print()


if __name__ == "__main__":
    main(parse_args())
