import json
import argparse
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torchtext.data.metrics import bleu_score
from transformers import AutoTokenizer, GPTNeoForCausalLM


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_file", type=str,
                        help="path/to/predictions.jsonl")
    return parser.parse_args()


def main(args):
    raw_preds = [json.loads(line) for line in open(args.predictions_file)]
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

    ppl_model_name = "Abirate/gpt_3_finetuned_multi_x_science"
    ppl_model = GPTNeoForCausalLM.from_pretrained(ppl_model_name).to(DEVICE)
    ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_name)

    preds_by_task = defaultdict(list)
    labels_by_task = defaultdict(list)
    full_recon_text = ""
    full_true_text = ""
    for example in tqdm(raw_preds, desc="Getting examples"):
        full_recon_text = full_recon_text + example["predictions"] + "\n\n"
        full_true_text = full_true_text + example["targets"] + "\n\n"

        preds_by_task["Reconstruction"].append(
            tokenizer.tokenize(example["predictions"]))
        labels_by_task["Reconstruction"].append([
            tokenizer.tokenize(example["targets"])])
        for (task, d) in example["tasks"].items():
            preds_by_task[task].append(d["predictions"])
            labels_by_task[task].append(d["labels"])

    results = {}
    for (task, preds) in tqdm(preds_by_task.items(), desc="tasks"):
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

    recon_ppl = compute_ppl(full_recon_text, ppl_model, ppl_tokenizer)
    true_ppl = compute_ppl(full_true_text, ppl_model, ppl_tokenizer)
    results["Reconstruction"]["recon ppl"] = recon_ppl
    results["Reconstruction"]["true_ppl"] = true_ppl

    format_results_as_markdown(results)


def compute_ppl(text, model, tokenizer):
    encodings = tokenizer(text, return_tensors="pt")
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride), desc="ppl"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which
            # averages over valid labels.
            # N.B. the model only calculates loss over trg_len - 1 labels,
            # because it internally shifts the labels to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.detach().cpu().item()


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
