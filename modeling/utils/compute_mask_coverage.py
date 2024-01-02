import json
import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_file", type=str,
                        help=".jsonl file containing token mask predictions.")
    return parser.parse_args()


def main(args):
    preds = [json.loads(line) for line in open(args.predictions_file)]
    masks = [np.array(pred["json"]["token_mask"]) for pred in preds]
    coverage = [(mask > 0.0).mean() for mask in masks]
    print(np.mean(coverage), np.std(coverage))


if __name__ == "__main__":
    main(parse_args())
