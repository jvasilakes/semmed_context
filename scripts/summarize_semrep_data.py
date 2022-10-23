import os
import argparse
from glob import glob
from collections import defaultdict

from pybrat import BratAnnotations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("anndir", type=str,
                        help="Directory containing .ann files.")
    return parser.parse_args()


def main(args):
    annglob = glob(os.path.join(args.anndir, "*.ann"))
    all_counts = None
    for annfile in annglob:
        counts = summarize_brat_file(annfile)
        if all_counts is None:
            all_counts = counts
        else:
            update_nested_dict(all_counts, counts)

    print_counts(all_counts)


def summarize_brat_file(annfile):
    counts = {"num_events": 0,
              "num_attributes": 0,
              "Factuality": defaultdict(int),
              "Polarity": defaultdict(int),
              "Certainty": defaultdict(int)}

    anns = BratAnnotations.from_file(annfile)
    for event in anns.events:
        counts["num_events"] += 1
        for (attr_name, attr) in event.attributes.items():
            counts["num_attributes"] += 1
            if attr_name in counts.keys():
                counts[attr_name][attr.value] += 1
    return counts


def update_nested_dict(orig_dict, update_dict):
    """
    Modifies orig_dict in place
    """
    for (key, value) in update_dict.items():
        if isinstance(value, dict):
            update_nested_dict(orig_dict[key], update_dict[key])
        else:
            orig_dict[key] += update_dict[key]


def print_counts(counts, indent=0):
    for (key, count) in counts.items():
        if isinstance(count, dict):
            print(f"{' ' * indent}{key}:")
            print_counts(counts[key], indent=indent+2)
        else:
            print(f"{' ' * indent}{key}: {count}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
