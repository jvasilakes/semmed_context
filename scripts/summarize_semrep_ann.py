import os
import argparse
from glob import glob
from tqdm import tqdm
from collections import defaultdict

from pybrat import BratAnnotations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("anndir", type=str,
                        help="Directory containing .ann files.")
    parser.add_argument("--per-predicate", action="store_true", default=False,
                        help="Compute counts per predicate.")
    return parser.parse_args()


def main(args):
    assert os.path.isdir(args.anndir)
    annglob = glob(os.path.join(args.anndir, "*.ann"))
    all_counts = None
    for annfile in tqdm(annglob):
        if args.per_predicate is True:
            counts = summarize_brat_file_per_predicate(annfile)
        else:
            counts = summarize_brat_file(annfile)
        if all_counts is None:
            all_counts = counts
        else:
            update_nested_dict(all_counts, counts)

    print_counts(all_counts)


def summarize_brat_file(annfile):
    counts = {"num_events": 0,
              "num_attributes": 0,
              "Predicate": defaultdict(int),
              "Factuality": defaultdict(int),
              "Polarity": defaultdict(int),
              "Certainty": defaultdict(int)}

    anns = BratAnnotations.from_file(annfile)
    for event in anns.events:
        counts["Predicate"][event.type] += 1
        counts["num_events"] += 1
        for (attr_name, attr) in event.attributes.items():
            counts["num_attributes"] += 1
            if attr_name in counts.keys():
                counts[attr_name][attr.value] += 1
    return counts


def summarize_brat_file_per_predicate(annfile):
    counts = {"num_events": 0,
              "num_attributes": 0,
              "Predicate": defaultdict(lambda: defaultdict(
                  lambda: defaultdict(int)))}

    anns = BratAnnotations.from_file(annfile)
    for event in anns.events:
        counts["num_events"] += 1
        for (attr_name, attr) in event.attributes.items():
            counts["num_attributes"] += 1
            counts["Predicate"][event.type][attr_name][attr.value] += 1
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


def print_counts(counts, indent=0, num_events=None):
    if num_events is None:
        num_events = counts["num_events"]
    for (key, count) in counts.items():
        if isinstance(count, dict):
            print(f"{' ' * indent}{key}:")
            print_counts(counts[key], indent=indent+2, num_events=num_events)
        else:
            try:
                percent = 100 * (count / num_events)
            except ZeroDivisionError:
                percent = 0
            print(f"{' ' * indent}{key}: {count} ({percent:.2f}%)")


if __name__ == "__main__":
    args = parse_args()
    main(args)
