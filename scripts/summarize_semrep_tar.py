import json
import argparse
import tarfile
from tqdm import tqdm
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tarfile", type=str,
                        help="Directory containing tar files.")
    parser.add_argument("--per-predicate", action="store_true", default=False,
                        help="Compute counts per predicate.")
    return parser.parse_args()


def main(args):
    all_counts = None
    for example in tqdm(load_raw_examples(args.tarfile)):
        if args.per_predicate is True:
            counts = summarize_example_per_predicate(example)
        else:
            counts = summarize_example(example)
        if all_counts is None:
            all_counts = counts
        else:
            update_nested_dict(all_counts, counts)

    print_counts(all_counts)


def load_raw_examples(tarf):
    assert tarfile.is_tarfile(tarf)
    archive = tarfile.open(tarf)
    for member in archive:
        example = json.load(archive.extractfile(member))
        yield example


def summarize_example(example):
    counts = {"num_events": 0,
              "Predicate": defaultdict(int),
              "Factuality": defaultdict(int),
              "Polarity": defaultdict(int),
              "Certainty": defaultdict(int)}

    counts["num_events"] += 1
    counts["Predicate"][example["labels"]["Predicate"]] += 1
    counts["Factuality"][example["labels"]["Factuality"]] += 1
    counts["Polarity"][example["labels"]["Polarity"]] += 1
    counts["Certainty"][example["labels"]["Certainty"]] += 1
    return counts


def summarize_example_per_predicate(example):
    counts = {"num_events": 0,
              "num_attributes": 0,
              "Predicate": defaultdict(lambda: defaultdict(
                  lambda: defaultdict(int)))}

    counts["num_events"] += 1
    pred = example["labels"]["Predicate"]
    for (attr_name, attr_val) in example["labels"].items():
        if attr_name == "Predicate":
            continue
        counts["Predicate"][pred][attr_name][attr_val] += 1
        counts["Predicate"][pred][attr_name][attr_val] += 1
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
