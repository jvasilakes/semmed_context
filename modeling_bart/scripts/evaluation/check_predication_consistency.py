import json
import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_file", type=str)
    parser.add_argument("outfile", type=str,
                        help="Where to save correct reconstructions")
    return parser.parse_args()


def main(args):
    preds = [json.loads(line) for line in open(args.predictions_file)]
    correct = []
    entities_found = []
    for pred in preds:
        entities = find_entities(pred["inputs"])
        recon = pred["predictions"].lower()
        found = [e in recon for e in entities]
        entities_found.append(found)
        if sum(found) == 2:
            correct.append(pred)

    entities_found = np.array(entities_found)
    total_found = entities_found.mean()
    at_least_one = (entities_found.sum(1) > 0).mean()
    both = (entities_found.sum(1) == 2).mean()

    print(f"Percentage of entities found: {total_found:.4f}")
    print(f"Percentage of examples with at least one entity found: {at_least_one:.4f}")  # noqa
    print(f"Percentage of examples with both entities found: {both:.4f}")

    with open(args.outfile, 'w') as outF:
        for pred in correct:
            json.dump(pred, outF)
            outF.write('\n')


def find_entities(text):
    entities = []
    start = None
    for (i, c) in enumerate(text):
        if c == "\x14":
            start = i + 1
        elif c == "\x15":
            entities.append(text[start:i].strip().lower())
            start = None
        else:
            pass
    return entities


if __name__ == "__main__":
    main(parse_args())
