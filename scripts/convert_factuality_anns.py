"""
Convert the single factuality event annotations to two separate
uncertainty and polarity annotations, per table 5 of the SemRep
Factuality paper, reproduced below.


Certainty | Polarity  | Factuality
----------|-----------|-----------
L3        | positive  |  Fact
L2        | positive  |  Probable
L1        | positive  |  Possible
L1 or L2  | negative  |  Doubtful
L3        | negative  |  Counterfact
-----------------------------------

Kilicoglu, H., Rosemblat, G., & Rindflesch, T. C. (2017).
Assigning factuality values to semantic relations extracted from biomedical
research literature. PLOS ONE, 12(7), e0179926.
https://doi.org/10.1371/journal.pone.0179926
"""

import os
import sys
import argparse
import warnings
from glob import glob
from tqdm import tqdm

from pybrat import BratAnnotations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("anndir", type=str,
                        help="Path to dir containing .ann files.")
    parser.add_argument("outdir", type=str,
                        help="Where to save the new annotations.")
    return parser.parse_args()


def main(args):
    os.makedirs(args.outdir, exist_ok=False)
    annglob = os.path.join(args.anndir, "*.ann")
    annfiles = glob(annglob)
    for annfile in tqdm(annfiles, file=sys.stdout):
        new_anns = convert_annotations(annfile)
        new_anns.save_brat(args.outdir)


def convert_annotations(annfile):
    conversion_table = {
        "Fact": {"certainty": "L3",
                 "polarity": "Positive"},
        "Probable": {"certainty": "L2",
                     "polarity": "Positive"},
        "Possible": {"certainty": "L1",
                     "polarity": "Positive"},
        "Doubtful": {"certainty": "L1",
                     "polarity": "Negative"},
        "Counterfact": {"certainty": "L3",
                        "polarity": "Negative"}
    }
    anns = BratAnnotations.from_file(annfile)
    if len(anns.events) == 0:
        # There are no events to convert
        return anns

    max_attr_id = max([int(attr.id.strip('A')) for attr in anns.attributes])

    current_attr_id = max_attr_id + 1
    for event in anns.events:
        if "Factuality" not in event.attributes:
            warnings.warn(f"Event {event} ({annfile}) has no factuality.")
        fact_attr = event.attributes["Factuality"]
        # For now, skip these two.
        if fact_attr.value in ["Uncommitted", "Conditional"]:
            continue
        converted = conversion_table[fact_attr.value]

        cert_attr = fact_attr.copy()
        cert_attr._type = "Certainty"
        cert_attr.value = converted["certainty"]
        cert_attr._id = f"A{current_attr_id}"
        current_attr_id += 1

        pol_attr = fact_attr.copy()
        pol_attr._type = "Polarity"
        pol_attr.value = converted["polarity"]
        pol_attr._id = f"A{current_attr_id}"
        current_attr_id += 1

        event.attributes["Certainty"] = cert_attr
        event.attributes["Polarity"] = pol_attr

    return anns


if __name__ == "__main__":
    args = parse_args()
    main(args)
