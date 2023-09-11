import os
import argparse
from glob import glob

from pybrat import BratAnnotations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("orig_dataset", type=str,
                        help="Path to dataset without CUIs.")
    parser.add_argument("new_dataset", type=str,
                        help="Path to dataset with CUIs.")
    parser.add_argument("outdir", type=str,
                        help="Where to save the new ann files.")
    return parser.parse_args()


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    orig_glob = os.path.join(args.orig_dataset, "*.ann")
    for orig_file in glob(orig_glob):
        bn = os.path.basename(orig_file)
        new_file = os.path.join(args.new_dataset, bn)
        orig_anns = BratAnnotations.from_file(orig_file)
        new_anns = BratAnnotations.from_file(new_file)
        for new_span in new_anns.spans:
            if "CUI" not in new_span.attributes.keys():
                continue
            for orig_span in orig_anns.spans:
                if new_span == orig_span:
                    new_attr = new_span.attributes["CUI"]
                    new_attr.reference = orig_span
                    orig_anns.add_annotation(new_attr)
                    break
        orig_anns.save_brat(args.outdir)


if __name__ == "__main__":
    main(parse_args())
