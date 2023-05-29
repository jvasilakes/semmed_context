import os
import sys
import argparse

sys.path.append(os.path.abspath("./"))
from data.datasets import SemRepFactDataset  # noqa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("anndir", type=str,
                        help="Directory containing .ann and .json files.")
    parser.add_argument("outdir", type=str,
                        help="Where to save the .tar.gz files.")
    return parser.parse_args()


def main(args):
    ds = SemRepFactDataset(args.anndir)
    ds.save(args.outdir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
