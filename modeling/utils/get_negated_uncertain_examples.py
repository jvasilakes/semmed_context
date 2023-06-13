import os
import sys
import argparse

from tqdm import tqdm
import webdataset as wds

sys.path.append(os.path.abspath("./"))
from data.datasets import SemRepFactDataset  # noqa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str)
    parser.add_argument("split", type=str, choices=["train", "val", "test"])
    parser.add_argument("outdir", type=str)
    return parser.parse_args()


def main(args):
    os.makedirs(args.outdir, exist_ok=False)
    ds = SemRepFactDataset(args.datadir)

    examples = getattr(ds, args.split)
    negated = []
    uncertain = []
    for ex in tqdm(examples):
        ex = ds.inverse_transform_labels(ex)
        labs = ex["json"]["labels"]
        if labs["Certainty"] == "Uncertain":
            uncertain.append(ex)
        if labs["Polarity"] == "Negative":
            negated.append(ex)

    neg_outfile = os.path.join(args.outdir, "negated.tar.gz")
    to_tar(negated, neg_outfile)
    unc_outfile = os.path.join(args.outdir, "uncertain.tar.gz")
    to_tar(uncertain, unc_outfile)


def to_tar(examples, outfile):
    with wds.TarWriter(outfile, compress=True) as sink:
        for ex in examples:
            sink.write(ex)


if __name__ == "__main__":
    args = parse_args()
    main(args)
