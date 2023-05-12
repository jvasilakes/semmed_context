import argparse

from config import config
from data.datasets import SemRepFactDataset


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    summ_parser = subparsers.add_parser(
        "summarize", help="Summarize a dataset")
    summ_parser.add_argument("config", type=str,
                             help="Path to the config file.")

    tar_parser = subparsers.add_parser(
        "tar", help="Create a webdataset tar from .ann and .json files.")
    tar_parser.add_argument("config", type=str,
                            help="Path to the config file.")
    tar_parser.add_argument("outdir", type=str,
                            help="Where to save the tar files.")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    if args.command == "summarize":
        config.load_yaml(args.config)
        ds = SemRepFactDataset.from_config(config)
        ds.summarize()

    elif args.command == "tar":
        config.load_yaml(args.config)
        ds = SemRepFactDataset.from_config(config)
        ds.to_tar(args.outdir)
