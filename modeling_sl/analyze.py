import os
import json
import argparse
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from src.datasets import SemRepFactDataset
from src.distributions import SLBeta, fuse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str)
    parser.add_argument("outdir", type=str)
    return parser.parse_args()


def main(args):
    ds = SemRepFactDataset(datadir=args.datadir, encoder=None,
                           tasks_to_load="all")

    pmids_by_triple = defaultdict(list)
    facts_by_triple = defaultdict(list)
    cui_to_text = defaultdict(list)
    for splitname in ["train", "val", "test"]:
        split = getattr(ds, splitname)
        for example in tqdm(split, desc=f"Processing {splitname}"):
            ex = example["json"]
            pred = ds.INVERSE_LABEL_ENCODINGS["Predicate"][ex["labels"]["Predicate"]]  # noqa
            triple = (ex["subject"][-1], pred, ex["object"][-1])
            triple_str = '_'.join(triple)
            pmids_by_triple[triple_str].append(ex["pmid"])
            facts_by_triple[triple].append(ex["labels"]["Factuality"])
            cui_to_text[ex["subject"][-1]].append(ex["subject"][0])
            cui_to_text[ex["object"][-1]].append(ex["object"][0])

    sorted_by_len = sorted(facts_by_triple.items(), key=lambda x: len(x[1]),
                           reverse=True)
    rows = []
    for (triple, facts) in tqdm(sorted_by_len, desc="Computing consensus"):
        named_triple = [cui_to_text[triple[0]][0], triple[0], triple[1],
                        cui_to_text[triple[2]][0], triple[2]]
        dist, factval = compute_consensus(facts)
        rows.append([*named_triple, dist.b.item(), dist.d.item(),
                     dist.u.item(), factval, len(facts)])

    colnames = ["subj_text", "subj_cui", "predicate", "obj_text", "obj_cui",
                'b', 'd', 'u', "Factuality", "support"]
    df = pd.DataFrame(rows, columns=colnames)
    slt_outfile = os.path.join(args.outdir, "consensus.csv")
    df.to_csv(slt_outfile, index=False)

    pmids_outfile = os.path.join(args.outdir, "pmids.json")
    with open(pmids_outfile, 'w') as outF:
        json.dump(dict(pmids_by_triple), outF)


def compute_consensus(factualities):
    """
    For converting Factuality labels to SL distributions:

      lab  value        b     d     u
      0    Fact         1     0     0
      1    Counterfact  0     1     0
      2    Probable     0.6   0.2   0.2
      3    Possible     0.2   0.2   0.6
      4    Doubtful     0.2   0.6   0.2

    For converting consensus back to Factuality:

      Fact:        0.95 <= b
      Probable:    0.65 < b <= 0.95
      Possible:    0.4 < b <= 0.65
      Doubtful:    0.05 < b <= 0.4
      Counterfact: b <= 0.05
    """
    sle_map = {
        0: {'b': 1.0, 'd': 0.0, 'u': 0.0},
        1: {'b': 0.0, 'd': 1.0, 'u': 0.0},
        2: {'b': 0.6, 'd': 0.2, 'u': 0.2},
        3: {'b': 0.2, 'd': 0.2, 'u': 0.6},
        4: {'b': 0.2, 'd': 0.6, 'u': 0.2}
    }
    fact_map = {0.05: "counterfact",
                0.4:  "doubtful",
                0.75: "possible",
                0.90: "probable",
                1.0:  "fact"}
    dists = [SLBeta(**sle_map[fact]) for fact in factualities]
    consensus_dist = fuse(dists)
    consensus_fact = None
    belief = consensus_dist.mode
    for (upper, factval) in fact_map.items():
        if belief <= upper:
            consensus_fact = factval
            break
    return consensus_dist, consensus_fact


if __name__ == "__main__":
    main(parse_args())
