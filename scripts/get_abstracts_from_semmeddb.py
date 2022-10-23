import os
import json
import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("semmeddb_output", type=str,
                        help="Path to file output by scripts/semrep_fact.py")
    parser.add_argument("outdir", type=str,
                        help="Directory in which to save the output.")
    return parser.parse_args()


def main(args):
    df = pd.read_csv(args.semmeddb_output)
    for (pmid, group) in df.groupby("PMID"):
        sentences = get_sentences(group)
        outfile = os.path.join(args.outdir, f"{pmid}.json")
        save_sentences(sentences, outfile)


def get_sentences(df_group):
    sentences = []
    seen = set()
    for row in df_group.sort_values("NUMBER"):
        if row.NUMBER in seen:
            continue
        sent = {"sent_index": row.NUMBER,
                "start_index": row.SENT_START_INDEX,
                "end_index": row.SENT_END_INDEX,
                "_text": row.SENTENCE}
        sentences.append(sent)
    return sentences


def save_sentences(sentences, outfile):
    with open(outfile, 'w') as outF:
        for sent in sentences:
            json.dump(sent, outF)
            outF.write('\n')


if __name__ == "__main__":
    args = parse_args()
    main(args)
