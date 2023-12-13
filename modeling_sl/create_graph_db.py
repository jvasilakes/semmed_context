import os
import json
import argparse

import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase

from src.distributions import SLBeta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", type=str)
    parser.add_argument("auth", type=str,
                        help="path to neo4j_auth.txt")
    return parser.parse_args()


def main(driver, args):
    consensus_file = os.path.join(args.indir, "consensus.csv")
    consensus = pd.read_csv(consensus_file)

    pmids_file = os.path.join(args.indir, "pmids.json")
    pmids = json.load(open(pmids_file))

    for (i, row) in tqdm(consensus.iterrows()):

        query = (
            "MERGE (p1:Entity {name: $subj_attrs.name, cui:$subj_attrs.cui}) "
            "MERGE (p2:Entity {name: $obj_attrs.name, cui:$obj_attrs.cui}) "
            f"MERGE (p1)-[:{row.predicate} {{factuality: $pred_attrs.factuality, belief: $pred_attrs.belief, confidence: $pred_attrs.confidence, pmids: $pred_attrs.pmids}}]->(p2) "  # noqa
            "RETURN p1.name, p2.name"
            )
        subj_attrs = {"name": row.subj_text, "cui": row.subj_cui}
        obj_attrs = {"name": row.obj_text, "cui": row.obj_cui}
        triple_str = '_'.join([row.subj_cui, row.predicate, row.obj_cui])

        slbeta = SLBeta(row.b, row.d, row.u)
        belief = slbeta.mode.item()  # most likely value under the Beta
        confidence = slbeta.max_uncertainty().b.item()
        pred_attrs = {"pmids": pmids[triple_str], "factuality": row.Factuality,
                      "belief": belief, "confidence": confidence}
        driver.execute_query(
            query, database_="neo4j",
            subj_attrs=subj_attrs, obj_attrs=obj_attrs, pred_attrs=pred_attrs,
            result_transformer_=lambda r: r.single(strict=True))


def parse_auth(auth_file):
    data = [line.strip() for line in open(auth_file).readlines()
            if not line.startswith('#')]
    auth_params = dict([datum.split('=') for datum in data])
    params = {"uri": auth_params["NEO4J_URI"],
              "auth": (auth_params["NEO4J_USERNAME"],
                       auth_params["NEO4J_PASSWORD"])}
    return params


if __name__ == "__main__":
    args = parse_args()
    auth_params = parse_auth(args.auth)
    with GraphDatabase.driver(**auth_params) as driver:
        main(driver, args)
