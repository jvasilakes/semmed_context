import os
import re
import json
import string
import sqlite3
import argparse

import pandas as pd
from tqdm import tqdm

import pybrat


PRED_COLUMNS = ["p.PMID", "c.PYEAR", "s.SENTENCE_ID", "p.PREDICATION_ID",
                "p.SUBJECT_CUI", "p.SUBJECT_SEMTYPE", "p.SUBJECT_NAME",
                "a.SUBJECT_TEXT", "a.SUBJECT_START_INDEX", "a.SUBJECT_END_INDEX",  # noqa
                "p.PREDICATE", "a.INDICATOR_TYPE",
                "a.PREDICATE_START_INDEX", "a.PREDICATE_END_INDEX",
                "p.OBJECT_CUI", "p.OBJECT_SEMTYPE", "p.OBJECT_NAME",
                "a.OBJECT_TEXT", "a.OBJECT_START_INDEX", "a.OBJECT_END_INDEX",
                "s.SENT_START_INDEX", "s.SENTENCE"]  # needed for getting the predicate text  # noqa

SENT_COLUMNS = ["s.PMID", "s.SENTENCE_ID", "s.TYPE", "s.NUMBER",
                "s.SENT_START_INDEX", "s.SENT_END_INDEX", "s.SENTENCE"]

# Drug repurposing predicates from
# Drug repurposing for COVID-19 via knowledge graph completion
# (Zhang et al., 2021) https://doi.org/10.1016/j.jbi.2021.103696
# These are used by default.
# They are defined in the Jupyter notebook at
# https://github.com/kilicogluh/lbd-covid/blob/master/filtering/COVID_CLF.ipynb
PREDICATES = ['COEXISTS_WITH', 'COMPLICATES', 'MANIFESTATION_OF',
              'PREVENTS', 'PRODUCES', 'TREATS', 'INTERACTS_WITH',
              'STIMULATES', 'INHIBITS', 'CAUSES', 'PREDISPOSES',
              'ASSOCIATED_WITH', 'DISRUPTS', 'AUGMENTS', 'AFFECTS']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("semmeddb_sqlite", type=str,
                        help="Path to semmed40.db")
    parser.add_argument("semtype_mappings_file", type=str,
                        help="Path to SemanticTypes_2018AB.txt")
    parser.add_argument("outdir", type=str,
                        help="Where to save the output.")
    parser.add_argument("--max-predications", type=int, default=10000,
                        help="Maximum number of predications to save.")
    return parser.parse_args()


def main(args):
    # We have to map short forms to long forms (w/o punctuation or whitespace)
    # for the SemRep Factuality pipeline to work properly later.
    os.makedirs(args.outdir, exist_ok=False)
    semtype_map = load_semtype_map(args.semtype_mappings_file)
    db_conn = sqlite3.connect(args.semmeddb_sqlite)
    predications, pmids = query_predicates(db_conn, n=args.max_predications)
    sentences = query_sentences(db_conn, pmids)
    save_predications_to_brat(predications, semtype_map, args.outdir)
    save_sentences_to_json(sentences, args.outdir)


def load_semtype_map(filepath):
    colnames = ["short_form", "code", "long_form"]
    df = pd.read_csv(filepath, header=None, names=colnames, sep='|')
    # {'aapp': "Amino Acid, Peptide, or Protein"}
    punct_regex = re.compile("[{0}]".format(re.escape(string.punctuation)))
    long_form_tokens = df.long_form.map(
        lambda s: punct_regex.sub('', s).split()).tolist()
    long_forms = [''.join([t.title() for t in tokens])
                  for tokens in long_form_tokens]
    semtype_map = dict(zip(df.short_form, long_forms))
    return semtype_map


def query_predicates(db_conn, n=-1):
    pred_columns = ','.join(PRED_COLUMNS)
    predicates = ','.join([f"'{pred}'" for pred in PREDICATES])
    query = f"""
    SELECT {pred_columns}
    FROM PREDICATION AS p
        INNER JOIN SENTENCE AS s
        ON p.SENTENCE_ID = s.SENTENCE_ID
        INNER JOIN PREDICATION_AUX AS a
        ON p.PREDICATION_ID = a.PREDICATION_ID
        INNER JOIN CITATIONS AS c
        ON s.PMID = c.PMID
        WHERE p.PREDICATE IN ({predicates})
    """

    cursor = db_conn.cursor()
    print("Getting predications...", end='', flush=True)

    pmids = set()
    predications = []
    total = n if n > 0 else None
    pbar = tqdm(total=total)
    try:
        cursor.execute(query)
        i = 0
        while True:
            if i == n:
                break
            row = cursor.fetchone()
            if row is None:
                break
            pmids.add(row[0])
            predications.append(row)
            i += 1
            pbar.update()

    except KeyboardInterrupt:
        print("Saving predications fetched so far.")
    cursor.close()
    print("Done", flush=True)
    return predications, pmids


def query_sentences(db_conn, pmids):
    sent_columns = ','.join(SENT_COLUMNS)
    pmids_str = ','.join([f"'{pmid}'" for pmid in pmids])
    query = f"""
    SELECT {sent_columns}
    FROM SENTENCE as s
    WHERE s.PMID IN ({pmids_str})
    """

    cursor = db_conn.cursor()
    print("Getting sentences...", end='', flush=True)

    sentences = []
    try:
        cursor.execute(query)
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            sentences.append(row)

    except KeyboardInterrupt:
        print("Saving sentences fetched so far.")
    cursor.close()
    print("Done", flush=True)
    return sentences


def save_predications_to_brat(predications, semtype_map, outdir):
    colnames = [c[2:] for c in PRED_COLUMNS]  # drop the table alias.
    df = pd.DataFrame(predications, columns=colnames)

    print("saving predications")
    for (pmid, group) in tqdm(df.groupby("PMID")):
        num_spans = 0
        num_events = 0
        num_attrs = 0
        events = []
        sorted_group = group.sort_values("PREDICATE_START_INDEX")
        for (i, row) in sorted_group.iterrows():
            event, num_spans, num_events, num_attrs = get_brat_event_from_row(
                row, semtype_map, num_spans, num_events, num_attrs)
            events.append(event)
        anns = pybrat.BratAnnotations.from_events(events)
        anns.save_brat(outdir=outdir, filename=f"{pmid}.ann")


def save_sentences_to_json(sentences, outdir):
    assert os.path.isdir(outdir)
    colnames = [c[2:] for c in SENT_COLUMNS]  # drop the table alias.
    df = pd.DataFrame(sentences, columns=colnames)

    print("saving sentences")
    for (pmid, group) in tqdm(df.groupby("PMID")):
        seen_sents = set()
        sentence_data = []
        sort_cols = ["TYPE", "NUMBER"]
        sort_ascend = [False, True]
        sorted_group = group.sort_values(sort_cols, ascending=sort_ascend)
        for (i, row) in sorted_group.iterrows():
            sent_num = row.NUMBER
            if row.TYPE == "ab":
                sent_num += 1
            if sent_num not in seen_sents:
                sent = {"sent_index": sent_num,
                        "start_char": row.SENT_START_INDEX,
                        "end_char": row.SENT_END_INDEX,
                        "_text": row.SENTENCE}
                seen_sents.add(sent_num)
                sentence_data.append(sent)

        abstract_text = ''
        prev_end_char = 0
        sentfile = os.path.join(outdir, f"{pmid}.json")
        with open(sentfile, 'w') as outF:
            for sent in sentence_data:
                json.dump(sent, outF)
                outF.write('\n')
                pad_amount = sent["start_char"] - prev_end_char
                abstract_text += ' ' * pad_amount
                prev_end_char = sent["end_char"]
                abstract_text += sent['_text'] + ' '

        txtfile = os.path.join(outdir, f"{pmid}.txt")
        with open(txtfile, 'w') as outF:
            outF.write(abstract_text)


def get_brat_event_from_row(row, semtype_map,
                            num_spans=0, num_events=0, num_attrs=0):
    pred_start = row.PREDICATE_START_INDEX - row.SENT_START_INDEX
    pred_end = row.PREDICATE_END_INDEX - row.SENT_START_INDEX
    pred_text = row.SENTENCE[pred_start:pred_end]

    attr = pybrat.Attribute(f"A{num_attrs}", row.INDICATOR_TYPE,
                            _type="indicatorType")
    num_attrs += 1
    pred = pybrat.Span(f"T{num_spans}", row.PREDICATE_START_INDEX,
                       row.PREDICATE_END_INDEX, pred_text, _type=row.PREDICATE,
                       attributes={"indicatorType": attr})
    num_spans += 1
    long_subj_semtype = semtype_map[row.SUBJECT_SEMTYPE]
    subj = pybrat.Span(f"T{num_spans}", row.SUBJECT_START_INDEX,
                       row.SUBJECT_END_INDEX, row.SUBJECT_TEXT,
                       _type=long_subj_semtype)
    num_spans += 1
    long_obj_semtype = semtype_map[row.OBJECT_SEMTYPE]
    obj = pybrat.Span(f"T{num_spans}", row.OBJECT_START_INDEX,
                      row.OBJECT_END_INDEX, row.OBJECT_TEXT,
                      _type=long_obj_semtype)
    num_spans += 1
    event = pybrat.Event(f"E{num_events}", pred, subj, obj,
                         _type=pred.type, _source_file=f"{row.PMID}.ann")
    num_events += 1
    return event, num_spans, num_events, num_attrs


if __name__ == "__main__":
    args = parse_args()
    main(args)
