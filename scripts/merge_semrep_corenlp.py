import os
import argparse
import xml.etree.ElementTree as ET
from glob import glob

import pybrat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("semmed_brat_dir", type=str,
                        help="""Directory containing .ann files
                                output by scripts/semrep_fact.py""")
    parser.add_argument("corenlp_dir", type=str,
                        help="""Directory containing xml
                                files output by CoreNLP.""")
    parser.add_argument("output_dir", type=str,
                        help="Where to save the output xml files.")
    return parser.parse_args()


def main(args):
    ann_glob = os.path.join(args.semmed_brat_dir, "*.ann")
    ann_files = glob(ann_glob)
    pmid2annfile = {os.path.splitext(os.path.basename(fpath))[0]: fpath
                    for fpath in ann_files}

    xml_glob = os.path.join(args.corenlp_dir, "*.xml")
    xml_files = glob(xml_glob)
    pmid2xmlfile = {os.path.splitext(os.path.basename(fpath))[0]: fpath
                    for fpath in xml_files}

    os.makedirs(args.output_dir, exist_ok=False)
    for (pmid, annfile) in pmid2annfile.items():
        xmlfile = pmid2xmlfile[pmid]
        merged_xml = merge_data(annfile, xmlfile)
        outpath = os.path.join(args.output_dir, f"{pmid}.xml")
        merged_xml.write(outpath)


def merge_data(annfile, xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    anns_as_xml = convert_anns_to_xml(annfile)
    root.extend(anns_as_xml)
    return tree


def convert_anns_to_xml(annfile):
    anns = pybrat.BratAnnotations.from_file(annfile)
    xmlanns = []
    for event in anns.events:
        pred, subj, obj = event.spans
        subj_elem = ET.Element("Subject", attrib={"idref": subj.id})
        obj_elem = ET.Element("Object", attrib={"idref": obj.id})
        event_attribs = {
            "id": event.id,
            "type": pred.type,
            "predicate": pred.id,
            "indicatorType": pred.attributes["indicatorType"].value
        }
        event = ET.Element("Event", attrib=event_attribs)
        event.insert(0, subj_elem)
        event.insert(1, obj_elem)
        xmlanns.append(event)
    for span in anns.spans:
        offset_str = f"{span.start_index}-{span.end_index}"
        span_attribs = {"xml:space": "preserve",
                        "id": span.id,
                        "type": span.type,
                        "charOffset": offset_str,
                        "headOffset": offset_str}
        span_elem = ET.Element("Term", attrib=span_attribs, text=span.text)
        xmlanns.append(span_elem)
    return xmlanns


if __name__ == "__main__":
    args = parse_args()
    main(args)
