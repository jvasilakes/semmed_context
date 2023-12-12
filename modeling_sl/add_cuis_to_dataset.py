import os
import argparse
from glob import glob
from tqdm import tqdm

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
    total_orig_spans = 0
    total_new_spans = 0
    for orig_file in tqdm(glob(orig_glob)):
        bn = os.path.basename(orig_file)
        new_file = os.path.join(args.new_dataset, bn)
        orig_anns = BratAnnotations.from_file(orig_file)
        subj_obj_spans = [span for event in orig_anns.events for span in event.spans[1:]]
        #subj_obj_spans = [span for span in orig_anns.spans
        #                  if "indicatorType" not in span.attributes]
        total_orig_spans += len(subj_obj_spans)
        try:
            new_anns = BratAnnotations.from_file(new_file)
        except FileNotFoundError:
            continue
        for event in new_anns.events:
            for new_span in event.spans[1:]:  # only subject and object
                if "CUI" not in new_span.attributes.keys():
                    continue
                for orig_span in subj_obj_spans:
                    if new_span == orig_span:
                        total_new_spans += 1
                        new_attr = new_span.attributes["CUI"]
                        new_attr.reference = orig_span
                        orig_anns.add_annotation(new_attr)
                        break

#        for new_span in new_anns.spans:
#            if "CUI" not in new_span.attributes.keys():
#                continue
#            for orig_span in orig_anns.spans:
#                if new_span == orig_span:
#                    total_new_spans += 1
#                    new_attr = new_span.attributes["CUI"]
#                    new_attr.reference = orig_span
#                    orig_anns.add_annotation(new_attr)
#                    break
        orig_anns.save_brat(args.outdir)
    print("Number of original spans found in new spans:")
    ratio = 100 * (total_new_spans / total_orig_spans)
    print(f"  {total_new_spans}/{total_orig_spans} ({ratio:.0f}%)")


if __name__ == "__main__":
    main(parse_args())
