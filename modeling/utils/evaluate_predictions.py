import json
import argparse
import warnings

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=str)
    return parser.parse_args()


def main(args):
    labels = []
    preds = []
    for raw_data in open(args.predictions, 'r'):
        data = json.loads(raw_data)
        labels.append(data["json"]["label"])
        preds.append(data["json"]["prediction"])

    sorted_labels = sorted(set(labels))
    cm = confusion_matrix(labels, preds, labels=sorted_labels)
    print_confusion_matrix(cm)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average=None,
                                                 labels=sorted_labels)

    print()
    print(sorted_labels)
    for (metric_name, vals) in zip(["P", "R", "F"], [p, r, f]):
        val_str = ' '.join(f"{val:.4f}" for val in vals)
        print(f"{metric_name}: {val_str}")


def print_confusion_matrix(cm):
    num_labs = cm.shape[0]
    row_data = []
    max_count_len = 0
    for i in range(num_labs):
        count_strs = [str(c) for c in cm[i]]
        max_row_str_len = max([len(sc) for sc in count_strs])
        if max_row_str_len > max_count_len:
            max_count_len = max_row_str_len
        row_data.append(count_strs)

    header = f"T↓/P→  {(' '*max_count_len).join(str(i) for i in range(num_labs))}"  # noqa
    print("\u0332".join(header + "  "))
    for i in range(num_labs):
        row_strs = [f"{s:<{max_count_len}}" for s in row_data[i]]
        print(f"{i}    │ {' '.join(row_strs)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
