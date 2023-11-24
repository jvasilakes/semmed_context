import os
import re
import sys
import json
import argparse
import warnings
from glob import glob
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, os.getcwd())
from config_summary import config  # noqa
from vae.models.util import DISTRIBUTION_REGISTRY  # noqa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str,
                        help="Config YAML file used in experiment.")
    parser.add_argument("--epoch", type=int, default=-1,
                        help="Epoch to evaluation")
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()


def main(args):
    config.load_yaml(args.config_file)
    print(config)

    logdir = os.path.join(config.Experiment.logdir.value,
                          config.Experiment.name.value,
                          f"version_{config.Experiment.version.value}",
                          f"seed_{config.Experiment.random_seed.value}")
    metadata_dir = os.path.join(logdir, "metadata")

    epoch = args.epoch
    if args.epoch == -1:
        file_glob = os.path.join(metadata_dir, "z_*.json")
        files = glob(file_glob)
        regex = re.compile(r'.*/z_([0-9]+)\.json')
        epochs = [int(regex.match(f).group(1)) for f in files]
        epoch = sorted(epochs)[-1]

    labfile = os.path.join(metadata_dir, f"labels_{epoch}.json")
    labels = json.load(open(labfile))

    paramfile = os.path.join(metadata_dir, f"params_{epoch}.json")
    params = json.load(open(paramfile))

    for i in args.num_resamples():
        # Informativeness
        # Prediction: predict labels from distributions and compute MIs
        pred_scores, mi_dict, entropy_dict = informativeness(
            config, params, labels, random_seed=i, verbose=args.verbose)

        # Independence (MIG)
        migs = disentanglement(mi_dict, entropy_dict)

        # Invariance (Pearson's rho)
        # Between each pair of latent distributions

    df = pd.DataFrame({'P': p, 'R': r, 'F1': f})
    avg_df = pd.DataFrame(df.mean(0)).transpose()
    avg_df["MI"] = mis.mean()

    header_str = f"{dist_task} -> {lab_task}"
    print(header_str)
    print('=' * len(header_str))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if verbose is True:
            print(df.to_latex())
            print()
        print(avg_df.to_latex(index=False))
    print()
    df = pd.DataFrame({task: val["MIG"] for (task, val) in migs.items()})

def informativeness(config, params, labels, random_seed=0, verbose=False):
    torch.manual_seed(random_seed)
    # {lab_task: dist_task: }
    pred_scores = defaultdict(dict)
    mi_dict = defaultdict(dict)
    entropies_dict = {}
    for (lab_task, labs) in tqdm(labels.items()):
        entropies_dict[lab_task] = compute_entropy_freq(labs)
        for (dist_task, ps) in params.items():
            if dist_task == "content":
                continue
            dist_name = config.Model.latent_structure.value[dist_task][1]
            dist_cls = DISTRIBUTION_REGISTRY[dist_name]

            all_zs = []
            for p_i in ps:
                dist = dist_cls(**p_i)
                all_zs.extend(dist.sample().tolist())
            all_zs_feats = np.array(all_zs)
            if len(all_zs_feats.shape) == 1:
                all_zs_feats = all_zs_feats.reshape(-1, 1)

            clf = LogisticRegression(
                random_state=random_seed, class_weight="balanced",
                penalty="none").fit(all_zs_feats, labs)
            all_preds = clf.predict(all_zs_feats)

            mis = mutual_info_classif(all_zs_feats, labs,
                                      discrete_features=False)
            mi_dict[lab_task][dist_task] = mis.mean()
            p, r, f, _ = precision_recall_fscore_support(
                    labs, all_preds, average=None, labels=sorted(set(labs)))
            pred_scores[lab_task][dist_task] = {'P': p, 'R': r, 'F1': f}

    return pred_scores, mi_dict, entropies_dict


def disentanglement(mi_dict, Hvs):
    migs = defaultdict(dict)
    for lab_name in mi_dict.keys():
        lab_mis = []
        latent_names = []
        for latent_name in mi_dict[lab_name].keys():
            mi = mi_dict[lab_name][latent_name]
            lab_mis.append(mi)
            latent_names.append(latent_name)
        sorted_pairs = sorted(zip(lab_mis, latent_names),
                              key=lambda x: x[0], reverse=True)
        sorted_lab_mis, sorted_names = zip(*sorted_pairs)
        Hv = Hvs[lab_name]
        mig_v = (sorted_lab_mis[0] - sorted_lab_mis[1]) / Hv
        migs[lab_name] = {"sorted_latents": sorted_names,
                          "MIG": mig_v,
                          "sorted_MIs": sorted_lab_mis,
                          "label_entropy": Hv}
    return migs


def compute_entropy_freq(xs, mean=True):
    xs = np.array(xs)
    counts = Counter(xs)
    probs = np.array([counts[x]/len(xs) for x in xs])
    if mean is True:
        probs = [np.mean(probs[xs == x]) for x in set(xs)]
    else:
        probs = probs / np.sum(probs)
    H = -np.sum(probs * np.log(probs))
    return H


if __name__ == "__main__":
    main(parse_args())
