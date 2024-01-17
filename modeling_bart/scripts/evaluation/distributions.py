import os
import re
import sys
import json
import argparse
import warnings
from glob import glob
from collections import defaultdict, Counter

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, os.getcwd())
from config_summary import config  # noqa
from vae.models.util import DISTRIBUTION_REGISTRY  # noqa


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="compute or summarize")

    compute_parser = subparsers.add_parser("compute")
    compute_parser.set_defaults(compute=True, summarize=False)
    compute_parser.add_argument("config_file", type=str,
                                help="Config YAML file used in experiment.")
    compute_parser.add_argument("--epoch", type=int, default=-1,
                                help="Epoch to evaluation")
    compute_parser.add_argument("--quiet", action="store_true",
                                default=False)
    compute_parser.add_argument("--num_resamples", type=int, default=10)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.set_defaults(compute=False, summarize=True)
    summarize_parser.add_argument("config_file", type=str,
                                  help="Config YAML file used in experiment.")
    args = parser.parse_args()
    if [args.compute, args.summarize] == [False, False]:
        parser.print_help()
    return args


def compute(args):
    config.load_yaml(args.config_file)
    print(config)

    logdir = os.path.join(config.Experiment.logdir.value,
                          config.Experiment.name.value,
                          f"version_{config.Experiment.version.value}",
                          f"seed_{config.Experiment.random_seed.value}")
    metadata_dir = os.path.join(logdir, "metadata")

    epoch = args.epoch
    if args.epoch == -1:
        file_glob = os.path.join(metadata_dir, "params_*.json")
        files = glob(file_glob)
        regex = re.compile(r'.*/params_([0-9]+)\.json')
        epoch_matches = [regex.match(f) for f in files]
        epochs = [int(match.group(1)) for match in epoch_matches
                  if match is not None]
        if epochs == []:
            raise OSError(f"No parameter files found at {file_glob}")
        epoch = sorted(epochs)[-1]

    labfile = os.path.join(metadata_dir, f"labels_{epoch}.json")
    labels = json.load(open(labfile))

    paramfile = os.path.join(metadata_dir, f"params_{epoch}.json")
    params = json.load(open(paramfile))

    evaldir = os.path.join(logdir, "evaluation")
    os.makedirs(evaldir, exist_ok=True)
    predscore_outf = os.path.join(evaldir, "predictions.jsonl")
    if os.path.exists(predscore_outf):
        raise OSError(f"{predscore_outf} already exists.")
    migs_outf = os.path.join(evaldir, "migs.jsonl")
    if os.path.exists(migs_outf):
        raise OSError(f"{migs_outf} already exists.")
    corrs_outf = os.path.join(evaldir, "rhos.jsonl")
    if os.path.exists(corrs_outf):
        raise OSError(f"{corrs_outf} already exists.")

    for i in trange(args.num_resamples):
        torch.manual_seed(i)
        np.random.seed(i)

        # Informativeness
        # Prediction: predict labels from distributions and compute MIs
        pred_scores, mi_dict, entropy_dict = informativeness(
            config, params, labels, quiet=args.quiet)
        with open(predscore_outf, 'a') as outF:
            json.dump(pred_scores, outF)
            outF.write('\n')

        # Independence (MIG)
        migs = independence(mi_dict, entropy_dict, quiet=args.quiet)
        with open(migs_outf, 'a') as outF:
            json.dump(migs, outF)
            outF.write('\n')

        # Invariance (Pearson's rho)
        # Between each pair of latent distributions
        corrs = invariance(config, params, labels, quiet=args.quiet)
        with open(corrs_outf, 'a') as outF:
            json.dump(corrs, outF)
            outF.write('\n')


def summarize(args):
    config.load_yaml(args.config_file)
    logdir = os.path.join(config.Experiment.logdir.value,
                          config.Experiment.name.value,
                          f"version_{config.Experiment.version.value}",
                          f"seed_{config.Experiment.random_seed.value}")
    pred_scores_file = os.path.join(logdir, "evaluation", "predictions.jsonl")
    pred_data = [json.loads(line.strip()) for line in open(pred_scores_file)]

    migs_file = os.path.join(logdir, "evaluation", "migs.jsonl")
    migs_data = [json.loads(line.strip()) for line in open(migs_file)]

    informativeness_table(pred_data, migs_data)
    plot_migs(migs_data)

    rhos_file = os.path.join(logdir, "evaluation", "rhos.jsonl")
    rhos_data = [json.loads(line.strip()) for line in open(rhos_file)]
    invariance_table(rhos_data)


def informativeness(config, params, labels, quiet=False):
    # {lab_task: dist_task: }
    pred_scores = defaultdict(dict)
    mi_dict = defaultdict(dict)
    entropies_dict = {}

    if quiet is False:
        pbar = tqdm(total=len(labels.items()), desc="Inf.")
    for (lab_task, labs) in labels.items():
        if quiet is False:
            pbar.update(1)
        entropies_dict[lab_task] = compute_entropy_freq(labs)
        for (dist_task, ps) in params.items():
            dist_name = config.Model.latent_structure.value[dist_task][1]
            dist_cls = DISTRIBUTION_REGISTRY[dist_name]

            all_zs = []
            for p_i in ps:
                if dist_task == "content":
                    p_i = {k: torch.as_tensor(v).squeeze()
                           for (k, v) in p_i.items()}
                dist = dist_cls(**p_i)
                all_zs.extend(dist.sample().tolist())
            all_zs_feats = np.array(all_zs)
            if len(all_zs_feats.shape) == 1:
                all_zs_feats = all_zs_feats.reshape(-1, 1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegression(class_weight="balanced",
                                         penalty="none").fit(all_zs_feats, labs)  # noqa
            all_preds = clf.predict(all_zs_feats)

            mis = mutual_info_classif(all_zs_feats, labs,
                                      discrete_features=False)
            mi_dict[lab_task][dist_task] = mis.mean()
            p, r, f, _ = precision_recall_fscore_support(
                    labs, all_preds, average=None, labels=sorted(set(labs)))
            pred_scores[lab_task][dist_task] = {'P': list(p),
                                                'R': list(r),
                                                "F1": list(f)}

    return pred_scores, mi_dict, entropies_dict


def independence(mi_dict, Hvs, quiet=False):
    migs = defaultdict(dict)
    if quiet is False:
        pbar = tqdm(total=len(mi_dict), desc="Ind..")
    for lab_name in tqdm(mi_dict.keys(), desc="Ind."):
        if quiet is False:
            pbar.update(1)
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
                          "MIG": mig_v.item(),
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


def invariance(config, params, labels, quiet=False):
    all_rhos = defaultdict(dict)
    if quiet is False:
        pbar = tqdm(total=len(params), desc="Inv..")
    for (task1, ps1) in params.items():
        if quiet is False:
            pbar.update(1)
        if task1 in ["Certainty", "Polarity"]:
            print("1: ", ps1)
        name1 = config.Model.latent_structure.value[task1][1]
        cls1 = DISTRIBUTION_REGISTRY[name1]

        zs1 = []
        for pi1 in ps1:
            if task1 == "content":
                pi1 = {k: torch.as_tensor(v).squeeze()
                       for (k, v) in pi1.items()}
            dist1 = cls1(**pi1)
            zs1.extend(dist1.sample().tolist())
        zs1 = np.array(zs1)

        for (task2, ps2) in params.items():
            if task2 == task1:
                continue
            if task2 in ["Certainty", "Polarity"]:
                print("2: ", ps1)
                input()
            name2 = config.Model.latent_structure.value[task2][1]
            cls2 = DISTRIBUTION_REGISTRY[name2]

            zs2 = []
            for pi2 in ps2:
                if task2 == "content":
                    pi2 = {k: torch.as_tensor(v).squeeze()
                           for (k, v) in pi2.items()}
                dist2 = cls2(**pi2)
                zs2.extend(dist2.sample().tolist())
            zs2 = np.array(zs2)

            if len(zs1.shape) == 1 and len(zs2.shape) == 1:
                rhos = [pearsonr(zs1, zs2)[0]]
            elif len(zs1.shape) == 2 and len(zs2.shape) == 1:
                rhos = [pearsonr(zs1[:, col], zs2)[0]
                        for col in range(zs1.shape[1])]
            elif len(zs1.shape) == 1 and len(zs2.shape) == 2:
                rhos = [pearsonr(zs1, zs2[:, col])[0]
                        for col in range(zs2.shape[1])]
            else:
                rhos = [pearsonr(zs1[:, col1], zs2[:, col2])[0]
                        for col1 in range(zs1.shape[1])
                        for col2 in range(zs2.shape[1])]
            rho = np.mean(rhos)
            all_rhos[task1][task2] = rho.item()
    return all_rhos


def informativeness_table(pred_data, migs_data):
    combined_metrics = defaultdict(lambda: pd.DataFrame())
    for (i, resample) in enumerate(pred_data):
        for (lab_task, dists_info) in resample.items():
            df = pd.DataFrame()
            for (dist_task, metrics) in dists_info.items():
                macro_avg = pd.DataFrame(metrics).mean(0)
                metrics_df = pd.DataFrame(macro_avg).transpose()
                metrics_df["Latent"] = dist_task
                df = pd.concat([df, metrics_df])
            df.set_index("Latent", inplace=True)
            mi_data = migs_data[i][lab_task]
            mis = dict(zip(mi_data["sorted_latents"], mi_data["sorted_MIs"]))
            sorted_mis = [mis[latent] for latent in df.index]
            df["MI"] = sorted_mis
            df = df[["MI", 'P', 'R', 'F1']]
            combined_metrics[lab_task] = pd.concat(
                [combined_metrics[lab_task], df])

    for (lab_task, df) in combined_metrics.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(lab_task)
            print('=' * len(lab_task))
            print(df.groupby("Latent").agg("mean").round(3).to_latex())


def plot_migs(migs_data):
    migs_dict = defaultdict(list)
    for resample in migs_data:
        for (task, data) in resample.items():
            migs_dict[task].append(data["MIG"])
    migs_df = pd.DataFrame(migs_dict)
    migs_df.boxplot(patch_artist=True, return_type="dict", widths=0.75)
    plt.show()


def invariance_table(rhos_data):
    combined_metrics = pd.DataFrame()
    for (i, resample) in enumerate(rhos_data):
        for (task, rhos) in resample.items():
            df = pd.DataFrame(rhos, index=[i])
            df["Task"] = task
            df.set_index("Task", inplace=True)
            combined_metrics = pd.concat([combined_metrics, df])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_df = combined_metrics.groupby("Task").agg("mean")
        mean_df = mean_df[mean_df.index]
        print(mean_df.round(3).to_latex())


if __name__ == "__main__":
    args = parse_args()
    if args.compute is True:
        compute(args)
    elif args.summarize is True:
        summarize(args)
