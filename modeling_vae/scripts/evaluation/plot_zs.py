import os
import re
import sys
import argparse
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


sys.path.insert(0, os.getcwd())
from config import config  # noqa
from vae import data as D  # noqa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--data_split", type=str,
                        choices=["train", "val", "test"])
    parser.add_argument("--epoch", type=int, default=-1)
    return parser.parse_args()


def main(args):
    config.load_yaml(args.config_file)
    metadata_dir = os.path.join(config.Experiment.checkpoint_dir.value,
                                config.Experiment.name.value,
                                "metadata")
    zs_dir = os.path.join(metadata_dir, 'z')
    if args.epoch == -1:
        epoch = get_last_epoch(zs_dir)
    else:
        epoch = args.epoch

    z_files = glob(os.path.join(zs_dir, f"{args.data_split}_*_{epoch}.log"))
    latent_names = get_latent_names(z_files)

    ids_dir = os.path.join(metadata_dir, "ordered_ids")
    ids_file = os.path.join(ids_dir, f"{args.data_split}_{epoch}.log")
    ids = [uuid.strip() for uuid in open(ids_file)]

    # id2labels = {uuid: {latent_name: value for latent_name in latent_names}}
    # But this will not include the "content" latent.
    # labels_set = {lname for lname in latent_names if lname is a supervised latent}  # noqa
    id2labels, labels_set = get_labels(config, args.data_split, latent_names)
    Vs = defaultdict(list)
    for uuid in ids:
        labels = id2labels[uuid]
        for (lab_name, val) in labels.items():
            Vs[lab_name].append(val)

    # Set up the subplots
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[1, 1],
                          height_ratios=[1, 1])
    ax_neg = fig.add_subplot(gs[0, 0])
    ax_neg.set_title("Negation", fontdict={"fontsize": 18})
    # ax_neg.set_xticks([])
    ax_neg.set_yticks([])
    ax_unc = fig.add_subplot(gs[0, 1])
    ax_unc.set_title("Uncertainty", fontdict={"fontsize": 18})
    # ax_unc.set_xticks([])
    ax_unc.set_yticks([])
    ax_con_neg = fig.add_subplot(gs[1, 0])
    ax_con_neg.set_aspect(1)
    ax_con_neg.set_title("Content - Negation", fontdict={"fontsize": 18})
    ax_con_neg.set_xticks([])
    ax_con_neg.set_yticks([])
    ax_con_unc = fig.add_subplot(gs[1, 1])
    ax_con_unc.set_aspect(1)
    ax_con_unc.set_title("Content - Uncertainty", fontdict={"fontsize": 18})
    ax_con_unc.set_xticks([])
    ax_con_unc.set_yticks([])

    for (latent_name, zfile) in zip(latent_names, z_files):
        zs = np.loadtxt(zfile, delimiter=',')

        if latent_name == "Polarity":
            plot_negation(zs, Vs[latent_name], ax_neg)
        elif latent_name == "Certainty":
            plot_uncertainty(zs, Vs[latent_name], ax_unc)
        elif latent_name == "content":
            plot_content(zs, Vs, ax_con_neg, variable="Polarity")
            plot_content(zs, Vs, ax_con_unc, variable="Certainty")
        else:
            continue
    plt.show()


def plot_negation(zs, labels, axis):
    colors = {"Positive": "#ef8a62", "Negative": "#67a9cf"}
    for lab_val in set(labels):
        mask = np.array(labels) == lab_val
        sns.histplot(zs[mask], color=colors[lab_val], alpha=0.8,
                     ax=axis, label=lab_val, linewidth=0)
    axis.legend(fontsize=14)


def plot_uncertainty(zs, labels, axis):
    colors = {"Certain": "#af8dc3", "Uncertain": "#7fbf7b"}
    ci = 0
    for lab_val in set(labels):
        mask = np.array(labels) == lab_val
        sns.histplot(zs[mask], color=colors[lab_val], alpha=0.8,
                     ax=axis, label=lab_val, linewidth=0)
        ci += 1
    axis.legend(fontsize=14)


def plot_content_old(zs, labels_dict, axis):
    z_emb = TSNE(n_components=2).fit_transform(zs)

    df = pd.DataFrame({"z0": z_emb[:, 0], "z1": z_emb[:, 1],
                       "negation": labels_dict["polarity"],
                       "uncertainty": labels_dict["uncertainty"]})
    colors = ["#ef8a62", "#67a9cf"]
    sns.scatterplot(data=df, x="z0", y="z1", hue="negation", alpha=0.8,
                    style="uncertainty", palette=colors, ax=axis)


def plot_content(zs, labels_dict, axis, variable="Polarity"):
    z_emb = TSNE(n_components=2).fit_transform(zs)

    key = variable
    colors = {"Certain": "#af8dc3", "Uncertain": "#7fbf7b"}
    if variable == "Polarity":
        colors = {"Positive": "#ef8a62", "Negative": "#67a9cf"}
    df = pd.DataFrame({"z0": z_emb[:, 0], "z1": z_emb[:, 1],
                       variable: labels_dict[key]})
    sns.scatterplot(data=df, x="z0", y="z1", hue=variable,
                    hue_order=labels_dict[key][::-1], palette=colors, ax=axis)
    axis.get_legend().remove()


def get_last_epoch(directory):
    files = os.listdir(directory)
    epochs = {int(re.findall(r'.*_([0-9]+)\.log', fname)[0])
              for fname in files}
    return max(epochs)


def get_latent_names(filenames):
    latent_names = []
    for fname in filenames:
        name = re.findall(r'.*_(\w+)_[0-9]+.log', fname)[0]
        latent_names.append(name)
    return latent_names


def get_labels(config, data_split, latent_names):
    dataset_name = config.Data.dataset_name.value
    datamodule_cls = D.DATAMODULE_REGISTRY[dataset_name]
    dm = datamodule_cls(config)
    dm.setup()
    data = getattr(dm.dataset, data_split)
    id2labels = {}
    labels_set = set()
    for datum in data:
        datum = dm.dataset.inverse_transform_labels(datum)
        labs = {key: val for (key, val) in datum["json"]["labels"].items()
                if key in latent_names}
        id2labels[datum["__key__"]] = labs
        labels_set.update(set(labs.keys()))
    return id2labels, labels_set


if __name__ == "__main__":
    args = parse_args()
    main(args)
