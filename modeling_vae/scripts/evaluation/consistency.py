import os
import csv
import sys
import logging
import argparse
import datetime
from collections import defaultdict

# External packages
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

# Local imports
sys.path.insert(0, os.getcwd())
from vae import utils, data, model, losses  # noqa
from config import config as params  # noqa


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help="Specify compute, or summarize")

    compute_parser = subparsers.add_parser("compute")
    compute_parser.set_defaults(compute=True, summarize=False)
    compute_parser.add_argument("config_file", type=str,
                                help="""Path to YAML file containing
                                        experiment parameters.""")
    compute_parser.add_argument("datasplit", type=str,
                                choices=["train", "val", "test"],
                                help="Dataset to summarize.")
    compute_parser.add_argument("--num_resamples", type=int, default=30,
                                required=False,
                                help="""Number of times to resample Z and
                                        decode for a given input example.""")
    compute_parser.add_argument("--verbose", action="store_true", default=False,  # noqa
                                help="""Show a progress bar.""")

    summ_parser = subparsers.add_parser("summarize")
    summ_parser.set_defaults(compute=False, summarize=True)
    summ_parser.add_argument("config_file", type=str,
                             help="""Path to YAML file containing
                                     experiment parameters.""")
    summ_parser.add_argument("datasplit", type=str,
                             choices=["train", "dev", "test"],
                             help="Dataset to summarize.")

    return parser.parse_args()


def compute(args):
    params.load_yaml(args.config_file)
    utils.set_seed(params.Experiment.random_seed.value)

    logging.basicConfig(level=logging.INFO)
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    logging.info(f"START: {now_str}")

    ckpt_dir = os.path.join(params.Experiment.checkpoint_dir.value,
                            params.Experiment.name.value)
    if not os.path.isdir(ckpt_dir):
        raise OSError(f"No model found at {ckpt_dir}")

    # Load the train data so we can fully specify the model
    datamodule_cls = data.DATAMODULE_REGISTRY[params.Data.dataset_name.value]
    dm = datamodule_cls(params)
    dm.setup()
    if args.datasplit == "train":
        dataloader = dm.train_dataloader()
    elif args.datasplit == "val":
        dataloader = dm.val_dataloader()
    elif args.datasplit == "test":
        dataloader = dm.test_dataloader()

    # Build the VAE
    label_dims_dict = dm.label_spec
    sos_idx = dm.tokenizer.cls_token_id
    eos_idx = dm.tokenizer.sep_token_id
    vae = model.build_vae(params, dm.tokenizer.vocab_size,
                          None, label_dims_dict,
                          DEVICE, sos_idx, eos_idx)
    optimizer = torch.optim.Adam(vae.trainable_parameters(),
                                 lr=params.Training.learn_rate.value)

    # Load the latest checkpoint, if there is one.
    if not os.path.isdir(ckpt_dir):
        raise OSError(f"No checkpoint found at '{ckpt_dir}'!")
    vae, optimizer, start_epoch, ckpt_fname = utils.load_latest_checkpoint(
            vae, optimizer, ckpt_dir)
    logging.info(f"Loaded checkpoint from '{ckpt_fname}'")

    logging.info("Successfully loaded model")
    logging.info(vae)
    vae.train()  # So we sample different latents each time.

    true_labels = defaultdict(list)
    list_fn = lambda: [[] for _ in range(args.num_resamples)]  # noqa
    # predictions given the input
    latent_predictions = defaultdict(list_fn)
    # predictions given the re-encoded input
    latent_predictions_hat = defaultdict(list_fn)
    bleus = list_fn()
    if args.verbose is True:
        try:
            pbar = tqdm(total=len(dataloader))
        except TypeError:  # WebLoader has no __len__
            pbar = tqdm(dataloader)
    for (i, batch) in enumerate(dataloader):
        in_Xbatch = batch["json"]["encoded"]["input_ids"].to(vae.device)
        target_Xbatch = batch["json"]["encoded"]["input_ids"].to(vae.device)
        Ybatch = {}
        for (task, val) in batch["json"]["labels"].items():
            Ybatch[task] = val.to(vae.device)
            true_labels[task].extend(val.tolist())
        lengths = batch["json"]["encoded"]["lengths"].to(vae.device)

        for resample in range(args.num_resamples):
            # output = {"decoder_logits": [batch_size, target_length, vocab_size]  # noqa
            #           "latent_params": [Params(z, mu, logvar)] * batch_size
            #           "dsc_logits": {latent_name: [batch_size, n_classes]}
            #           "adv_logits": {latent_name-label_name: [batch_size, n_classes]}  # noqa
            #           "token_predictions": [batch_size, target_length]
            output = vae(in_Xbatch, lengths, teacher_forcing_prob=0.0)

            # Get the discriminators' predictions for each latent space.
            for (label_name, logits) in output["dsc_logits"].items():
                preds = vae.discriminators[label_name].predict(logits)
                latent_predictions[label_name][resample].extend(
                    preds.cpu().tolist())

            # Get the decoded reconstructions ...
            Xbatch_hat = output["token_predictions"]
            condition = (Xbatch_hat == vae.eos_token_idx) | (Xbatch_hat == 0)
            num_pad = torch.where(condition,                    # if
                                  torch.tensor(1),              # then
                                  torch.tensor(0)).sum(axis=1)  # else
            lengths_hat = Xbatch_hat.size(1) - num_pad
            Xbatch_hat = Xbatch_hat.to(vae.device)
            lengths_hat = lengths_hat.to(vae.device)
            # ... and encode them again ...
            output_hat = vae(Xbatch_hat, lengths_hat, teacher_forcing_prob=0.0)

            # Measure self-BLEU
            bleu = losses.compute_bleu(
                target_Xbatch, Xbatch_hat, dm.tokenizer)
            bleus[resample].append(bleu)

            # ... and get the discriminators' predictions for the new input.
            for (label_name, logits) in output_hat["dsc_logits"].items():
                preds = vae.discriminators[label_name].predict(logits)
                latent_predictions_hat[label_name][resample].extend(
                    preds.cpu().tolist())

        if args.verbose is True:
            pbar.update(1)
        else:
            logging.info(f"{i}")

    results = []
    for label_name in latent_predictions.keys():
        trues = np.array(true_labels[label_name])
        preds = np.array(latent_predictions[label_name])
        preds_hat = np.array(latent_predictions_hat[label_name])
        for resample in range(preds.shape[0]):
            p, r, f, _ = precision_recall_fscore_support(
                trues, preds[resample, :], average="macro")
            row = [resample, label_name, "y", "y_hat", p, r, f]
            results.append(row)

            p, r, f, _ = precision_recall_fscore_support(
                trues, preds_hat[resample, :], average="macro")
            row = [resample, label_name, "y", "y_hat_prime", p, r, f]
            results.append(row)

            p, r, f, _ = precision_recall_fscore_support(
                preds[resample, :], preds_hat[resample, :], average="macro")
            row = [resample, label_name, "y_hat", "y_hat_prime", p, r, f]
            results.append(row)

    outdir = os.path.join(params.Experiment.checkpoint_dir.value,
                          params.Experiment.name.value,
                          "evaluation/consistency")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(
        outdir, f"decoder_predictions_{args.datasplit}.csv")
    with open(outfile, 'w') as outF:
        writer = csv.writer(outF, delimiter=',')
        writer.writerow(["batch", "sample_num", "label", "true", "pred",
                         "precision", "recall", "F1"])
        for (batch, row) in enumerate(results):
            writer.writerow([batch] + row)

    bleu_outfile = os.path.join(outdir, f"self_bleus_{args.datasplit}.csv")
    with open(bleu_outfile, 'w') as outF:
        writer = csv.writer(outF, delimiter=',')
        writer.writerow(["batch", "sample_num", "BLEU"])
        for (resample, sample_bleus) in enumerate(bleus):
            for (batch, b) in enumerate(sample_bleus):
                writer.writerow([batch, resample, b])


def summarize(args):
    params.load_yaml(args.config_file)
    outdir = os.path.join(params.Experiment.checkpoint_dir.value,
                          params.Experiment.name.value,
                          "evaluation/consistency")
    infile = os.path.join(
        outdir, f"decoder_predictions_{args.datasplit}.csv")
    df = pd.read_csv(infile)
    summ_df = df.groupby(["label", "true", "pred"]).agg(
        ["mean", "std"]).drop(["sample_num", "batch"], axis="columns")
    print(summ_df.to_string())

    fig, ax = plt.subplots(figsize=(10, 4))
    means = df.groupby(["label", "true", "pred"]).mean().drop(
        ["sample_num", "batch"], axis="columns")
    errs = df.groupby(["label", "true", "pred"]).std().drop(
        ["sample_num", "batch"], axis="columns")
    plots = means.plot.barh(yerr=errs, rot=0, subplots=True,
                            ax=ax, sharey=True, layout=(1, 3), legend=False)

    colors = list(matplotlib.colors.TABLEAU_COLORS.values())[:3]
    for plot in plots[0]:
        for (i, bar) in enumerate(plot.patches):
            c = colors[i % 3]
            bar.set_color(c)
            if i < 3:
                bar.set_hatch('/')
                bar.set_edgecolor('k')

    plt.tight_layout()
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    plot_outfile = os.path.join(
        outdir, "plots", f"decoder_predictions_{args.datasplit}.pdf")
    fig.savefig(plot_outfile, dpi=300)
    plot_outfile = os.path.join(
        outdir, "plots", f"decoder_predictions_{args.datasplit}.png")
    fig.savefig(plot_outfile, dpi=300)


if __name__ == "__main__":
    args = parse_args()
    if args.compute is True:
        compute(args)
    elif args.summarize is True:
        summarize(args)
