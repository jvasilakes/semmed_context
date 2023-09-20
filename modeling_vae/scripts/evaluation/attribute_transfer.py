import os
import sys
import json
import logging
import argparse
from collections import Counter, defaultdict

import torch
import torch.distributions as D
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.getcwd())
from vae import utils, data, model, losses  # noqa
from config import config as params  # noqa


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    compute_parser = subparsers.add_parser("compute")
    compute_parser.set_defaults(cmd="compute")
    compute_parser.add_argument(
        "config_file", type=str,
        help="YAML config file for experiment to evaluate.")
    compute_parser.add_argument("datasplit", type=str,
                                choices=["train", "val", "test"],
                                help="Which data split to run on.")
    compute_parser.add_argument("--verbose", action="store_true",
                                default=False)

    summ_parser = subparsers.add_parser("summarize")
    summ_parser.set_defaults(cmd="summarize")
    summ_parser.add_argument("outfile", type=str,
                             help="""outfile from compute command.""")

    return parser.parse_args()


def compute(args):
    logging.basicConfig(level=logging.INFO)
    params.load_yaml(args.config_file)
    utils.set_seed(params.Experiment.random_seed.value)

    # Set logging directory
    # Set model checkpoint directory
    ckpt_dir = os.path.join(params.Experiment.checkpoint_dir.value,
                            params.Experiment.name.value)
    if not os.path.isdir(ckpt_dir):
        raise OSError(f"No checkpoint found at '{ckpt_dir}'!")

    # Read train data
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

    label_dims_dict = dm.label_spec
    sos_idx = dm.tokenizer.cls_token_id
    eos_idx = dm.tokenizer.sep_token_id
    vae = model.build_vae(params, dm.tokenizer.vocab_size,
                          None, label_dims_dict,
                          DEVICE, sos_idx, eos_idx)
    optimizer = torch.optim.Adam(vae.trainable_parameters(),
                                 lr=params.Training.learn_rate.value)

    vae, _, start_epoch, ckpt_fname = utils.load_latest_checkpoint(
        vae, optimizer, ckpt_dir, map_location=DEVICE)
    print(f"Loaded checkpoint from '{ckpt_fname}'")
    print(vae)

    labs_df = None
    results = run_transfer(vae, dataloader, params, labs_df, dm.tokenizer,
                           args.verbose)

    outfile = os.path.join(
        params.Experiment.checkpoint_dir.value,
        params.Experiment.name.value,
        f"evaluation/attribute_transfer_{args.datasplit}.json")
    with open(outfile, 'w') as outF:
        for row in results:
            json.dump(row, outF)
            outF.write('\n')


def run_transfer(model, dataloader, params, id2labs_df, tokenizer,
                 verbose=False):
    model.eval()
    results = []
    if verbose is True:
        try:
            pbar = tqdm(total=len(dataloader))
        except TypeError:  # WebLoader has no __len__
            pbar = tqdm(dataloader)

    # First, get the output for each example in the dataset,
    # recording the latent values for each task-label.
    logged_outputs = []
    latents_by_task = defaultdict(lambda: defaultdict(list))
    for (i, batch) in enumerate(dataloader):
        in_Xbatch = batch["json"]["encoded"]["input_ids"].to(model.device)
        Ybatch = {}
        for (task, vals) in batch["json"]["labels"].items():
            Ybatch[task] = vals.to(model.device)
        lengths = batch["json"]["encoded"]["lengths"].to(model.device)

        # trg_output = {"decoder_logits": [batch_size, target_length, vocab_size]  # noqa
        #           "latent_params": {latent_name: [Params(z, mu, logvar)] * batch_size}  # noqa
        #           "dsc_logits": {latent_name: [batch_size, n_classes]}
        #           "adv_logits": {latent_name-label_name: [batch_size, n_classes]}  # noqa
        #           "token_predictions": [batch_size, target_length]
        trg_output = model(in_Xbatch, lengths, teacher_forcing_prob=0.0)
        trg_output["latent_params"] = {
            task: params._asdict() for (task, params)
            in trg_output["latent_params"].items()}
        examples = uncollate(trg_output)
        for (j, ex) in enumerate(examples):
            ex["input_ids"] = in_Xbatch[j]
            ex["labels"] = {task: vals[j] for (task, vals) in Ybatch.items()}
            ex["length"] = lengths[j]
            ex = utils.send_to_device(ex, "cpu")
            logged_outputs.append(ex)

        for (task, vals) in Ybatch.items():
            for (j, val) in enumerate(vals):
                params = (trg_output["latent_params"][task]["mu"][j].detach(),
                          trg_output["latent_params"][task]["logvar"][j].detach())  # noqa
                latents_by_task[task][val.detach().cpu().item()].append(params)

    # Get the average parameter value for each task-value combo.
    for task in latents_by_task.keys():
        for (val, latents) in latents_by_task[task].items():
            mean_mu = torch.stack([lat[0] for lat in latents], dim=0).mean(0)
            mean_logvar = torch.stack(
                [lat[1] for lat in latents], dim=0).mean(0)
            latents_by_task[task][val] = (mean_mu, mean_logvar)

    # Then, go back over the examples, flipping the latent values and decoding.
    for example in logged_outputs:
        for task in model.discriminators.keys():
            orig_lab = example["labels"][task].detach().cpu().item()
            other_labs = set(latents_by_task[task]).difference(set([orig_lab]))
            for inverse_lab in other_labs:
                inverse_mu, inverse_logvar = latents_by_task[task][inverse_lab]
                inverse_var = inverse_logvar.exp()
                inverse_z = D.Normal(inverse_mu, inverse_var).sample().cpu()
                trg_params = {latent_name: param["z"].clone()
                              for (latent_name, param)
                              in example["latent_params"].items()}
                trg_params[task] = inverse_z
                zs = [trg_params[task] for task in sorted(trg_params.keys())]
                z = torch.cat(zs).unsqueeze(0)
                max_length = example["length"] + 5
                transferred_output = model.sample(z.to(model.device), max_length=max_length)

                inverse_logits = model.discriminators[task](inverse_z.unsqueeze(0).to(model.device))
                inverse_pred = model.discriminators[task].predict(inverse_logits)

                source_text = tokenizer.decode(example["input_ids"],
                                               skip_special_tokens=True)
                transferred_text = tokenizer.decode(
                    transferred_output["token_predictions"].squeeze(),
                    skip_special_tokens=True)

                row = {"latent": task,
                       "source_text": source_text,
                       "transferred_text": transferred_text,
                       "source_label": orig_lab,
                       "inverse_label": inverse_lab,
                       "inverse_label_predicted": inverse_pred.detach().cpu().item()
                       }
                results.append(row)
        if verbose is True:
            pbar.update(1)
        else:
            print(f"{i}")
    if verbose is True:
        pbar.close()

    return results


def get_source_examples(labs_batch, dataset, latent_name, id2labs_df):
    labs = labs_batch[latent_name].flatten().numpy().astype(int)
    labs_decoded = dataset.label_encoders[latent_name].inverse_transform(labs)
    encoded_value_counts = Counter(labs_decoded)

    idx2example = {}
    for (value, count) in encoded_value_counts.items():
        encoded_value = dataset.label_encoders[latent_name].transform([value])[0]  # noqa
        # Get the idxs of the examples in the batch that we need to
        #   find source examples for.
        idxs = np.argwhere(labs == encoded_value).flatten()
        # Get the IDs of the corresponding number of source examples
        #  with a different value than the target.
        samples = id2labs_df[id2labs_df[latent_name] != value].sample(count)
        # Get the processed examples corresponding to those IDs.
        examples = [dataset.get_by_id(uuid) for uuid in samples.index]
        for (idx, ex) in zip(idxs, examples):
            idx2example[idx] = ex

    # When the above for loop is finished, we should have an example for each
    # index.
    ordered_examples = [idx2example[i] for i in range(len(idx2example))]
    # So we just have to turn it into a batch that can be fed into the model.
    batch = utils.pad_sequence_denoising(ordered_examples)
    return batch


def summarize(args):
    results = [json.loads(line) for line in open(args.outfile)]

    inv_labels = defaultdict(list)
    predictions = defaultdict(list)
    for result in results:
        latent = result["latent"]
        inv_labels[latent].append(result["inverse_label"])
        predictions[latent].append(result["inverse_label_predicted"])

    print()
    for (latent, preds) in predictions.items():
        inv_labs = np.array(inv_labels[latent])
        preds = np.array(preds)
        accs = {}
        for lab_val in set(inv_labs):
            idxs = np.where(inv_labs == lab_val)
            acc = accuracy_score(inv_labs[idxs], preds[idxs])
            accs[lab_val] = acc
        print(f"Transfering {latent}")
        print(" -------------------------------- ")
        print("|  Transferring to  |  Accuracy  |")
        print("|--------------------------------|")
        for (label, acc) in accs.items():
            print(f"|{label:^19}|{acc:^12.4f}|")
        print(" -------------------------------- ")
        print()


def uncollate(batch):
    """
    Modified from
    https://lightning-flash.readthedocs.io/en/stable/_modules/flash/core/data/batch.html#default_uncollate  # noqa

    This function is used to uncollate a batch into samples.
    The following conditions are used:

    - if the ``batch`` is a ``dict``, the result will be a list of dicts
    - if the ``batch`` is list-like, the result is guaranteed to be a list

    Args:
        batch: The batch of outputs to be uncollated.

    Returns:
        The uncollated list of predictions.

    Raises:
        ValueError: If input ``dict`` values are not all list-like.
        ValueError: If input ``dict`` values are not all the same length.
        ValueError: If the input is not a ``dict`` or list-like.
    """
    def _is_list_like_excluding_str(x):
        if isinstance(x, str):
            return False
        try:
            iter(x)
        except TypeError:
            return False
        return True

    if isinstance(batch, dict):
        if any(not _is_list_like_excluding_str(sub_batch)
               for sub_batch in batch.values()):
            raise ValueError("When uncollating a dict, all sub-batches (values) are expected to be list-like.")  # noqa
        uncollated_vals = [uncollate(val) for val in batch.values()]
        uncollated_keys_vals = [(key, uncollate(val))
                                for (key, val) in batch.items()]
        for (i, (k, v)) in enumerate(uncollated_keys_vals):
            if len(v) == 0:
                uncollated_vals.pop(i)
                batch.pop(k)
        if len(set([len(v) for v in uncollated_vals])) > 1:
            print([(k, len(v)) for (k, v) in uncollated_keys_vals])
            raise ValueError("When uncollating a dict, all sub-batches (values) are expected to have the same length.")  # noqa
        elements = list(zip(*uncollated_vals))
        return [dict(zip(batch.keys(), element)) for element in elements]
    if isinstance(batch, (list, tuple, torch.Tensor)):
        return list(batch)
    raise ValueError(
        "The batch of outputs to be uncollated is expected to be a `dict` or list-like "  # noqa
        f"(e.g. `Tensor`, `list`, `tuple`, etc.), but got input of type: {type(batch)}"  # noqa
    )


if __name__ == "__main__":
    args = parse_args()

    if args.cmd == "compute":
        compute(args)
    elif args.cmd == "summarize":
        summarize(args)
