import os
import sys
import json
import argparse
import logging
from tqdm import tqdm
from collections import defaultdict

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

sys.path.insert(0, os.getcwd())
from vae import utils, data, model, losses  # noqa
from config import config as params  # noqa


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str,
                        help="""YAML Config file for model to
                                use for reconstruction.""")
    parser.add_argument("-N", type=int, default=-1,
                        help="Number of examples to use.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Use tqdm progress bars.")
    return parser.parse_args()


def main(args):
    logging.basicConfig(level=logging.INFO)
    params.load_yaml(args.config_file)

    logging.info("Running reconstruction...")
    sents, recons = reconstruct_with_model(params, N=args.N,
                                           verbose=args.verbose)

    logging.info("Loading GPT2...")
    model_id = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id).to(DEVICE)

    logging.info("Computing PPLs...")
    for (datasplit, split_sents) in sents.items():
        ppl = compute_ppl(split_sents, tokenizer, model, verbose=args.verbose)
        recon_ppl = compute_ppl(recons[datasplit], tokenizer, model,
                                verbose=args.verbose)
        logging.info(f"({datasplit}) orig: {ppl:.4f} | recon: {recon_ppl:4f}")

    outfile = os.path.join(
        params.Experiment.logdir.value,
        params.Experiment.name.value,
        "evaluation/perplexity.jsonl")
    with open(outfile, 'w') as outF:
        for (datasplit, sents) in sents.items():
            for (sent, recon) in zip(sents, recons[datasplit]):
                json.dump({"datasplit": datasplit, "sentence": sent,
                           "reconstruction": recon}, outF)
                outF.write('\n')
    logging.info(f"Reconstructions saved to {outfile}")


def compute_ppl(sentences, tokenizer, model, stride=512, verbose=False):
    encodings = tokenizer.encode("\n\n".join(sentences), return_tensors='pt')
    max_length = model.config.n_positions

    nlls = []
    if verbose is True:
        pbar = tqdm()
    for i in range(0, encodings.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.size(1))
        trg_len = end_loc - i
        input_ids = encodings[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nll = outputs[0] * trg_len
        nlls.append(nll)
        if verbose is True:
            pbar.update(1)
        else:
            if i % 10 == 0:
                logging.info(f"{i}")
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl


def reconstruct_with_model(params, N=-1, num_resamples=1, verbose=False):
    utils.set_seed(params.Experiment.random_seed.value)

    # Set logging directory
    # Set model checkpoint directory
    ckpt_dir = os.path.join(params.Experiment.logdir.value,
                            params.Experiment.name.value,
                            "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise OSError(f"No checkpoint found at '{ckpt_dir}'!")

    # Read train data
    # Load the train data so we can fully specify the model
    datamodule_cls = data.DATAMODULE_REGISTRY[params.Data.dataset_name.value]
    dm = datamodule_cls(params)
    dm.setup()
    dataloaders = [("train", dm.train_dataloader()),
                   ("val", dm.val_dataloader()),
                   ("test", dm.test_dataloader())]

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

    orig_sentences = defaultdict(list)
    reconstructed_sentences = defaultdict(list)
    for (datasplit, dataloader) in dataloaders:
        if args.verbose is True:
            try:
                pbar = tqdm(total=len(dataloader))
            except TypeError:  # WebLoader has no __len__
                pbar = tqdm(dataloader)
        for (i, batch) in enumerate(dataloader):
            in_Xbatch = batch["json"]["encoded"]["input_ids"].to(vae.device)
            Ybatch = {}
            for (task, vals) in batch["json"]["labels"].items():
                Ybatch[task] = vals.to(vae.device)
            lengths = batch["json"]["encoded"]["lengths"].to(vae.device)

            # trg_output = {"decoder_logits": [batch_size, target_length, vocab_size]  # noqa
            #           "latent_params": {latent_name: [Params(z, mu, logvar)] * batch_size}  # noqa
            #           "dsc_logits": {latent_name: [batch_size, n_classes]}
            #           "adv_logits": {latent_name-label_name: [batch_size, n_classes]}  # noqa
            #           "token_predictions": [batch_size, target_length]
            trg_output = vae(in_Xbatch, lengths, teacher_forcing_prob=0.0)

            origs = dm.tokenizer.batch_decode(
                in_Xbatch, skip_special_tokens=True)
            orig_sentences[datasplit].extend(origs)
            # Get the decoded reconstructions ...
            Xbatch_hat = trg_output["token_predictions"]
            recons = dm.tokenizer.batch_decode(
                Xbatch_hat, skip_special_tokens=True)
            reconstructed_sentences[datasplit].extend(recons)
        if args.verbose is True:
            pbar.update(1)
        else:
            if i % 10 == 0:
                logging.info(f"({datasplit}) {i}")
    return orig_sentences, reconstructed_sentences


if __name__ == "__main__":
    args = parse_args()
    main(args)
