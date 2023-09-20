# Built-in packages
import os
import sys
import csv
import time
import logging
import argparse
import datetime
from pprint import pformat
from collections import defaultdict

# External packages
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Local packages
from vae import utils, data, model, losses
from config import config as params


torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="Run model training")
    train_parser.add_argument(
        "params_file", type=str, help="Path to yaml config file.")
    train_parser.add_argument("--quiet", action="store_true", default=False)

    val_parser = subparsers.add_parser("validate", help="Run model validation")
    val_parser.add_argument(
        "params_file", type=str, help="Path to yaml config file.")
    val_parser.add_argument("--quiet", action="store_true", default=False)

    test_parser = subparsers.add_parser(
        "test", help="Run model validation on test set")
    test_parser.add_argument(
        "params_file", type=str, help="Path to yaml config file.")
    test_parser.add_argument("--quiet", action="store_true", default=False)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    return args


class LossLogger(object):

    def __init__(self, summary_writer, epoch):
        self.losses = {}
        self.summary_writer = summary_writer
        self.epoch = epoch

    def __repr__(self):
        return str(self.losses)

    def __str__(self):
        return pformat(self.losses)

    def __getitem__(self, key):
        return self.losses[key]

    def update(self, d, subdict=None):
        """
        Update self.losses with dict d
        """
        to_update = self.losses if subdict is None else subdict
        for (key, val) in d.items():
            if isinstance(val, dict):
                if key not in to_update.keys():
                    to_update[key] = {}
                self.update(val, subdict=to_update[key])
            else:
                if key not in to_update.keys():
                    to_update[key] = []
                val = self._to_scalar(val)
                to_update[key].append(val)

    def _log(self, i, subdict=None, base_keystr='',
             collapse_fn=None, collapse_fn_args=[]):
        if collapse_fn is None:
            raise NotImplementedError("Need to specify a collapse_fn")
        to_log = self.losses if subdict is None else subdict
        for (key, val) in to_log.items():
            keystr = f"{base_keystr}_{key}"
            if isinstance(val, dict):
                self._log(i, subdict=to_log[key], base_keystr=keystr,
                          collapse_fn=collapse_fn,
                          collapse_fn_args=collapse_fn_args)
            elif isinstance(val, list):
                val = self._to_scalar(val)
                logval = collapse_fn(val, *collapse_fn_args)
                self.summary_writer.add_scalar(keystr, logval, i)
            else:
                raise ValueError("Encountered lone scalar '{keystr}: {val}' in LossLogger.log")  # noqa

    def log_epoch(self, subdict=None, base_keystr="avg"):
        self._log(i=self.epoch, subdict=subdict, base_keystr=base_keystr,
                  collapse_fn=np.mean)

    def log_step(self, step, subdict=None, base_keystr="step"):
        self._log(i=step, subdict=subdict, base_keystr=base_keystr,
                  collapse_fn=list.__getitem__, collapse_fn_args=[-1])

    def summarize(self, key):
        val = self.losses[key]
        val = self._to_scalar(val)
        return np.mean(val), np.std(val)

    @classmethod
    def _to_scalar(cls, xs):
        try:
            out = []
            for x in xs:
                out.append(cls._to_scalar(x))
            return out
        except TypeError:
            if isinstance(xs, torch.Tensor):
                return xs.cpu().detach().item()
            elif isinstance(xs, np.ndarray):
                return xs.item()
            else:
                return xs


def safe_dict_update(d1, d2):
    for (key, val) in d2.items():
        if key not in d1.keys():
            d1.update({key: val})


def compute_all_losses(model, model_outputs, Xbatch, Ybatch,
                       lengths, kl_weights_dict, mi_loss_weight):
    # model_outputs = {
    #   "decoder_logits": [batch_size, target_length, vocab_size],
    #   "latent_params": [Params(z, mu, logvar)] * batch_size,
    #   "dsc_logits": {latent_name: [batch_size, n_classes]},
    #   "adv_logits": {adversary_name: [batch_size, n_classes]},
    #   "token_predictions": [batch_size, target_length]}
    L = dict()
    safe_dict_update(
        L, losses.reconstruction_loss(
            Xbatch, model_outputs["decoder_logits"], lengths)
    )

    safe_dict_update(
        L, losses.compute_kl_divergence_losses(
            model, model_outputs["latent_params"], kl_weights_dict)
    )
    safe_dict_update(
        L, losses.compute_discriminator_losses(
            model, model_outputs["dsc_logits"], Ybatch)
    )
    safe_dict_update(
        L, losses.compute_adversarial_losses(
            model, model_outputs["adv_logits"], Ybatch)
    )
    safe_dict_update(
        L, losses.compute_mi_losses(
            model, model_outputs["latent_params"], beta=mi_loss_weight)
    )
    total_loss = (L["reconstruction_loss"] +
                  L["total_weighted_kl"] +
                  L["total_dsc_loss"] +
                  L["total_adv_loss"] +
                  L["total_mi"])
    return total_loss, L


def log_params(params_dict, example_ids, logdir, dataset_name, epoch):
    """
    :param defaultdict params_dict: {latent_name: {parameter: [p1...pN]}}
    :param str logdir:
    :param str dataset_name:
    :param int epoch:
    """
    metadata_dir = os.path.join(logdir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    # Log example IDs in the same order as their parameters.
    ids_dir = os.path.join(metadata_dir, "ordered_ids")
    os.makedirs(ids_dir, exist_ok=True)
    ids_outfile = os.path.join(ids_dir, f"{dataset_name}_{epoch}.log")
    with open(ids_outfile, 'w') as outF:
        for i in example_ids:
            outF.write(f"{i}\n")

    for latent_name in params_dict.keys():
        for (param_name, values) in params_dict[latent_name].items():
            param_dir = os.path.join(metadata_dir, param_name)
            os.makedirs(param_dir, exist_ok=True)
            outfile = os.path.join(
                    param_dir, f"{dataset_name}_{latent_name}_{epoch}.log")
            with open(outfile, 'w') as outF:
                writer = csv.writer(outF, delimiter=',')
                for value in values:
                    row = [f"{dim:.4f}" for dim in value]
                    writer.writerow(row)


def trainstep(model, optimizer, dataloader, params, epoch, tokenizer,
              verbose=True, summary_writer=None, logdir=None):

    epoch_start = time.time()

    if summary_writer is None:
        summary_writer = SummaryWriter()
    if logdir is None:
        logdir = "logs"

    loss_logger = LossLogger(summary_writer, epoch)
    # Log example IDs in same order as latent parameters
    all_sent_ids = []
    all_latent_params = defaultdict(lambda: defaultdict(list))

    model.train()
    if verbose is True:
        try:
            pbar = tqdm(total=len(dataloader))
        except TypeError:  # WebLoader has no __len__
            pbar = tqdm(dataloader)
    step = 0
    for (i, batch) in enumerate(dataloader):
        in_Xbatch = batch["json"]["encoded"]["input_ids"].to(model.device)
        target_Xbatch = batch["json"]["encoded"]["input_ids"].to(model.device)
        Ybatch = {}
        for (task, val) in batch["json"]["labels"].items():
            Ybatch[task] = val.to(model.device)
        lengths = batch["json"]["encoded"]["lengths"].to(model.device)
        batch_ids = batch["__key__"]

        # output = {"decoder_logits": [batch_size, target_length, vocab_size]
        #           "latent_params": [Params(z, mu, logvar)] * batch_size
        #           "dsc_logits": {latent_name: [batch_size, n_classes]}
        #           "adv_logits": {latent_name-label_name: [batch_size, n_classes]}  # noqa
        #           "token_predictions": [batch_size, target_length]
        output = model(
            in_Xbatch, lengths,
            teacher_forcing_prob=params.Training.teacher_forcing_prob.value)

        kl_weights_dict = {}
        lambdas = params.Training.lambdas.value
        for (latent_name, weight) in lambdas.items():
            kl_weights_dict[latent_name] = weight
            loss_logger.update({"kl_weights": kl_weights_dict})

        # DO NOT CHANGE MI LOSS WEIGHT! IT WORKS NOW BUT WONT IF YOU CHANGE IT!
        mi_loss_weight = 0.01
        loss_logger.update({"mi_loss_weight": mi_loss_weight})

        # COMPUTE MANY MANY LOSSES
        total_loss, losses_dict = compute_all_losses(
            model, output, target_Xbatch, Ybatch,
            lengths, kl_weights_dict, mi_loss_weight)
        loss_logger.update({"total_loss": total_loss})
        loss_logger.update(losses_dict)

        # Update the model
        # I don't exactly know why I need to call backward, update all
        # the adversaries, and then call step(), but it works and I've
        # checked that everything updates properly.
        # with utils.AutogradDebugger():  # uncomment for interactive debugging
        total_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5.0)
        key = "idv_adv_dsc_losses"
        for (adv_name, adv_dsc_loss) in losses_dict[key].items():
            # Update only the adversaries
            # with utils.AutogradDebugger():
            model.adversaries[adv_name].optimizer_step(adv_dsc_loss)
        optimizer.step()
        optimizer.zero_grad()

        # Update the MI estimators
        key = "idv_mi_estimates"
        for (latent_pair_name, mi_loss) in losses_dict[key].items():
            mi_estimator = model.mi_estimators[latent_pair_name]
            mi_estimator.train()
            latent_name_1, latent_name_2 = latent_pair_name.split('-')
            params1 = output["latent_params"][latent_name_1]
            params2 = output["latent_params"][latent_name_2]
            mi_loss = mi_estimator.learning_loss(
                params1.z.detach(), params2.z.detach())
            mi_estimator.optimizer_step(mi_loss)
            loss_logger.update({"mi_estimator_loss": {latent_pair_name: mi_loss}})  # noqa
            mi_estimator.eval()

        # Log latents
        all_sent_ids.extend(batch_ids)
        for (l_name, l_params) in output["latent_params"].items():
            for (param_name, param_batch) in l_params._asdict().items():
                param_batch = param_batch.detach().cpu().tolist()
                all_latent_params[l_name][param_name].extend(param_batch)

        # Measure Autoencoding by reencoding the reconstructed output.
        x_prime = output["token_predictions"].to(model.device)
        output_prime = model(
            x_prime, lengths,
            teacher_forcing_prob=params.Training.teacher_forcing_prob.value)

        for (l_name, l_params) in output_prime["latent_params"].items():
            orig_z = output["latent_params"][l_name].z
            z_prime = l_params.z
            diff = torch.norm(z_prime - orig_z, p=None, dim=1).mean()
            loss_logger.update({"idv_ae": {l_name: diff.item()}})

        # Measure self-BLEU
        bleu = losses.compute_bleu(
            target_Xbatch, x_prime, tokenizer)
        loss_logger.update({"bleu": bleu})

        loss_logger.log_step(step)
        if verbose is True:
            pbar.update(1)
            pbar.set_description(f"EPOCH: {epoch}")

        step += 1

    if verbose is True:
        pbar.close()

    epoch_time = time.time() - epoch_start
    difftime_str = str(datetime.timedelta(seconds=epoch_time))

    loss_logger.log_epoch()
    log_params(all_latent_params, all_sent_ids, logdir, "train", epoch)

    tlmu, tlsig = loss_logger.summarize("total_loss")
    rcmu, rcsig = loss_logger.summarize("reconstruction_loss")
    klmu, klsig = loss_logger.summarize("total_kl")
    dscmu, dscsig = loss_logger.summarize("total_dsc_loss")
    advmu, advsig = loss_logger.summarize("total_adv_loss")
    mimu, misig = loss_logger.summarize("total_mi")

    logstr = f"TRAIN ({epoch}) TOTAL: {tlmu:.4f} +/- {tlsig:.4f}"
    logstr += f" | RECON: {rcmu:.4f} +/- {rcsig:.4f}"
    logstr += f" | KL: {klmu:.4f} +/- {klsig:.4f}"
    logstr += f" | DISCRIM: {dscmu:.4f} +/- {dscsig:.4f}"
    if model.adversarial_loss is True:
        logstr += f" | ADVERSE: {advmu:.4f} +/- {advsig:.4f}"
    if model.mi_loss is True:
        logstr += f" | MI: {mimu:.4f} +/- {misig:.4f}"
    logstr += f" | Epoch time: {difftime_str}"
    logging.info(logstr)

    return model, optimizer


def evalstep(model, dataloader, params, epoch, tokenizer, name="val",
             verbose=True, summary_writer=None, logdir=None):

    if summary_writer is None:
        summary_writer = SummaryWriter()
    if logdir is None:
        logdir = "logs"

    loss_logger = LossLogger(summary_writer, epoch)
    # Log example IDs and latent params
    all_sent_ids = []
    all_latent_params = defaultdict(lambda: defaultdict(list))

    model.eval()
    if verbose is True:
        try:
            pbar = tqdm(total=len(dataloader))
        except TypeError:  # WebLoader has no __len__
            pbar = tqdm(dataloader)
    for (i, batch) in enumerate(dataloader):
        in_Xbatch = batch["json"]["encoded"]["input_ids"].to(model.device)
        target_Xbatch = batch["json"]["encoded"]["input_ids"].to(model.device)
        Ybatch = {}
        for (task, val) in batch["json"]["labels"].items():
            Ybatch[task] = val.to(model.device)
        lengths = batch["json"]["encoded"]["lengths"].to(model.device)
        batch_ids = batch["__key__"]

        output = model(in_Xbatch, lengths, teacher_forcing_prob=0.0)

        kl_weights_dict = {}
        lambdas = params.Training.lambdas.value
        for (latent_name, weight) in lambdas.items():
            weight_val = weight
            # During evaluation we don't want cyclic annealing.
            if weight_val == "cyclic":
                weight_val = 1.0  # Don't weight it on eval.
            kl_weights_dict[latent_name] = weight_val

        mi_loss_weight = 1.0
        total_loss, losses_dict = compute_all_losses(
            model, output, target_Xbatch, Ybatch, lengths,
            kl_weights_dict, mi_loss_weight)
        loss_logger.update({"total_loss": total_loss})
        loss_logger.update(losses_dict)

        # Measure self-BLEU
        x_prime = output["token_predictions"].to(model.device)
        bleu = losses.compute_bleu(target_Xbatch, x_prime, tokenizer)
        loss_logger.update({"bleu": bleu})

        # Log latents
        all_sent_ids.extend(batch_ids)
        for (l_name, l_params) in output["latent_params"].items():
            for (param_name, param_batch) in l_params._asdict().items():
                param_batch = param_batch.detach().cpu().tolist()
                all_latent_params[l_name][param_name].extend(param_batch)

        if verbose is True:
            pbar.update(1)
            pbar.set_description(f" ↳ EVAL ({name})")

    if verbose is True:
        pbar.close()

    loss_logger.log_epoch()
    log_params(all_latent_params, all_sent_ids, logdir, name, epoch)

    tlmu, tlsig = loss_logger.summarize("total_loss")
    rcmu, rcsig = loss_logger.summarize("reconstruction_loss")
    klmu, klsig = loss_logger.summarize("total_kl")
    dscmu, dscsig = loss_logger.summarize("total_dsc_loss")
    advmu, advsig = loss_logger.summarize("total_adv_loss")
    mimu, misig = loss_logger.summarize("total_mi")

    logstr = f"{name.upper()} ({epoch}) TOTAL: {tlmu:.4f} +/- {tlsig:.4f}"
    logstr += f" | RECON: {rcmu:.4f} +/- {rcsig:.4f}"
    logstr += f" | DISCRIM: {dscmu:.4f} +/- {dscsig:.4f}"
    logstr += f" | KL: {klmu:.4f} +/- {klsig:.4f}"
    if model.adversarial_loss is True:
        logstr += f" | ADVERSE: {advmu:.4f} +/- {advsig:.4f}"
    if model.mi_loss is True:
        logstr += f" | MI: {mimu:.4f} +/- {misig:.4f}"
    logging.info(logstr)


def run(args):
    params.load_yaml(args.params_file)
    utils.set_seed(params.Experiment.random_seed.value)

    # Set logging directory
    logdir = os.path.join("logs", params.Experiment.name.value)
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, "run.log")
    print(f"Logging to {logfile}")
    logging.basicConfig(filename=logfile, level=logging.INFO)

    # Log parameters
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    logging.info(f"START: {now_str}")
    logging.info(str(params))

    # Set model checkpoint directory
    ckpt_dir = os.path.join(params.Experiment.checkpoint_dir.value,
                            params.Experiment.name.value)
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Read train data
    # Always load the train data since we need it to build the model
    datamodule_cls = data.DATAMODULE_REGISTRY[params.Data.dataset_name.value]
    dm = datamodule_cls(params)
    dm.setup()
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    test_dataloader = dm.test_dataloader()

    if args.command in ["train", "validate"]:
        train_writer_path = os.path.join(
            logdir, "summary", "train")
        train_writer = SummaryWriter(log_dir=train_writer_path)
        val_writer_path = os.path.join(
            logdir, "summary", "val")
        val_writer = SummaryWriter(log_dir=val_writer_path)

    if args.command == "test":
        test_writer_path = os.path.join(
            logdir, "summary", "test")
        test_writer = SummaryWriter(log_dir=test_writer_path)

    # Build the VAE
    label_dims_dict = dm.label_spec
    sos_idx = dm.tokenizer.cls_token_id
    eos_idx = dm.tokenizer.sep_token_id
    vae = model.build_vae(params, dm.tokenizer.vocab_size,
                          None, label_dims_dict,
                          DEVICE, sos_idx, eos_idx)
    logging.info(vae)

    optimizer = torch.optim.Adam(vae.trainable_parameters(),
                                 lr=params.Training.learn_rate.value)

    # If there is a checkpoint at checkpoint_dir, we load it and continue
    # training/evaluating from there.
    # If no checkpoints exist at checkpoint_dir, load_latest_checkpoint
    # will return the same model and opt, and start_epoch=0
    checkpoint_found = False
    logging.info("Trying to load latest model checkpoint from")
    logging.info(f"  {ckpt_dir}")
    vae, optimizer, start_epoch, ckpt_fname = utils.load_latest_checkpoint(
            vae, optimizer, ckpt_dir)
    if ckpt_fname is None:
        logging.warning("No checkpoint found!")
    else:
        checkpoint_found = True
        logging.info(f"Loaded checkpoint '{ckpt_fname}'")

    # Log the experiment parameter file to recreate this run.
    config_logfile = os.path.join(logdir, "config.yaml")
    params.yaml(outpath=config_logfile)

    # TRAIN
    eval_every = 1
    train_data_to_log = [ex for (i, ex) in enumerate(dm.dataset.train) if i < 20]  # noqa
    val_data_to_log = [ex for (i, ex) in enumerate(dm.dataset.val) if i < 20]
    test_data_to_log = [ex for (i, ex) in enumerate(dm.dataset.test) if i < 20]
    if args.command == "train":
        logging.info("TRAINING")
        logging.info("Ctrl-C to interrupt and keep most recent model.")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Training from base model.")

        epoch_range = range(
            start_epoch, start_epoch + params.Training.epochs.value)
        for epoch in epoch_range:
            try:
                vae, optimizer = trainstep(
                    vae, optimizer, train_dataloader, params, epoch,
                    dm.tokenizer, verbose=not args.quiet,
                    summary_writer=train_writer, logdir=logdir)
                # Log train inputs and their reconstructions
                utils.log_reconstructions(vae, train_data_to_log,
                                          dm.tokenizer, "train", epoch, logdir)
                if epoch % eval_every == 0:
                    evalstep(vae, val_dataloader, params, epoch, dm.tokenizer,
                             verbose=not args.quiet, summary_writer=val_writer,
                             logdir=logdir)
                    # Log val inputs and their reconstructions
                    val_data_to_log = dm.dataset.val[:20]
                    utils.log_reconstructions(
                        vae, val_data_to_log, dm.tokenizer, "val",
                        epoch, logdir)
                # Save the model
                logging.info(f"Saving model checkpoint to {ckpt_dir}")
                ckpt_fname = f"model_{epoch}.pt"
                ckpt_path = os.path.join(ckpt_dir, ckpt_fname)
                logging.info(f"Saving trained model to {ckpt_path}")
                torch.save({"model_state_dict": vae.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch},
                           ckpt_path)
                checkpoint_found = True
                start_epoch = epoch

            except KeyboardInterrupt:
                logging.warning(f"Training interrupted at epoch {epoch}!")
                break

    # VALIDATE
    if args.command == "validate":
        start_epoch = start_epoch - 1
        evalstep(vae, val_dataloader, params, start_epoch, dm.tokenizer,
                 verbose=not args.quiet, summary_writer=val_writer, logdir=logdir)  # noqa
        utils.log_reconstructions(vae, val_data_to_log, dm.tokenizer,
                                  "val", start_epoch, logdir)

    # TEST
    if args.command == "test":
        start_epoch = start_epoch - 1
        evalstep(vae, test_dataloader, params, start_epoch,
                 dm.tokenizer, verbose=not args.quiet,
                 summary_writer=test_writer, logdir=logdir, name="test")
        utils.log_reconstructions(vae, test_data_to_log, dm.tokenizer,
                                  "test", start_epoch, logdir)

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    logging.info(f"END: {now_str}")


if __name__ == "__main__":
    args = parse_args()
    run(args)
