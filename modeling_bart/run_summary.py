import os
import re
import json
import argparse
import warnings
from glob import glob
from datetime import datetime
from collections import defaultdict

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from config_summary import config
from vae.data import SemRepFactDataModule
from vae.models import MODEL_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="Run model training")
    train_parser.add_argument(
        "config_file", type=str, help="Path to yaml config file.")
    train_parser.add_argument("--quiet", action="store_true", default=False)

    val_parser = subparsers.add_parser("validate", help="Run model validation")
    val_parser.add_argument(
        "config_file", type=str, help="Path to yaml config file.")
    val_parser.add_argument("--split", type=str, default="val",
                            choices=["train", "val", "test"])
    val_parser.add_argument("--quiet", action="store_true", default=False)

    predict_parser = subparsers.add_parser("predict", help="Run prediction")
    predict_parser.add_argument(
        "config_file", type=str, help="Path to yaml config file.")
    predict_parser.add_argument("--split", type=str, default="val",
                                choices=["train", "val", "test"])
    predict_parser.add_argument("--quiet", action="store_true", default=False)

    return parser.parse_args()


def main(args):
    config.load_yaml(args.config_file)

    start_time = datetime.now()
    print(f"Experiment start: {start_time}")
    print()
    print(config)

    pl.seed_everything(config.Experiment.random_seed.value, workers=True)

    run_kwargs = {"quiet": args.quiet}
    if args.command == "train":
        run_fn = run_train
    elif args.command == "validate":
        run_fn = run_validate
        run_kwargs["datasplit"] = args.split
    elif args.command == "predict":
        run_fn = run_predict
        run_kwargs["datasplit"] = args.split
    else:
        raise argparse.ArgumentError(f"Unknown command '{args.command}'.")
    run_fn(config, **run_kwargs)

    end_time = datetime.now()
    print()
    print(f"Experiment end: {end_time}")
    print(f"  Time elapsed: {end_time - start_time}")


def run_train(config, quiet=False):
    logdir = config.Experiment.logdir.value
    version = config.Experiment.version.value
    exp_name = config.Experiment.name.value
    seed = config.Experiment.random_seed.value
    version_str = f"version_{version}/seed_{seed}"
    version_dir = os.path.join(logdir, exp_name, version_str)

    datamodule = SemRepFactDataModule(config)
    datamodule.setup()

    model_class = MODEL_REGISTRY[config.Model.model_name.value]
    epoch = 0
    ckpt_path = None
    if os.path.exists(version_dir):
        try:
            ckpt_path, hparams_path = find_model_checkpoint(config)
            epoch = int(re.match(r'.*\=([0-9]+)\.ckpt', ckpt_path).group(1))
            epoch += 1
        except OSError:
            raise OSError(f"Version dir already exists at {version_dir}, but no checkpoint found!")  # noqa
        finally:
            warnings.warn(f"Resuming training checkpoint from {ckpt_path}.")
    os.makedirs(version_dir, exist_ok=True)
    config.yaml(outpath=os.path.join(version_dir, "config.yaml"))
    model = model_class.from_config(
        config, tasks_spec=datamodule.label_spec, logdir=version_dir,
        epoch=epoch)
    model.train()

    logger = TensorBoardLogger(
        save_dir=logdir, version=version_str, name=exp_name)

    filename_fmt = f"{{epoch:02d}}"  # noqa F541 f-string is missing placeholders
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(version_dir, "checkpoints"),
        monitor="val_loss_no_kl", mode="min", filename=filename_fmt)
    lr_cb = LearningRateMonitor(logging_interval="step")

    if torch.cuda.is_available():
        available_gpus = min(1, torch.cuda.device_count())
    else:
        available_gpus = 0
        warnings.warn("No CUDA devices found. Using CPU.")
    print(f"GPUs: {available_gpus}")
    trainer = pl.Trainer(
        max_epochs=config.Training.epochs.value,
        accumulate_grad_batches=config.Training.accumulate_grad_batches.value,
        gradient_clip_val=1,
        gpus=available_gpus,
        logger=logger,
        callbacks=[checkpoint_cb, lr_cb],
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        enable_progress_bar=not quiet)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


def run_validate(config, datasplit, quiet=False):
    datamodule = SemRepFactDataModule(config)
    datamodule.setup()

    model_class = MODEL_REGISTRY[config.Model.model_name.value]
    ckpt_path, hparams_path = find_model_checkpoint(config)
    epoch = int(re.match(r'.*\=([0-9]+)\.ckpt', ckpt_path).group(1))
    #model = model_class.load_from_checkpoint(
    #    checkpoint_path=ckpt_path,
    #    hparams_file=hparams_path)
    logdir = config.Experiment.logdir.value
    version = config.Experiment.version.value
    exp_name = config.Experiment.name.value
    seed = config.Experiment.random_seed.value
    version_str = f"version_{version}/seed_{seed}"
    version_dir = os.path.join(logdir, exp_name, version_str)
    model = model_class.from_config(
        config, tasks_spec=datamodule.label_spec, logdir=version_dir,
        epoch=epoch)
    model.eval()

    if torch.cuda.is_available():
        available_gpus = min(1, torch.cuda.device_count())
    else:
        available_gpus = 0
        warnings.warn("No CUDA devices found. Using CPU.")
    print(f"GPUs: {available_gpus}")

    trainer = pl.Trainer(
        logger=False,
        gpus=available_gpus,
        enable_progress_bar=not quiet)

    if datasplit == "train":
        val_dataloader_fn = datamodule.train_dataloader
    elif datasplit == "val":
        val_dataloader_fn = datamodule.val_dataloader
    elif datasplit == "test":
        val_dataloader_fn = datamodule.test_dataloader
    else:
        raise ValueError(f"Unknown data split '{datasplit}'")
    val_dataloader = val_dataloader_fn()
    trainer.validate(model, dataloaders=val_dataloader,
                     verbose=False, ckpt_path=ckpt_path)[0]


def run_predict(config, datasplit, quiet=False):
    datamodule = SemRepFactDataModule(config)
    datamodule.setup()

    model_class = MODEL_REGISTRY[config.Model.model_name.value]
    ckpt_path, hparams_path = find_model_checkpoint(config)
    epoch = int(re.match(r'.*\=([0-9]+)\.ckpt', ckpt_path).group(1))
    #model = model_class.load_from_checkpoint(
    #    checkpoint_path=ckpt_path,
    #    hparams_file=hparams_path)
    logdir = config.Experiment.logdir.value
    version = config.Experiment.version.value
    exp_name = config.Experiment.name.value
    seed = config.Experiment.random_seed.value
    version_str = f"version_{version}/seed_{seed}"
    version_dir = os.path.join(logdir, exp_name, version_str)
    model = model_class.from_config(
        config, tasks_spec=datamodule.label_spec, logdir=version_dir,
        epoch=epoch)
    model.eval()

    if torch.cuda.is_available():
        available_gpus = min(1, torch.cuda.device_count())
    else:
        available_gpus = 0
        warnings.warn("No CUDA devices found. Using CPU.")
    print(f"GPUs: {available_gpus}")

    trainer = pl.Trainer(
        logger=False,
        gpus=available_gpus,
        enable_progress_bar=not quiet)

    if datasplit == "train":
        predict_dataloader_fn = datamodule.train_dataloader
    elif datasplit == "val":
        predict_dataloader_fn = datamodule.val_dataloader
    elif datasplit == "test":
        predict_dataloader_fn = datamodule.test_dataloader
    else:
        raise ValueError(f"Unknown data split '{datasplit}'")
    predict_dataloader = predict_dataloader_fn()
    preds = trainer.predict(
        model, dataloaders=predict_dataloader, ckpt_path=ckpt_path)

    preddir = os.path.join(
        config.Experiment.logdir.value,
        config.Experiment.name.value,
        f"version_{config.Experiment.version.value}",
        f"seed_{config.Experiment.random_seed.value}",
        "predictions")

    ignore_keys = ["token_type_ids", "attention_mask",
                   "position_ids", "levitated_idxs"]
    preds = unbatch(preds, ignore_keys=ignore_keys)
    preds_decoded = decode_and_split_by_task(preds, datamodule)
    print("Saving...")

    outdir = os.path.join(preddir, datasplit)
    os.makedirs(outdir, exist_ok=True)
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    pred_fname = f"predictions_{ckpt_name}.jsonl"
    outfile = os.path.join(outdir, pred_fname)
    with open(outfile, 'w') as outF:
        for example in preds_decoded:
            json.dump(example, outF)
            outF.write('\n')


def find_model_checkpoint(config):
    logdir = os.path.join(
        config.Experiment.logdir.value,
        config.Experiment.name.value,
        f"version_{config.Experiment.version.value}",
        f"seed_{config.Experiment.random_seed.value}")
    hparams_path = os.path.join(logdir, "hparams.yaml")
    if not os.path.isfile(hparams_path):
        raise OSError(f"No hparams.yaml found in {logdir}.")

    ckpt_glob = os.path.join(logdir, "checkpoints", "*.ckpt")
    ckpt_files = glob(ckpt_glob)
    if len(ckpt_files) == 0:
        raise OSError(f"No checkpoints found at {ckpt_glob}.")
    return ckpt_files[0], hparams_path


def unbatch(batches, ignore_keys=[]):
    unbatched = []
    for batch in batches:
        first_key = list(batch.keys())[0]
        batch_size = len(batch[first_key])
        for i in range(batch_size):
            datum = {}
            for (key, value) in batch.items():
                if key in ignore_keys:
                    continue
                if len(value) == 0:
                    continue
                if isinstance(value, dict):
                    converted = unbatch([value], ignore_keys=ignore_keys)[i]
                else:
                    converted = maybe_convert(batch[key][i])
                datum[key] = converted
            unbatched.append(datum)
    return unbatched


def maybe_convert(value):
    """
    Check if a value is a tensor, and convert
    it back to a standard Python object if it is.
    """
    if torch.is_tensor(value):
        value = value.squeeze()
        if value.dim() == 0:
            return value.item()
        else:
            return value.tolist()
    return value


def decode_and_split_by_task(unbatched, datamodule):
    for example in unbatched:
        encodings = example["json"]["encoded"]
        input_ids = encodings["input_ids"]
        input_tokens = datamodule.tokenizer.decode(
            input_ids, skip_special_tokens=True)
        target_ids = encodings["labels"]
        target_tokens = datamodule.tokenizer.decode(
            target_ids, skip_special_tokens=True)
        pred_ids = example["json"]["token_predictions"]
        pred_tokens = datamodule.tokenizer.decode(
            pred_ids, skip_special_tokens=True)
        task_predictions = {}
        for (task, preds) in example["json"]["tasks"].items():
            task_predictions[task] = {
                "predictions": preds,
                "labels": example["json"]["labels"][task]}
        if "__url__" not in example:
            example["__url__"] = None
        yield {"__key__": example["__key__"],
               "__url__": example["__url__"],
               "inputs": input_tokens,
               "targets": target_tokens,
               "predictions": pred_tokens,
               "tasks": task_predictions}


def format_results_as_markdown_table(results_dict):
    columns = ["task"]
    rows = defaultdict(dict)
    for (k, v) in results_dict.items():
        tmp = k.split('_')
        task, metric = tmp[0], tmp[-1]
        if metric not in columns:
            columns.append(metric)
        rows[task][metric] = v

    max_col_len = max([len(col) for col in columns]) + 1
    max_task_len = max([len(task) for task in rows.keys()]) + 1
    max_len = max([max_col_len, max_task_len])
    header = [f" {col:<{max_len}}" for col in columns]
    table = '|' + '|'.join(header) + '|'
    table += '\n|' + ('|'.join(['-' * (max_len + 1) for _ in header])) + '|'
    for (task, metrics) in rows.items():
        table += '\n|'
        for col in columns:
            if col == "task":
                table += f" {task:<{max_len}}|"
            else:
                try:
                    table += f" {metrics[col]:<{max_len}.4f}|"
                except KeyError:
                    table += f" {'-':<{max_len}}|"
    return table


if __name__ == "__main__":
    args = parse_args()
    main(args)
