import os
import argparse
import warnings
from collections import defaultdict
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from config import config
from data import SemRepFactDataModule
from models import MODEL_REGISTRY


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="Run model training")
    train_parser.add_argument(
        "config_file", type=str, help="Path to yaml config file.")

    val_parser = subparsers.add_parser("validate", help="Run model validation")
    val_parser.add_argument(
        "config_file", type=str, help="Path to yaml config file.")
    val_parser.add_argument("--split", type=str, default="val",
                            choices=["train", "val", "test"])

    return parser.parse_args()


def main(args):
    config.load_yaml(args.config_file)

    start_time = datetime.now()
    print(f"Experiment start: {start_time}")
    print()
    print(config)

    pl.seed_everything(config.Experiment.random_seed.value, workers=True)

    run_kwargs = {}
    if args.command == "train":
        run_fn = run_train
    elif args.command == "validate":
        run_fn = run_validate
        run_kwargs["datasplit"] = args.split
    else:
        raise argparse.ArgumentError(f"Unknown command '{args.command}'.")
    run_fn(config, **run_kwargs)

    end_time = datetime.now()
    print()
    print(f"Experiment end: {end_time}")
    print(f"  Time elapsed: {end_time - start_time}")


def run_train(config):
    logdir = config.Experiment.logdir.value
    version = config.Experiment.version.value
    exp_name = config.Experiment.name.value
    version_dir = os.path.join(logdir, exp_name, f"version_{version}")
    os.makedirs(version_dir, exist_ok=False)
    config.yaml(outpath=os.path.join(version_dir, "config.yaml"))

    datamodule = SemRepFactDataModule(config)
    datamodule.setup()
    print(datamodule)
    input()

    model_class = MODEL_REGISTRY[config.Model.model_name.value]
    model = model_class.from_config(config, datamodule.label_spec)
    print(model); input()
    model.train()

    logger = TensorBoardLogger(
        save_dir=logdir, version=version, name=exp_name)

    filename_fmt = f"{{epoch:02d}}"
    checkpoint_cb = ModelCheckpoint(
        monitor="avg_val_loss", mode="min", filename=filename_fmt)

    if torch.cuda.is_available():
        available_gpus = min(1, torch.cuda.device_count())
    else:
        available_gpus = 0
        warnings.warn("No CUDA devices found. Using CPU.")
    print(f"GPUs: {available_gpus}")
    trainer = pl.Trainer(
        max_epochs=config.Training.epochs.value,
        gpus=available_gpus,
        logger=logger,
        callbacks=[checkpoint_cb],
        log_every_n_steps=1,
        deterministic=True,
        check_val_every_n_epoch=1)
    trainer.fit(model, datamodule=datamodule)


def run_validate(config, datasplit):
    datamodule = SemRepFactDataModule(config)
    datamodule.setup()

    model_class = MODEL_REGISTRY[config.Model.model_name.value]
    model = model_class.from_config(config, datamodule.label_spec)
    model.eval()

    if torch.cuda.is_available():
        available_gpus = min(1, torch.cuda.device_count())
    else:
        available_gpus = 0
        warnings.warn("No CUDA devices found. Using CPU.")
    print(f"GPUs: {available_gpus}")

    trainer = pl.Trainer(
        logger=False,
        deterministic=True,
        gpus=available_gpus)

    if datasplit == "train":
        val_dataloader_fn = datamodule.train_dataloader
    elif datasplit == "val":
        val_dataloader_fn = datamodule.val_dataloader
    elif datasplit == "test":
        val_dataloader_fn = datamodule.test_dataloader
    else:
        raise ValueError(f"Unknown data split '{datasplit}'")
    val_dataloader = val_dataloader_fn()
    results = trainer.validate(
        model, dataloaders=val_dataloader, verbose=False)[0]
    format_results_as_markdown_table(results)


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
    print(table)


if __name__ == "__main__":
    args = parse_args()
    main(args)
