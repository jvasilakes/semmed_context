import os
import json
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import BartForConditionalGeneration
from transformers import logging as transformers_logging
from sklearn.metrics import precision_recall_fscore_support

from .util import register_model
from .bart_vae import BartVAEForConditionalGeneration


# Ignore warning that BartModel is not using some parameters.
transformers_logging.set_verbosity_error()


class AbstractBartSummaryModel(pl.LightningModule):

    @classmethod
    def from_config(cls, config, logdir=None):
        return cls(config.Model.bart_model_name_or_path.value,
                   lr=config.Training.lr.value,
                   weight_decay=config.Training.weight_decay.value,
                   logdir=logdir)

    def __init__(self, bart_model_name_or_path, lr=2e-5,
                 weight_decay=0.0, logdir=None):
        super().__init__()
        self.bart_model_name_or_path = bart_model_name_or_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.bart = None
        self.epoch = 0

        self.logdir = logdir
        if self.logdir is not None:
            os.makedirs(os.path.join(self.logdir, "latents"), exist_ok=True)

    def forward(self, bart_inputs, task_labels):
        bart_outputs = self.bart(**bart_inputs, task_labels=task_labels)
        return bart_outputs

    def get_model_outputs(self, batch):
        outputs = self(batch["json"]["encoded"],
                       task_labels=batch["json"]["labels"])
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.get_model_outputs(batch)
        self.log("train_recon_loss", outputs.loss)
        total_loss = outputs.loss
        # TODO: add KL weights to config
        for (key, kl) in outputs.kls.items():
            total_loss += 0.01 * kl
            self.log(f"train_kl_{key}", kl)
            if key in outputs.task_losses:
                task_loss = outputs.task_losses[key]
                total_loss += task_loss
                self.log(f"train_task_loss_{key}", task_loss)
        return {"loss": total_loss,
                "latent_params": outputs.latent_params,
                "task_logits": outputs.task_logits,
                "task_labels": batch["json"]["labels"]}

    def training_epoch_end(self, metrics):
        loss_vals = []
        epoch_zs = defaultdict(list)
        task_preds = defaultdict(list)
        task_labels = defaultdict(list)
        for batch in metrics:
            loss_vals.append(batch["loss"].detach().cpu().numpy())
            for (latent, params) in batch["latent_params"].items():
                epoch_zs[latent].extend(
                    params.rsample().detach().cpu().tolist())
            for (task, logits) in batch["task_logits"].items():
                task_preds[task].extend(logits.argmax(-1).detach().cpu().tolist())  # noqa
                task_labels[task].extend(batch["task_labels"][task].cpu().tolist())  # noqa

        for (task, preds) in task_preds.items():
            p, r, f, _ = precision_recall_fscore_support(
                task_labels[task], preds, average="macro", zero_division=0)
            self.log(f"train_precision_{task}", p)
            self.log(f"train_recall_{task}", r)

        if self.logdir is not None:
            with open(f"{self.logdir}/latents/{self.epoch}.json", 'w') as outF:
                json.dump(dict(epoch_zs), outF)
        self.epoch += 1

    def validation_step(self, batch, batch_idx):
        outputs = self.get_model_outputs(batch)

        total_loss = outputs.loss
        # TODO: add KL weights to config
        for (key, kl) in outputs.kls.items():
            total_loss += 0.01 * kl
            if key in outputs.task_losses:
                task_loss = outputs.task_losses[key]
                total_loss += task_loss

        token_pred_ids = outputs.logits.argmax(-1)
        task_preds = {task: logits.argmax(-1).cpu().tolist() for (task, logits)
                      in outputs.task_logits.items()}
        metrics = {"recon_loss": outputs.loss.detach().cpu().item(),
                   "task_losses": {key: loss.detach().cpu().item()
                                   for (key, loss) in outputs.task_losses.items()},  # noqa
                   "kls": {key: kl.detach().cpu().item()
                           for (key, kl) in outputs.kls.items()},
                   "token_preds": token_pred_ids,
                   "token_labels": batch["json"]["encoded"]["labels"],
                   "task_preds": task_preds,
                   "task_labels": {key: labs.cpu().tolist() for (key, labs)
                                   in batch["json"]["labels"].items()}
                   }
        return metrics

    def validation_epoch_end(self, all_metrics):
        recon_losses = []
        latent_kls = defaultdict(list)
        task_losses = defaultdict(list)
        token_preds = []
        token_labels = []
        task_preds = defaultdict(list)
        task_labels = defaultdict(list)
        for ex in all_metrics:
            recon_losses.append(ex["recon_loss"])
            for (latent, kl) in ex["kls"].items():
                latent_kls[latent].append(kl)
            for (task, loss) in ex["task_losses"].items():
                task_losses[task].append(loss)
                task_preds[task].extend(ex["task_preds"][task])
                task_labels[task].extend(ex["task_labels"][task])
            token_preds.append(ex["token_preds"])
            token_labels.append(ex["token_labels"])
        self.log("val_recon_loss", np.mean(recon_losses))
        for (latent, kls) in latent_kls.items():
            self.log(f"val_kl_{latent}", np.mean(kls))
        for (task, losses) in task_losses.items():
            self.log(f"val_task_loss_{task}", np.mean(losses))
        self.log("val_total_loss",
                 (np.mean(recon_losses) +
                  np.sum([np.mean(kls) for kls in latent_kls.values()]) +
                  np.sum([np.mean(losses) for losses in task_losses.values()]))
                 )

        for (task, preds) in task_preds.items():
            p, r, f, _ = precision_recall_fscore_support(
                task_labels[task], preds, average="macro", zero_division=0)
            self.log(f"val_precision_{task}", p)
            self.log(f"val_recall_{task}", r)

    def predict_step(self, batch, batch_idx):
        outputs = self.get_model_outputs(batch)
        pred_ids = outputs.logits.argmax(-1)
        batch_cp = deepcopy(batch)
        batch_cp["json"]["predictions"] = pred_ids
        return batch_cp

    def configure_optimizers(self):
        opt = AdamW(self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay)
        return opt


@register_model("default")
class BartSummaryModel(AbstractBartSummaryModel):

    def __init__(self, bart_model_name_or_path, lr=2e-5,
                 weight_decay=0.0, logdir=None, **kwargs):
        super().__init__(bart_model_name_or_path, lr, weight_decay, logdir)
        self.bart = BartForConditionalGeneration.from_pretrained(
            self.bart_model_name_or_path)
        self.save_hyperparameters()


@register_model("vae")
class BartVAESummaryModel(AbstractBartSummaryModel):

    @classmethod
    def from_config(cls, config, logdir=None, tasks_spec=None):
        return cls(config.Model.bart_model_name_or_path.value,
                   latent_structure=config.Model.latent_structure.value,
                   tasks_spec=tasks_spec,
                   label_weights=config.Data.label_weights.value,
                   lr=config.Training.lr.value,
                   weight_decay=config.Training.weight_decay.value,
                   logdir=logdir)

    def __init__(self, bart_model_name_or_path, latent_structure,
                 tasks_spec=None, label_weights=None,
                 lr=2e-5, weight_decay=0.0, logdir=None):
        super().__init__(bart_model_name_or_path, lr, weight_decay, logdir)
        self.latent_structure = latent_structure
        self.bart = BartVAEForConditionalGeneration.from_pretrained(
            self.bart_model_name_or_path, latent_structure=latent_structure,
            tasks_spec=tasks_spec, label_weights=label_weights)
        self.save_hyperparameters()
