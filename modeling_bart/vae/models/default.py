from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import BartForConditionalGeneration
from transformers import logging as transformers_logging

from .util import register_model


# Ignore warning that BertModel is not using some parameters.
transformers_logging.set_verbosity_error()


@register_model("default")
class BARTSummaryModel(pl.LightningModule):

    @classmethod
    def from_config(cls, config):
        return cls(config.Model.bart_model_name_or_path.value,
                   lr=config.Training.lr.value,
                   weight_decay=config.Training.weight_decay.value)

    def __init__(self, bart_model_name_or_path, lr=2e-5, weight_decay=0.0):
        super().__init__()
        self.bart_model_name_or_path = bart_model_name_or_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.bart = BartForConditionalGeneration.from_pretrained(
            self.bart_model_name_or_path)
        self.save_hyperparameters()

    def forward(self, bart_inputs):
        bart_outputs = self.bart(**bart_inputs)
        return bart_outputs

    def get_model_outputs(self, batch):
        outputs = self(batch["json"]["encoded"])
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.get_model_outputs(batch)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.get_model_outputs(batch)
        pred_ids = self.bart.generate(batch["json"]["encoded"]["input_ids"],
                                      do_sample=True, num_beams=2)
        metrics = {"loss": outputs.loss.detach().cpu().item(),
                   "preds": pred_ids,
                   "labels": batch["json"]["encoded"]["labels"]}
        return metrics

    def predict_step(self, batch, batch_idx):
        pred_ids = self.bart.generate(batch["json"]["encoded"]["input_ids"],
                                      do_sample=True, num_beams=2)
        batch_cp = deepcopy(batch)
        batch_cp["json"]["predictions"] = pred_ids
        return batch_cp

    def validation_epoch_end(self, all_metrics):
        losses = []
        preds = []
        labels = []
        for ex in all_metrics:
            losses.append(ex["loss"])
            preds.append(ex["preds"])
            labels.append(ex["labels"])
        self.log("val_loss", np.mean(losses))

    def configure_optimizers(self):
        opt = AdamW(self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay)
        return opt
