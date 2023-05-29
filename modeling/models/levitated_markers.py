import warnings
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertConfig, BertModel
from transformers import logging as transformers_logging
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import precision_recall_fscore_support

from .util import register_model, ENTITY_POOLER_REGISTRY


# Ignore warning that BertModel is not using some parameters.
transformers_logging.set_verbosity_error()


@register_model("levitated_marker")
class LevitatedMarkerClassificationModel(pl.LightningModule):

    @classmethod
    def from_config(cls, config, label_spec):
        return cls(config.Model.bert_model_name_or_path.value,
                   label_spec=label_spec,
                   entity_pool_fn=config.Model.entity_pool_fn.value,
                   levitated_pool_fn=config.Model.levitated_pool_fn.value,
                   lr=config.Training.lr.value,
                   weight_decay=config.Training.weight_decay.value,
                   dropout_prob=config.Training.dropout_prob.value)

    def __init__(
            self,
            bert_model_name_or_path,
            label_spec,
            entity_pool_fn,
            levitated_pool_fn,
            lr=1e-3,
            weight_decay=0.0,
            dropout_prob=0.0):
        super().__init__()
        self.bert_model_name_or_path = bert_model_name_or_path
        self.label_spec = label_spec
        self.entity_pool_fn = entity_pool_fn
        self.levitated_pool_fn = levitated_pool_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_prob = dropout_prob

        self.bert_config = BertConfig.from_pretrained(
            self.bert_model_name_or_path)
        self.bert = BertModel.from_pretrained(
            self.bert_model_name_or_path, config=self.bert_config)

        pooler_insize = pooler_outsize = self.bert_config.hidden_size
        # Multiply by 2 because we have subject and object.
        self.entity_pooler = ENTITY_POOLER_REGISTRY[self.entity_pool_fn](
            2 * pooler_insize, pooler_outsize)

        self.levitated_pooler = ENTITY_POOLER_REGISTRY[self.levitated_pool_fn](
            pooler_insize, pooler_outsize)

        self.classifier_heads = nn.ModuleDict()
        # Multiply by 2 because we'll concat entity and
        # levitated marker representations.
        classifier_insize = 2 * self.bert_config.hidden_size
        for (task, num_labels) in label_spec.items():
            self.classifier_heads[task] = nn.Sequential(
                nn.Dropout(self.dropout_prob),
                nn.Linear(classifier_insize, num_labels)
            )

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.save_hyperparameters()

    def forward(self, bert_inputs, entity_token_idxs,
                levitated_token_idxs, labels=None):
        bert_outputs = self.bert(**bert_inputs,
                                 output_hidden_states=True,
                                 output_attentions=True,
                                 return_dict=True)

        pooled_entities = self.entity_pooler(
            bert_outputs.last_hidden_state, entity_token_idxs)

        pooled_levitated = self.levitated_pooler(
            bert_outputs.last_hidden_state, levitated_token_idxs)

        pooled_output = torch.cat([pooled_entities, pooled_levitated], dim=1)

        clf_outputs = {}
        for (task, clf_head) in self.classifier_heads.items():
            logits = clf_head(pooled_output)
            if labels is not None:
                clf_loss = self.loss_fn(logits.view(-1, self.label_spec[task]),
                                        labels[task].view(-1))
            else:
                clf_loss = None
            clf_outputs[task] = SequenceClassifierOutput(
                loss=clf_loss,
                logits=logits,
                hidden_states=bert_outputs.hidden_states,
                attentions=bert_outputs.attentions)
        return clf_outputs

    def get_model_outputs(self, batch):
        subj_obj_idxs = torch.stack(
            [batch["json"]["subject_idxs"],
             batch["json"]["object_idxs"]])
        # unsqueeze(0) because we have to introduce the function dimension.
        levitated_idxs = batch["json"]["levitated_idxs"].unsqueeze(0)
        outputs_by_task = self(batch["json"]["encoded"],
                               subj_obj_idxs, levitated_idxs,
                               labels=batch["json"]["labels"])
        return outputs_by_task

    def training_step(self, batch, batch_idx):
        outputs_by_task = self.get_model_outputs(batch)
        total_loss = torch.tensor(0.).to(self.device)
        for (task, outputs) in outputs_by_task.items():
            total_loss += outputs.loss
            self.log(f"train_loss_{task}", outputs.loss)
        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs_by_task = self.get_model_outputs(batch)
        metrics = {}
        for (task, outputs) in outputs_by_task.items():
            preds = torch.argmax(outputs.logits, axis=1)
            metrics[task] = {"loss": outputs.loss,
                             "preds": preds,
                             "labels": batch["json"]["labels"][task]}
        return metrics

    def predict_step(self, batch, batch_idx):
        outputs_by_task = self.get_model_outputs(batch)
        batch_cp = deepcopy(batch)
        batch_cp["json"]["predictions"] = {}
        for (task, outputs) in outputs_by_task.items():
            softed = torch.nn.functional.softmax(outputs.logits, dim=1)
            preds = torch.argmax(softed, dim=1)
            batch_cp["json"]["predictions"][task] = preds
        return batch_cp

    def validation_epoch_end(self, all_metrics):
        losses_by_task = defaultdict(list)
        preds_by_task = defaultdict(list)
        labels_by_task = defaultdict(list)
        for example in all_metrics:
            for (task, metrics) in example.items():
                losses_by_task[task].append(metrics["loss"].detach().cpu().numpy())  # noqa
                preds_by_task[task].extend(metrics["preds"].detach().cpu().numpy())  # noqa
                labels_by_task[task].extend(metrics["labels"].detach().cpu().numpy())  # noqa

        val_losses = []
        f1s = []
        for task in losses_by_task.keys():
            task_losses = np.mean(losses_by_task[task])
            task_preds = np.array(preds_by_task[task])
            task_labels = np.array(labels_by_task[task])

            self.log(f"{task}_val_loss", task_losses, prog_bar=False)
            val_losses.append(task_losses)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, r, f1, _ = precision_recall_fscore_support(
                    task_labels, task_preds, average="macro")
                f1s.append(f1)
                res = {f"{task}_precision": p,
                       f"{task}_recall": r,
                       f"{task}_f1": f1}
                self.log_dict(res, prog_bar=False)

        self.log_dict({"avg_val_loss": np.mean(val_losses),
                       "avg_val_f1": np.mean(f1s)})

    def configure_optimizers(self):
        opt = AdamW(self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay)
        return opt
