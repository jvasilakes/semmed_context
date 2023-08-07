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
from sklearn.metrics import precision_recall_fscore_support

from .util import (register_model, ENTITY_POOLER_REGISTRY,
                   TASK_ENCODER_REGISTRY, LOSS_REGISTRY,
                   SequenceClassifierOutputWithTokenMask)


# Ignore warning that BertModel is not using some parameters.
transformers_logging.set_verbosity_error()


@register_model("levitated_marker_hier")
class LevitatedMarkerHierModel(pl.LightningModule):

    @classmethod
    def from_config(cls, config, label_spec):
        loss = LOSS_REGISTRY[config.Training.loss_fn.value](reduction="mean")
        return cls(config.Model.bert_model_name_or_path.value,
                   label_spec=label_spec,
                   loss_fn=loss,
                   task_encoder_type=config.Model.TaskEncoder.encoder_type.value,  # noqa
                   task_encoder_kwargs=config.Model.TaskEncoder.init_kwargs.value,  # noqa
                   entity_pool_fn=config.Model.entity_pool_fn.value,
                   project_entities=config.Model.project_entities.value,
                   levitated_pool_fn=config.Model.levitated_pool_fn.value,
                   lr=config.Training.lr.value,
                   weight_decay=config.Training.weight_decay.value,
                   dropout_prob=config.Training.dropout_prob.value)

    def __init__(
            self,
            bert_model_name_or_path,
            label_spec,
            loss_fn,
            task_encoder_type,
            task_encoder_kwargs,
            entity_pool_fn,
            project_entities,
            levitated_pool_fn,
            lr=1e-3,
            weight_decay=0.0,
            dropout_prob=0.0):
        super().__init__()

        if "Predicate" not in label_spec.keys():
            raise KeyError("LevitatedMarkerHier requires 'Predicate' as a task.")  # noqa
        if len(label_spec.keys()) < 2:
            raise ValueError("LevitatedMarkerHier requires at least 2 tasks.")

        self.bert_model_name_or_path = bert_model_name_or_path
        self.label_spec = label_spec
        self.loss_fn = loss_fn
        self.task_encoder_type = task_encoder_type
        self.task_encoder_kwargs = task_encoder_kwargs
        self.entity_pool_fn = entity_pool_fn
        self.project_entities = project_entities
        self.levitated_pool_fn = levitated_pool_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_prob = dropout_prob

        self.bert_config = BertConfig.from_pretrained(
            self.bert_model_name_or_path)
        self.bert = BertModel.from_pretrained(
            self.bert_model_name_or_path, config=self.bert_config)

        # Multiply by 2 because we have subject and object.
        bert_hidden_size = self.bert_config.hidden_size
        entity_pooler_insize = 2 * bert_hidden_size
        entity_pooler_outsize = bert_hidden_size
        if self.project_entities is False:
            # Don't project the entities down, so keep outsize equal to insize
            entity_pooler_outsize = entity_pooler_insize
        self.entity_pooler = ENTITY_POOLER_REGISTRY[self.entity_pool_fn](
            entity_pooler_insize, entity_pooler_outsize)

        lev_pooler_insize = lev_pooler_outsize = bert_hidden_size
        # The alignment model computes attentions between the entity_pooler
        # output and the hidden representations of each token.
        lev_alignment_insize = entity_pooler_outsize + bert_hidden_size
        classifier_insize = entity_pooler_outsize + lev_pooler_outsize
        self.task_encoders = nn.ModuleDict()
        self.levitated_poolers = nn.ModuleDict()
        self.classifier_heads = nn.ModuleDict()
        for (task, num_labels) in label_spec.items():
            self.task_encoders[task] = TASK_ENCODER_REGISTRY[self.task_encoder_type](  # noqa
                self.bert_config, **self.task_encoder_kwargs)
            lev_pooler_insize_ = lev_pooler_insize
            lev_alignment_insize_ = lev_alignment_insize
            classifier_insize_ = classifier_insize
            if task != "Predicate":
                # Auxiliary tasks take in the pooled predicate representation
                # (which is entities_pooled + levitated_pooled = 768 + 768)
                # as well as their own levitated_pooled (768) so
                # total is 768 * 3 = 2304.
                lev_alignment_insize_ = lev_alignment_insize + bert_hidden_size
                classifier_insize_ = classifier_insize + bert_hidden_size
            self.levitated_poolers[task] = ENTITY_POOLER_REGISTRY[self.levitated_pool_fn](  # noqa
                lev_pooler_insize_, lev_pooler_outsize, lev_alignment_insize_)
            self.classifier_heads[task] = nn.Sequential(
                nn.Dropout(self.dropout_prob),
                nn.Linear(classifier_insize_, num_labels)
            )

        self.save_hyperparameters()

    def forward(self, bert_inputs, entity_token_idxs,
                levitated_token_idxs, labels=None):
        bert_outputs = self.bert(**bert_inputs,
                                 output_hidden_states=True,
                                 output_attentions=True,
                                 return_dict=True)

        pooled_entities = self.entity_pooler(
            bert_outputs.last_hidden_state, entity_token_idxs)

        extended_mask = get_extended_attention_mask(
            bert_inputs["attention_mask"], bert_inputs["input_ids"].size())

        clf_outputs = {}

        # First, get the outputs for the Predicate task in the usual fashion.
        predicate_hidden = self.task_encoders["Predicate"](
                bert_outputs.last_hidden_state, extended_mask)
        if isinstance(predicate_hidden, tuple):
            # BertLayer outputs a tuple, but other task_encoders don't.
            predicate_hidden = predicate_hidden[0]
        tmp = self.levitated_poolers["Predicate"](
            predicate_hidden, levitated_token_idxs, pooled_entities)
        predicate_levitated, pred_lev_attentions = tmp
        predicate_pooled = torch.cat([pooled_entities, predicate_levitated],
                                     dim=1)

        pred_logits = self.classifier_heads["Predicate"](predicate_pooled)
        if labels is not None:
            predicate_loss = self.loss_fn(
                pred_logits.view(-1, self.label_spec["Predicate"]),
                labels["Predicate"].view(-1))
        else:
            predicate_loss = None
        clf_outputs["Predicate"] = SequenceClassifierOutputWithTokenMask(
            loss=predicate_loss,
            logits=pred_logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
            mask=pred_lev_attentions[0])

        # Then get the outputs for the other tasks, using the pooled output of
        # the Predicate task instead of pooled_entities.
        # detach() so we don't backprop gradients from other tasks through
        # the Predicate layers.
        predicate_pooled_detached = predicate_pooled.detach()
        for (task, clf_head) in self.classifier_heads.items():
            if task == "Predicate":
                continue
            task_hidden = self.task_encoders[task](
                bert_outputs.last_hidden_state, extended_mask)
            if isinstance(task_hidden, tuple):
                # BertLayer outputs a tuple, but other task_encoders don't.
                task_hidden = task_hidden[0]
            tmp = self.levitated_poolers[task](
                task_hidden, levitated_token_idxs,
                predicate_pooled_detached)
            pooled_levitated, levitated_attentions = tmp
            pooled_output = torch.cat(
                [predicate_pooled_detached, pooled_levitated], dim=1)
            logits = clf_head(pooled_output)
            if labels is not None:
                clf_loss = self.loss_fn(logits.view(-1, self.label_spec[task]),
                                        labels[task].view(-1))
            else:
                clf_loss = None
            clf_outputs[task] = SequenceClassifierOutputWithTokenMask(
                loss=clf_loss,
                logits=logits,
                hidden_states=bert_outputs.hidden_states,
                attentions=bert_outputs.attentions,
                mask=levitated_attentions[0])
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
                       "avg_val_f1": np.mean(f1s)}, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        outputs_by_task = self.get_model_outputs(batch)
        batch_cp = deepcopy(batch)
        batch_cp["json"]["predictions"] = {}
        batch_cp["json"]["token_masks"] = {}
        for (task, outputs) in outputs_by_task.items():
            softed = torch.nn.functional.softmax(outputs.logits, dim=1)
            preds = torch.argmax(softed, dim=1)
            batch_cp["json"]["predictions"][task] = preds
            batch_cp["json"]["token_masks"][task] = []
            for (i, example_mask) in enumerate(outputs.mask):
                position_ids = batch["json"]["encoded"]["position_ids"][i]
                levitated_idxs = batch["json"]["levitated_idxs"][i]  # noqa
                moved_mask = self.move_levitated_mask_to_tokens(
                    example_mask, position_ids, levitated_idxs)
                batch_cp["json"]["token_masks"][task].append(
                    moved_mask.squeeze())
        return batch_cp

    def move_levitated_mask_to_tokens(self, mask, position_ids, levitated_idxs):  # noqa
        """
        Reassign the attention probabilities from the levitated markers
        to the tokens the stand for.
        """
        new_mask = torch.zeros_like(mask)
        levitated_positions = position_ids[levitated_idxs]
        for li in levitated_positions.unique():
            single_span_idxs = torch.where(levitated_positions == li)
            new_mask[li] = mask[levitated_idxs][single_span_idxs].sum()
        return new_mask

    def configure_optimizers(self):
        opt = AdamW(self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay)
        return opt


def get_extended_attention_mask(attention_mask, input_shape):
    """
    Makes broadcastable attention so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to,
            zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask,
        with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions
    #   [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable
    #   to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]  # noqa
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(  # noqa
                input_shape, attention_mask.shape
            )
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
