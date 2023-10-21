from typing import Optional, Tuple, Dict

import torch
from dataclasses import dataclass
from transformers.file_utils import ModelOutput


MODEL_REGISTRY = {}


def register_model(name):
    def add_to_registry(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return add_to_registry


TASK_ENCODER_REGISTRY = {}


def register_task_encoder(name):
    def add_to_registry(cls):
        TASK_ENCODER_REGISTRY[name] = cls
        return cls
    return add_to_registry


ENTITY_POOLER_REGISTRY = {}


def register_entity_pooler(name):
    def add_to_registry(cls):
        ENTITY_POOLER_REGISTRY[name] = cls
        return cls
    return add_to_registry


LOSS_REGISTRY = {}


def register_loss(name):
    def add_to_registry(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return add_to_registry


DISTRIBUTION_REGISTRY = {}


def register_distribution(name):
    def add_to_registry(cls):
        DISTRIBUTION_REGISTRY[name] = cls
        return cls
    return add_to_registry


@dataclass
class SequenceClassifierOutputWithTokenMask(ModelOutput):
    logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    mask_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    mask: Optional[torch.FloatTensor] = None


@dataclass
class VAEEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    latent_params: Dict = None
    task_logits: Dict[str, torch.FloatTensor] = None


@dataclass
class Seq2SeqVAEModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    latent_params: Dict = None
    task_logits: Dict[str, torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Seq2SeqVAELMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    kls: Dict[str, torch.FloatTensor] = None
    latent_params: Dict = None
    task_logits: Dict[str, torch.FloatTensor] = None
    task_losses: Dict[str, torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
