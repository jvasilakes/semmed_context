import copy

import torch.nn as nn
from transformers.models.bert.modeling_bert import BertLayer, BertAttention

from .util import register_task_encoder


@register_task_encoder("identity")
class Identity(nn.Identity):
    """
    Identity but allows for variable keyword arguments.
    Keyword arguments are ignored. Only the first
    argument to forward is returned.
    """
    def forward(self, x, *args, **kwargs):
        return x


register_task_encoder("bert-layer")(BertLayer)


@register_task_encoder("bert-attention")
class BertAttentionEncoder(nn.Module):

    def __init__(self, bert_config, decoder_size):
        super().__init__()
        self.decoder_size = decoder_size
        self.encoder = nn.Linear(bert_config.hidden_size, self.decoder_size)
        attn_config = copy.deepcopy(bert_config)
        attn_config.hidden_size = decoder_size
        attn_layer = BertAttention(attn_config)
        self.attention_layers = nn.ModuleList(
            [copy.deepcopy(attn_layer) for _ in range(6)])
        self.decoder = nn.Linear(self.decoder_size, bert_config.hidden_size)

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.encoder(hidden_states)
        for attn_layer in self.attention_layers:
            hidden_states = attn_layer(hidden_states, attention_mask)[0]
        hidden_states = self.decoder(hidden_states)
        return hidden_states
