from typing import List, Tuple, Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from transformers.models.bart.modeling_bart import (
    BartConfig, BartPretrainedModel, BartEncoder, BartDecoder,
    BartForConditionalGeneration, shift_tokens_right)
from transformers.modeling_outputs import BaseModelOutput

from .util import Seq2SeqVAEModelOutput, Seq2SeqVAELMOutput
from .cross_attention import CrossAttention


def kl_divergence(mu, logvar):
    kl = 0.5 * (torch.exp(logvar) + torch.pow(mu, 2) - 1 - logvar)
    kl = kl.mean(0).sum()
    return kl


class BartVAEModel(BartPretrainedModel):
    """
    Basically BART-CVAE from

    DiscoDVT: Generating Long Text with Discourse-Aware
        Discrete Variational Transformer
    Ji and Huang 2021
    http://arxiv.org/abs/2110.05999
    """
    _tied_weights_keys = ["encoder.embed_tokens.weight",
                          "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig, latent_structure: dict,
                 tasks_spec: dict = None, special_tokens_map: dict = None):
        super().__init__(config)
        self.latent_structure = latent_structure
        self.tasks_spec = tasks_spec or {}
        self.special_tokens_map = special_tokens_map or {}

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        hidden_size = config.hidden_size
        self.context2params = nn.ModuleDict()
        self.z_projectors = nn.ModuleDict()
        self.discriminators = nn.ModuleDict()
        total_latent_dim = 0
        attn_head_dim = 64
        for (latent_name, latent_dim) in self.latent_structure.items():
            #indim = attn_head_dim
            #if latent_name == "entity":
            #    indim = hidden_size
            indim = hidden_size
            self.context2params[latent_name] = nn.Linear(
                    # 2 for mu, logvar
                    indim, 2 * latent_dim)
            self.z_projectors[latent_name] = nn.Linear(
                latent_dim, hidden_size)
            total_latent_dim += latent_dim
            if latent_name in self.tasks_spec.keys():
                self.discriminators[latent_name] = nn.Linear(
                    latent_dim, self.tasks_spec[latent_name])
        num_latents = len(set(self.latent_structure.keys()) - {"entity"})
        # TODO: add sparse to config
        self.latent_cross_attn = CrossAttention(
            dim=hidden_size, heads=num_latents, dim_head=attn_head_dim,
            context_dim=hidden_size, sparse=False)
        self.layernorm = nn.LayerNorm(hidden_size)

        self.sentence_projector = nn.Linear(2 * hidden_size, hidden_size)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqVAEModelOutput]:
        # different to other models, Bart automatically creates
        # decoder_input_ids from input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."  # noqa
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id,
                self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # noqa
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  # noqa
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache  # noqa
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # noqa

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs,
        # we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,  # noqa
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,  # noqa
            )

        # ========================================================
        # Step 1: Get the entity representations by masking the hidden_states
        # ========================================================
        hidden_states = encoder_outputs.last_hidden_state
        # TODO: remove hard coded entity start id
        entity_mask = input_ids.eq(50128).to(hidden_states.device)
        if len(torch.unique_consecutive(entity_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of entity_start tokens.")  # noqa
        entity_reps = hidden_states[entity_mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1))

        # ========================================================
        # Step 2: Get the sentence representation by concatenating
        # and projecting the entity representations.
        # ========================================================
        sentence_representation = self.sentence_projector(
            entity_reps.view(hidden_states.size(0), -1))

        # ========================================================
        # Step 3: Compute attention between the sentence representation and
        # the hidden states using a separate attention head for each
        # latent space that is not the entity space.
        # ========================================================
        #sent_out, hidden_out, sent_attn, hidden_attn, sent_by_head, hidden_by_head = self.latent_cross_attn(  # noqa
        #    sentence_representation.unsqueeze(1), hidden_states,
        #    context_mask=attention_mask,
        #    return_attn=True, return_head_output=True)

        # ========================================================
        # Step 4a: Assign a latent space to the output of each attention head.
        # ========================================================
        #sent_by_head = sent_by_head.squeeze(2)
        #latents_no_entities = sorted(set(self.latent_structure.keys()) - {"entity"})  # noqa
        #sent_by_latent = dict(zip(latents_no_entities,
        #                          sent_by_head.permute(1, 0, 2)))

        # ========================================================
        # Step 4b: Compute the latent values for each latent space
        # besides the entity space.
        # ========================================================
        latents_no_entities = sorted(set(self.latent_structure.keys()) - {"entity"})  # noqa
        sent_by_latent = {latent_name: sentence_representation
                          for latent_name in latents_no_entities}
        latent_params = self.compute_latent_params(sent_by_latent)
        zs = []
        task_logits = {}
        for latent_name in sorted(latent_params.keys()):
            z = latent_params[latent_name].rsample()
            zs.append(self.z_projectors[latent_name](z))
            if latent_name in self.discriminators.keys():
                task_logits[latent_name] = self.discriminators[latent_name](z)

        # ========================================================
        # Step 4c: Compute the latent values for each entity,
        # projecting them into the same latent space.
        # ========================================================
        for (i, entity_rep) in enumerate(entity_reps.permute(1, 0, 2)):
            entity_params = self.compute_latent_params({"entity": entity_rep})
            latent_params[f"entity{i}"] = entity_params["entity"]
            z = entity_params["entity"].rsample()
            zs.append(self.z_projectors["entity"](z))

        # ========================================================
        # Step 5: Decode providing the latent values in place of the
        # encoder hidden states. This means that the decoder will
        # only compute cross attentions between the output and the latents.
        # ========================================================
        decoder_init = self.layernorm(torch.stack(zs, dim=1))
        encoder_attention_mask = attention_mask[:, :decoder_init.size(1)]
        # Skip connection + layer normalization
        # decoder_init = self.layernorm(hidden_states + hidden_out)

        # decoder outputs consists of
        # (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=decoder_init,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return Seq2SeqVAEModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            latent_params=latent_params,
            task_logits=task_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def compute_latent_params(self, context_per_latent):
        latent_dists = {}
        for (name, context) in context_per_latent.items():
            layer = self.context2params[name]
            params = layer(context)
            mu, logvar = params.chunk(2, dim=-1)
            dist = D.Normal(mu, torch.exp(logvar))
            latent_dists[name] = dist
        return latent_dists

    def _mask_and_pool_hidden(self, input_ids, hidden_states, mask_token_ids):
        mask = torch.zeros_like(input_ids).bool()
        for mask_id in mask_token_ids:
            mask = torch.logical_or(mask, input_ids.eq(mask_id))
        if len(torch.unique_consecutive(mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of special tokens.")  # noqa
        token_reps = hidden_states[mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1))
        sentence_representation = token_reps.max(1).values
        return sentence_representation


class BartVAEForConditionalGeneration(BartForConditionalGeneration):

    def __init__(self, config: BartConfig, latent_structure,
                 tasks_spec, label_weights=None):
        super().__init__(config)
        self.model = BartVAEModel(config, latent_structure, tasks_spec)
        self.label_weights = label_weights or {}
        self.label_weights = {task: torch.as_tensor(ws)
                              for (task, ws) in self.label_weights.items()}
        self.loss_fct = nn.CrossEntropyLoss(
            ignore_index=self.config.pad_token_id)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        task_labels: Optional[Dict[str, torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqVAELMOutput]:

        if return_dict is None:
            return_dict = self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id,
                    self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            masked_lm_loss = self.loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            rval = output
            if masked_lm_loss is not None:
                rval = ((masked_lm_loss,) + output)
            return rval

        kls = {latent_name: D.kl_divergence(dist, D.Normal(0, 1)).mean(0).sum()
               for (latent_name, dist) in outputs.latent_params.items()}

        task_losses = {}
        if task_labels is not None:
            for (task, logits) in outputs.task_logits.items():
                if task in task_labels:
                    try:
                        weights = self.label_weights[task].to(self.device)
                    except KeyError:
                        weights = None
                    if logits.dim() == 1:
                        if weights is not None:
                            assert len(weights) == 2
                            pos_weight = weights[1]
                        else:
                            pos_weight = None
                        task_losses[task] = F.binary_cross_entropy_with_logits(
                            logits, task_labels[task], pos_weight=pos_weight)
                    else:
                        task_losses[task] = F.cross_entropy(
                            logits, task_labels[task], weight=weights)

        return Seq2SeqVAELMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            kls=kls,
            latent_params=outputs.latent_params,
            task_logits=outputs.task_logits,
            task_losses=task_losses,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
