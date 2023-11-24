from typing import List, Tuple, Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bart.modeling_bart import (
    BartConfig, BartPretrainedModel, BartEncoder, BartDecoder,
    BartForConditionalGeneration, shift_tokens_right)

from .util import (DISTRIBUTION_REGISTRY,
                   Seq2SeqVAEModelOutput, Seq2SeqVAELMOutput, VAEEncoderOutput)


class BartVAEEncoder(BartEncoder):
    bart_forward = BartEncoder.forward

    def __init__(self, config, shared, latent_structure, tasks_spec=None):
        super().__init__(config, shared)
        # copy latent_structure so we don't modify config
        self.latent_structure = dict(latent_structure)
        self.tasks_spec = tasks_spec or {}

        hidden_size = config.hidden_size
        self.context2params = nn.ModuleDict()
        self.discriminators = nn.ModuleDict()
        total_latent_dim = 0
        for (latent_name, (latent_dim, dist_name)) in self.latent_structure.items():  # noqa
            if isinstance(dist_name, str):
                dist = DISTRIBUTION_REGISTRY[dist_name]
            else:
                dist = dist_name
            self.latent_structure[latent_name][1] = dist
            # x2 for each entity mention
            indim = 2 * hidden_size
            outdim = latent_dim * dist.num_params
            if dist_name == "sle-dirichlet":
                outdim = latent_dim + 1
            self.context2params[latent_name] = nn.Linear(
                    indim, outdim)
            total_latent_dim += latent_dim
            if latent_name in self.tasks_spec.keys():
                # For HardKuma and GumbelSoftmax, just take the value itself.
                # TODO: discrete is the wrong word.
                if dist.discrete is True:
                    disc = nn.Identity()
                else:
                    # Otherwise, project from the distribution to logits.
                    outdim = self.tasks_spec[latent_name]
                    if outdim == 1:
                        transform = nn.Sigmoid()
                    else:
                        transform = nn.Softmax(-1)
                    disc = nn.Sequential(
                            nn.Linear(latent_dim, outdim), transform)
                self.discriminators[latent_name] = disc
        self.z2hidden = nn.Linear(total_latent_dim, hidden_size)

    def forward(self, *args, **kwargs):
        encoder_outputs = self.bart_forward(*args, **kwargs)
        try:
            input_ids = kwargs["input_ids"]
        except KeyError:
            input_ids = args[0]

        # ========================================================
        # Step 1: Get the entity representations by masking the hidden_states
        # ========================================================
        hidden_states = encoder_outputs.last_hidden_state
        # TODO: remove hard coded entity start id
        entity_mask = input_ids.eq(50128).to(hidden_states.device)
        # entity_mask = input_ids.eq(2).to(hidden_states.device)
        if len(torch.unique_consecutive(entity_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of entity_start tokens.")  # noqa
        entity_reps = hidden_states[entity_mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1))

        # ========================================================
        # Step 2: Get the sentence representation by concatenating
        # the entity representations.
        # ========================================================
        sentence_representation = entity_reps.view(entity_reps.size(0), -1)

        # ========================================================
        # Step 3: Compute the latent values for each latent space
        # and predict if there is an associated task.
        # ========================================================
        latent_params = self.compute_latent_params(sentence_representation)
        zs = []
        task_logits = {}
        for latent_name in sorted(latent_params.keys()):
            z = latent_params[latent_name].rsample()
            zs.append(z)
            if latent_name in self.discriminators.keys():
                logits = self.discriminators[latent_name](z).squeeze(1)
                task_logits[latent_name] = logits

        # ========================================================
        # Step 4: Decode providing the latent values in place of the
        # encoder hidden states. This means that the decoder will
        # only compute cross attention between the output and the latents.
        # ========================================================
        zs = torch.cat(zs, dim=1)
        decoder_init = self.z2hidden(zs)
        encoder_attention_mask = kwargs["attention_mask"][:, 0].unsqueeze(1)

        return VAEEncoderOutput(last_hidden_state=decoder_init,
                                hidden_states=decoder_init.unsqueeze(1),
                                attentions=encoder_attention_mask,
                                latent_params=latent_params,
                                task_logits=task_logits)

    def compute_latent_params(self, context):
        latent_dists = {}
        sorted_latents = sorted(self.context2params.keys())
        for latent_name in sorted_latents:
            layer = self.context2params[latent_name]
            params = layer(context)
            dist_cls = self.latent_structure[latent_name][1]
            dist = dist_cls.from_bunched_params(params)
            latent_dists[latent_name] = dist
        return latent_dists


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

    def __init__(self, config: BartConfig,
                 latent_structure: dict,
                 tasks_spec: dict = None):
        super().__init__(config)
        self.latent_structure = latent_structure

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartVAEEncoder(config, self.shared,
                                      latent_structure, tasks_spec)
        self.decoder = BartDecoder(config, self.shared)

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

        # decoder outputs consists of
        # (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_outputs.attentions,
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
            latent_params=encoder_outputs.latent_params,
            task_logits=encoder_outputs.task_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


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
        cross_attn_head_mask: Optional[torch.Tensor] = None,
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

        kls = {latent_name: dist.kl_loss() for (latent_name, dist)
               in outputs.latent_params.items()}

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
                            ex_weights = weights[task_labels[task]]
                        else:
                            ex_weights = None
                        task_losses[task] = F.binary_cross_entropy(
                            logits, task_labels[task].float(),
                            weight=ex_weights)
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
