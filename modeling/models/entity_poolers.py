import torch

from .util import register_entity_pooler


class BaseEntityPooler(torch.nn.Module):

    def __init__(self, insize, outsize):
        super().__init__()
        self.insize = insize
        self.outsize = outsize
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(self.insize, self.outsize),
            torch.nn.Tanh())

    def forward(self, hidden, token_idxs):
        # token_idxs = [F, B, T, S]
        #  F: entity functions, e.g., subject, object
        #  B: batch
        #  E: entity span
        #  S: start/end index of the entity
        assert token_idxs.dim() == 4, "token_idxs = [F, B, T, S]"
        all_pooled = []
        # token_idxs is grouped by entity function.
        # E.g., all subjects in a batch, then all objects.
        for entity_idxs in token_idxs:
            # So we create a batched entity mask for each function
            entity_mask = self.get_token_mask_from_indices(
                entity_idxs, hidden.size())
            entity_mask = entity_mask.to(hidden.device)
            masked_hidden = hidden * entity_mask
            pooled = self.pool_fn(masked_hidden, entity_mask)
            all_pooled.append(pooled)
        # Then we concatenate the entity representations for each
        # entity function.
        all_pooled = torch.cat(all_pooled, dim=1)
        # and project it down.
        projected = self.output_layer(all_pooled)
        return projected

    def string(self):
        return f"{self.__class__.__name__}({self.insize}, {self.outsize})"

    def get_token_mask_from_indices(self, token_idxs, hidden_size):
        token_mask = torch.zeros(hidden_size)
        for (batch_idx, spans) in enumerate(token_idxs):
            for span in spans:
                span_idxs = torch.arange(*span)
                token_mask[batch_idx, span_idxs, :] = 1.
        return token_mask

    def pool_fn(self, masked_hidden, token_mask):
        raise NotImplementedError()


@register_entity_pooler("max")
class MaxEntityPooler(BaseEntityPooler):

    def pool_fn(self, masked_hidden, token_mask):
        # Replace masked with -inf to avoid zeroing out
        # hidden dimensions if the non-masked values are all negative.
        masked_hidden[torch.logical_not(token_mask)] = -torch.inf
        pooled = torch.max(masked_hidden, axis=1)[0]
        # Just in case all values are -inf
        pooled = torch.nan_to_num(pooled)
        return pooled


@register_entity_pooler("first")
class FirstEntityPooler(BaseEntityPooler):

    def pool_fn(self, masked_hidden, token_mask):
        first_nonzero_idxs = token_mask.max(1).indices[:, 0]
        batch_idxs = torch.arange(token_mask.size(0))
        pooled = masked_hidden[batch_idxs, first_nonzero_idxs, :]
        return pooled
