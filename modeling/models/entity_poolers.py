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
        token_mask = self.get_token_mask_from_indices(
            token_idxs, hidden.size())
        token_mask = token_mask.to(hidden.device)
        masked_hidden = hidden * token_mask
        pooled = self.pool_fn(masked_hidden, token_mask)
        projected = self.output_layer(pooled)
        return projected

    def string(self):
        return f"{self.__class__.__name__}({self.insize}, {self.outsize})"

    def get_token_mask_from_indices(self, token_idxs, hidden_size):
        token_mask = torch.zeros(hidden_size)
        for (example_i, example_idxs) in enumerate(zip(*token_idxs)):
            idxs = torch.concat(example_idxs)
            token_mask[example_i, idxs, :] = 1.
        return token_mask

    def pool_fn(self, masked_hidden, token_mask):
        raise NotImplementedError()


@register_entity_pooler("max")
class MaxEntityPooler(BaseEntityPooler):

    def __init__(self, insize, outsize):
        super().__init__(insize, outsize)

    def pool_fn(self, masked_hidden, token_mask):
        # Replace masked with -inf to avoid zeroing out
        # hidden dimensions if the non-masked values are all negative.
        masked_hidden[torch.logical_not(token_mask)] = -torch.inf
        pooled = torch.max(masked_hidden, axis=1)[0]
        # Just in case all values are -inf
        pooled = torch.nan_to_num(pooled)
        return pooled
