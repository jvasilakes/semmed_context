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
        # and project it down, if the outsize is different.
        if self.outsize == self.insize:
            projected = all_pooled
        else:
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


class BaseAttentionEntityPooler(BaseEntityPooler):

    def __init__(self, insize, outsize):
        super().__init__(insize, outsize)
        self.alignment_model = torch.nn.Linear(2 * insize, 1)

    def forward(self, hidden, token_idxs, pooled_entities):
        """
        token_idxs = [F, B, T, S]
         F: entity functions, e.g., subject, object
         B: batch
         E: entity span
         S: start/end index of the entity
        """
        assert token_idxs.dim() == 4, "token_idxs = [F, B, T, S]"
        all_pooled = []
        all_attentions = []
        # token_idxs is grouped by entity function.
        # E.g., all subjects in a batch, then all objects.
        for entity_idxs in token_idxs:
            # So we create a batched entity mask for each function
            entity_mask = self.get_token_mask_from_indices(
                entity_idxs, hidden.size())
            entity_mask = entity_mask.to(hidden.device)
            masked_hidden = hidden * entity_mask
            pooled, attentions = self.pool_fn(
                masked_hidden, entity_mask, pooled_entities)
            all_pooled.append(pooled)
            all_attentions.append(attentions)
        # Then we concatenate the entity representations for each
        # entity function.
        all_pooled = torch.cat(all_pooled, dim=1)
        # and project it down.
        projected = self.output_layer(all_pooled)
        return projected, all_attentions

    def pool_fn(self, masked_hidden, entity_mask, pooled_entities):
        """
        projection_fn = some.projection_fn
        return self.generic_attention_pooler(
            masked_hidden, entity_mask, pooled_entities, projection_fn)
        """
        raise NotImplementedError()

    def generic_attention_pooler(self, masked_hidden, entity_mask,
                                 pooled_entities, projection_fn):
        """
        projection_fn: A function that maps scores to the simplex,
            e.g. softmax.
        """
        entity_repeated = pooled_entities.unsqueeze(1).repeat(
            1, masked_hidden.size(1), 1)
        entity_repeated_masked = entity_repeated * entity_mask
        # Compute alignments between the pooled_entities and each
        # hidden token representation.
        alignment_inputs = torch.cat((entity_repeated_masked, masked_hidden),
                                     dim=2)
        attention_scores = self.alignment_model(alignment_inputs)
        attention_mask = entity_mask[:, :, 0].bool()
        # Project the attention scores for each example.
        # We need to do it example by example because each has a different
        # token mask.
        attention_probs = torch.zeros_like(attention_scores)
        batch_size = masked_hidden.size(0)
        for example_idx in range(batch_size):
            masked_scores = torch.masked_select(
                attention_scores[example_idx],
                attention_mask[example_idx].unsqueeze(1))
            if masked_scores.nelement() == 0:
                probs = torch.empty_like(masked_scores)
            else:
                probs = projection_fn(masked_scores)
            prob_idxs = attention_mask[example_idx]
            attention_probs[example_idx][prob_idxs] = probs.unsqueeze(1)
        # Scale the token representations by their attention probabilities
        # and sum over the token dimension to obtain the weighted average.
        pooled = (masked_hidden * attention_probs).sum(1)
        return pooled, attention_probs


@register_entity_pooler("attention-softmax")
class SoftmaxAttentionEntityPooler(BaseAttentionEntityPooler):

    def pool_fn(self, masked_hidden, entity_mask, pooled_entities):
        projection_fn = torch.nn.Softmax(dim=0)
        return self.generic_attention_pooler(
            masked_hidden, entity_mask, pooled_entities, projection_fn)


@register_entity_pooler("attention-sparsegen")
class SparsegenAttentionEntityPooler(BaseAttentionEntityPooler):

    def pool_fn(self, masked_hidden, entity_mask, pooled_entities):
        projection_fn = SparsegenLin(dim=0, lam=0.5)
        return self.generic_attention_pooler(
            masked_hidden, entity_mask, pooled_entities, projection_fn)


class SparsegenLin(torch.nn.Module):
    """
    Generic sparsegen-lin function as described in

    Laha, A., Chemmengath, S. A., Agrawal, P., Khapra, M.,
    Sankaranarayanan, K., & Ramaswamy, H. G. (2018).
    On Controllable Sparse Alternatives to Softmax.
    Advances in Neural Information Processing Systems, 31.
    https://proceedings.neurips.cc/paper/2018/hash/6a4d5952d4c018a1c1af9fa590a10dda-Abstract.html  # noqa

    Implementation modified from
    https://github.com/KrisKorrel/sparsemax-pytorch

    As lam --> 1, output approaches one-hot vector.
    As lam --> -inf, output approaches uniform.
    """

    def __init__(self, dim=None, lam=0.0):
        """
        Args:
            dim (int, optional): The dimension over which to apply
                                 the sparsegen function.
            lam (float): The lambda parameter. Default 0.0.
        """
        super().__init__()

        self.dim = -1 if dim is None else dim
        assert lam < 1
        self.lam = lam

    def __str__(self):
        return f"SparsegenLin(dim={self.dim}, lam={self.lam})"

    def __repr__(self):
        return f"SparsegenLin(dim={self.dim}, lam={self.lam})"

    def forward(self, inputs):
        """Forward function.
        Args:
            inputs (torch.Tensor): Input tensor. First dimension is batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        inputs = inputs.transpose(0, self.dim)
        original_size = inputs.size()
        inputs = inputs.reshape(inputs.size(0), -1)
        inputs = inputs.transpose(0, 1)
        dim = 1

        number_of_logits = inputs.size(dim)

        # Translate inputs by max for numerical stability
        inputs = inputs - torch.max(inputs, dim=dim, keepdim=True)[0].expand_as(inputs)  # noqa

        # Sort inputs in descending order.
        # (NOTE: Can be replaced with linear time selection method:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html
        zs = torch.sort(input=inputs, dim=dim, descending=True)[0]
        ks = torch.arange(start=1, end=number_of_logits + 1, step=1,
                          dtype=inputs.dtype).view(1, -1)
        ks = ks.expand_as(zs).to(zs.device)

        # Determine sparsity of projection
        bound = 1 - self.lam + ks * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(inputs.type())
        k = torch.max(is_gt * ks, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1 + self.lam) / k
        taus = taus.expand_as(inputs)

        # Sparsemax
        ps = (inputs - taus) / (1 - self.lam)
        self.output = torch.max(torch.zeros_like(inputs), ps)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        grad_sum = torch.sum(grad_output * nonzeros, dim=dim)
        grad_sum /= torch.sum(nonzeros, dim=dim)
        self.grad_inputs = nonzeros * (grad_output - grad_sum.expand_as(grad_output))  # noqa

        return self.grad_inputs
