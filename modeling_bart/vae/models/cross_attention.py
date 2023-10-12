import torch
from torch import nn
from einops import rearrange
from torch import einsum


def stable_softmax(t, dim=-1):
    t = t - t.amax(dim=dim, keepdim=True)
    return t.softmax(dim=dim)


class CrossAttention(nn.Module):

    """
    Modified from
    https://github.com/lucidrains/bidirectional-cross-attention/
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        context_dim=None,
        dropout=0.0,
        talking_heads=False,
        prenorm=False,
        sparse=False,
    ):
        super().__init__()
        context_dim = context_dim or dim

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()  # noqa

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()  # noqa
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()  # noqa

        self.probs = stable_softmax
        if sparse is True:
            self.probs = SparsegenLin(lam=0.5)

    def forward(
        self,
        x,
        context,
        mask=None,
        context_mask=None,
        return_attn=False,
        rel_pos_bias=None
    ):
        b, i, j = x.shape[0], x.shape[-2], context.shape[-2]
        h = self.heads
        device = x.device

        x = self.norm(x)
        context = self.context_norm(context)

        # get shared query/keys and values for sequence and context
        qk, v = self.to_qk(x), self.to_v(x)
        context_qk = self.context_to_qk(context)
        context_v = self.context_to_v(context)

        # split out head
        qk, context_qk, v, context_v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (qk, context_qk, v, context_v))

        # get similarities
        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale  # noqa

        # relative positional bias, if supplied
        if rel_pos_bias is not None:
            sim = sim + rel_pos_bias

        if mask is not None or context_mask is not None:
            mask = mask or torch.ones(
                (b, i), device=device, dtype=torch.bool)
            context_mask = context_mask or torch.ones(
                (b, j), device=device, dtype=torch.bool)
            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')  # noqa
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # get attention along both sequence and context length dimensions
        # shared similarity matrix
        attn = self.probs(sim, dim=-1)
        context_attn = self.probs(sim, dim=-2)

        # dropouts
        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)

        # talking heads
        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)

        # src sequence aggregates values from context, context aggregates
        # values from src sequence
        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, v)

        # merge heads and combine out
        out, context_out = map(lambda t: rearrange(
            t, 'b h n d -> b n (h d)'), (out, context_out))

        out = self.to_out(out)
        context_out = self.context_to_out(context_out)

        if return_attn:
            return out, context_out, attn, context_attn

        return out, context_out


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

        assert lam < 1
        self.lam = lam

    def __str__(self):
        return f"SparsegenLin(lam={self.lam})"

    def __repr__(self):
        return f"SparsegenLin(lam={self.lam})"

    def forward(self, inputs, dim=-1):
        """Forward function.
        Args:
            inputs (torch.Tensor): Input tensor. First dimension is batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        inputs = inputs.transpose(0, dim)
        original_size = inputs.size()
        inputs = inputs.reshape(inputs.size(0), -1)
        inputs = inputs.transpose(0, 1)
        orig_dim = dim
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
        output = output.transpose(0, orig_dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        grad_sum = torch.sum(grad_output * nonzeros, dim=dim)
        grad_sum /= torch.sum(nonzeros, dim=dim)
        self.grad_inputs = nonzeros * (grad_output - grad_sum.expand_as(grad_output))  # noqa

        return self.grad_inputs
