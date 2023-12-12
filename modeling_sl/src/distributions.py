import torch
import torch.distributions as D

import sle
import sle.plotting as plotting


class SLBeta(D.Beta):
    """
    Beta distribution reparameterized to used belief, disbelief,
    and uncertainty parameters in accordance with Subjective Logic.

    b + d + u must equal 1

    Args:
        b (Tensor): belief parameters.
        d (Tensor): disbelief parameters.
        u (Tensor): uncertainty parameters.
        a (None or Tensor): prior probabilities (default Uniform)
        W (int): prior weight (default 2)

    Example:

        >>> params = torch.softmax(torch.randn(3, 3, 1), dim=1)  # 3 examples and 3 parameters  # noqa
        >>> b = params[:, 0]
        >>> d = params[:, 1]
        >>> u = params[:, 2]
        >>> d = SLBeta(b, d, u)

    See help(torch.distribution.Beta) for more information.
    """
    def __init__(self, b, d, u, a=None, W=2):
        # First, convert to tensors
        b = torch.as_tensor(b)
        d = torch.as_tensor(d)
        u = torch.as_tensor(u)
        W = torch.as_tensor(W)
        # If prior not specified, use Uniform
        if a is None:
            a = 0.5
        a = torch.as_tensor(a)

        total = b + d + u
        assert torch.isclose(total, torch.ones_like(total)).all(), str(b)
        self.b = b
        self.d = d
        eps = torch.tensor(sle.EPS, device=u.device)
        self.u = torch.where(torch.isclose(u, torch.tensor(0.)),
                             eps, u)
        self.a = a
        self.W = W
        c0, c1 = self.reparameterize()
        super().__init__(c0, c1)

    def __repr__(self):
        b = self.b.detach()
        d = self.d.detach()
        u = self.u.detach()
        return f"SLBeta({b}, {d}, {u})"

    def __str__(self):
        b = self.b.detach()
        d = self.d.detach()
        u = self.u.detach()
        return f"SLBeta({b}, {d}, {u})"

    @property
    def device(self):
        return self.b.device

    def to(self, device):
        b = self.b.to(device)
        d = self.d.to(device)
        u = self.u.to(device)
        a = self.a.to(device)
        W = self.W.to(device)
        return self.__class__(b, d, u, a, W)

    def parameters(self):
        return {'b': self.b, 'd': self.d, 'u': self.u}

    def reparameterize(self):
        b = self.b
        d = self.d
        u = self.u
        a = self.a
        W = self.W
        c0 = ((W * b) / u) + (W * a)
        c1 = ((W * d) / u) + (W * (1 - a))
        return (c0, c1)

    def max_uncertainty(self):
        """
        Uncertainty maximization operation.

        SLT Book Eq. 3.27: the uncertainty maximum is
          \\hat{u}_x = min_i[ p(x_i) / a(x_i) ]

        Then solve P = b + au for b, given we know P, a, and u.
        """
        # Projected probabilities
        # p = self.b + self.a * self.u
        p = self.mean

        new_b = torch.zeros_like(self.b)
        new_d = torch.zeros_like(self.d)
        new_u = torch.zeros_like(self.u)

        neg_idxs = p < 0.5
        new_u[neg_idxs] = (p / self.a)[neg_idxs]
        new_d[neg_idxs] = 1. - new_u[neg_idxs]
        new_b[neg_idxs] = torch.tensor(0.)

        pos_idxs = ~neg_idxs
        new_u[pos_idxs] = ((1. - p) / (1. - self.a))[pos_idxs]
        new_b[pos_idxs] = 1. - new_u[pos_idxs]
        new_d[pos_idxs] = torch.tensor(0.)
        return self.__class__(new_b, new_d, new_u, a=self.a, W=self.W)

    def cumulative_fusion(self, *others):
        """
        Aleatory Cumulative fusion operator. See SLT Book section 12.3.
        For Epistemic Cumulative fusion do

        Beta1.cumulative_fusion(Beta2).max_uncertainty()

        Multiple cumulative fusions can be chained like so


        ```
        sle.fuse(Beta1, Beta2, ...,  BetaN)
        ```
        or

        ```
        Beta1.fuse(Beta2, ..., BetaN)
        ```
        """
        return sle.fuse([self] + list(others))

    def trust_discount(self, trust_level=None, trust_opinion=None):
        """
        Trust discounting operator. See SLT Book section 14.3.2.
        """
        if trust_level is not None:
            if trust_level < 0. or trust_level > 1.:
                raise ValueError(f"trust_level must be in [0,1]. Got {trust_level}")  # noqa
            trust_opinion = SLBeta(trust_level, 1. - trust_level, 0.).max_uncertainty()  # noqa

        elif trust_opinion is not None:
            if not isinstance(trust_opinion, SLBeta):
                raise ValueError(f"trust_opinion argument must be of type SLBeta. Got {type(trust_opinion)}.")  # noqa

        new_b = trust_opinion.mean * self.b
        new_d = trust_opinion.mean * self.d
        new_u = 1 - (trust_opinion.mean * (self.b + self.d))
        return self.__class__(new_b, new_d, new_u, a=self.a, W=self.W)

    def get_uniform(self):
        """
        Create a uniform SLBeta distribution.
        """
        b = torch.zeros_like(self.b)
        d = torch.zeros_like(self.d)
        u = torch.ones_like(self.u)
        return self.__class__(b, d, u)

    def plot(self):
        plotting.plot_beta(self, title=str(self)).show()


class SLDirichlet(D.Dirichlet):
    """
    Dirichlet distribution reparameterized to used belief and uncertainty
    parameters in accordance with Subjective Logic.

    b.sum(dim=1) + u must equal 1.

    Args:
        b (Tensor): belief parameters.
        u (Tensor): uncertainty parameters.
        a (None or Tensor): prior probabilities (default Uniform)
        W (int): prior weight (default 2)

    Example:

        >>> params = torch.softmax(torch.randn(2, 4, 1), dim=1)  # 2 examples with 3 labels + uncertainty  # noqa
        >>> b = params[:, :2]
        >>> u = params[:, 3]
        >>> d = SLDirichlet(b, u)

    See help(torch.distribution.Dirichlet) for more information.
    """

    def __init__(self, b, u, a=None, W=2):
        b = torch.as_tensor(b, dtype=torch.float32)
        u = torch.as_tensor(u, dtype=torch.float32)
        if u.dim() < b.dim():
            u = u.unsqueeze(0)
        assert u.dim() == b.dim()
        W = torch.as_tensor(W)
        total = b.sum(dim=-1, keepdim=True) + u
        assert torch.isclose(total, torch.ones_like(total)).all(), str(b)

        self.b = b
        eps = torch.tensor(sle.EPS, device=u.device)
        self.u = torch.where(torch.isclose(u, torch.tensor(0.)),
                             eps, u)

        # If prior not specified, use Uniform
        if a is None:
            a = torch.zeros_like(b).fill_(1 / self.b.size(-1))
        self.a = torch.as_tensor(a, dtype=torch.float32)
        assert self.a.shape == self.b.shape
        self.W = torch.as_tensor(W, dtype=torch.float32)
        alphas = self.reparameterize()
        super().__init__(alphas)

    def __repr__(self):
        b = self.b.detach()
        u = self.u.detach()
        return f"SLDirichlet({b}, {u})"

    def __str__(self):
        b = self.b.detach()
        u = self.u.detach()
        return f"SLDirichlet({b}, {u})"

    def __eq__(self, other):
        return all([
            torch.isclose(self.b, other.b).all(),
            torch.isclose(self.u, other.u).all(),
            torch.isclose(self.a, other.a).all(),
            torch.isclose(self.W, other.W)]
        )

    @property
    def device(self):
        return self.b.device

    def to(self, device):
        b = self.b.to(device)
        u = self.u.to(device)
        a = self.a.to(device)
        W = self.W.to(device)
        return self.__class__(b, u, a, W)

    def parameters(self):
        return {'b': self.b, 'u': self.u}

    def reparameterize(self):
        alphas = ((self.W * self.b) / self.u) + (self.a * self.W)
        return alphas

    def max_uncertainty(self):
        """
        Uncertainty maximization operation.

        SLT Book Eq. 3.27: the uncertainty maximum is
          \\hat{u}_x = min_i[ p(x_i) / a(x_i) ]

        Then solve P = b + au for b, given we know P, a, and u.
        """
        ps = self.mean
        us = ps / self.a
        vals, idxs = us.sort(dim=-1)
        new_u, _ = vals.tensor_split([1], dim=-1)
        new_b = ps - self.a * new_u
        return self.__class__(new_b, new_u, a=self.a, W=self.W)

    def cumulative_fusion(self, *others):
        """
        Aleatory Cumulative fusion operator. See SLT Book 12.3
        For Epistemic Cumulative fusion do

        Beta1.cumulative_fusion(Beta2).max_uncertainty()
        """
        return sle.fuse([self] + list(others))

    def trust_discount(self, trust_level=None, trust_opinion=None):
        if trust_level is not None:
            if trust_level < 0. or trust_level > 1.:
                raise ValueError(f"trust_level must be in [0,1]. Got {trust_level}")  # noqa
            trust_opinion = SLBeta(trust_level, 1. - trust_level, 0.).max_uncertainty()  # noqa

        elif trust_opinion is not None:
            if not isinstance(trust_opinion, SLBeta):
                raise ValueError(f"trust_opinion argument must be of type SLBeta. Got {type(trust_opinion)}.")  # noqa

        new_b = trust_opinion.mean * self.b
        new_u = 1 - (trust_opinion.mean * (self.b.sum()))
        return self.__class__(new_b, new_u, a=self.a, W=self.W)

    def get_uniform(self):
        """
        Create a uniform SLDirichlet distribution.
        """
        b = torch.zeros_like(self.b)
        u = torch.ones_like(self.u)
        return self.__class__(b, u)

    def plot(self):
        plotting.plot_dirichlet(self, title=str(self)).show()


def fuse(dists):

    def compute_denominator(u1, u2):
        denom = (u1 + u2) - (u1 * u2)
        return denom

    def compute_b(b1, b2, u1, u2, denom):
        b_num = (b1 * u2) + (b2 * u1)
        b = b_num / denom
        return b

    def compute_u(u1, u2, denom):
        u_num = u1 * u2
        u = u_num / denom
        return u

    def compute_a(a1, a2, u1, u2):
        a_num1 = (a1 * u2) + (a2 * u1)
        a_num2 = (a1 + a2) * (u1 * u2)
        a_denom = (u1 + u2) - (2 * u1 * u2)
        a = (a_num1 - a_num2) / a_denom
        return a

    dist_cls = dists[0].__class__
    assert dist_cls in [SLBeta, SLDirichlet], "Can only fuse SLBeta or SLDirichlet"  # noqa
    assert all([isinstance(d, dist_cls) for d in dists]), "Found both SLBeta and SLDirichlet!"  # noqa

    if len(dists) == 1:
        return dists[0]

    b, u, a = dists[0].b, dists[0].u, dists[0].a
    for dist in dists[1:]:
        denom = compute_denominator(u, dist.u)
        b = compute_b(b, dist.b, u, dist.u, denom)
        u = compute_u(u, dist.u, denom)
        a = compute_a(a, dist.a, u, dist.u)

    args = [b, u, a]
    if dist_cls == SLBeta:
        # To account for precision errors
        if torch.isclose(b, torch.tensor(1.0)):
            d = 0.0
        else:
            d = 1. - (b + u)
        args = [b, d, u, a]
    return dist_cls(*args)
