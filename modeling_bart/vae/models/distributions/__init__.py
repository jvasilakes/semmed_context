import torch
import torch.distributions as D


import sle
from ..util import register_distribution
from .helpers import Kumaraswamy, StretchedAndRectifiedDistribution


@register_distribution("normal")
class Normal(D.Normal):
    num_params = 2
    discrete = False

    @classmethod
    def from_bunched_params(cls, params: torch.FloatTensor, **kwargs):
        loc, scale = params.chunk(2, dim=-1)
        scale = torch.exp(scale)
        return cls(loc, scale)

    def __init__(self, loc, scale):
        self.parameters = {"loc": loc, "scale": scale}
        super().__init__(loc, scale)

    def kl_loss(self):
        standard_dist = D.Normal(0, 1)
        return D.kl_divergence(self, standard_dist).mean(0).sum()

    def predict(self):
        if self.loc.dim() == 1:
            return (self.rsample().sigmoid() >= 0.5).long()
        else:
            return self.rsample().argmax(1)


@register_distribution("hardkuma")
class HardKumaraswamy(StretchedAndRectifiedDistribution):
    """
    Approximately binomial.
    """

    num_params = 2
    discrete = True
    arg_constraints = {}

    @classmethod
    def from_bunched_params(cls, params: torch.FloatTensor, **kwargs):
        a, b = params.chunk(2, dim=-1)
        return cls(a, b)

    def __init__(self, a, b):
        self.a = torch.exp(a)
        self.b = torch.exp(b)
        self.parameters = {'a': self.a, 'b': self.b}
        self.kuma = Kumaraswamy(self.a, self.b)
        super().__init__(self.kuma)

    def rsample(self, size=None):
        prob_sample = super().rsample()
        x_ = torch.clamp(prob_sample, 1e-5, 1-1e-5)
        logit_sample = torch.log(x_ / (1 - x_))
        return logit_sample

    def kl_loss(self):
        """
        KL divergence between this distribution and the uniform.
        """
        unif = D.Uniform(0, 1)
        constant = - torch.log(
            unif.cdf(torch.tensor(1)) - unif.cdf(torch.tensor(0)))
        kl = -(self.kuma.entropy() - constant)
        return kl.mean(0).sum()

    def predict(self):
        return (self.rsample() >= 0.5).long()


@register_distribution("gumbel-softmax")
class GumbelSoftmax(object):
    """
    Approximately multinomial.

    Adapted from https://github.com/Schlumberger/joint-vae/blob/master/jointvae/models.py  # noqa
    """
    num_params = 1
    discrete = True
    eps = 1e-12
    temperature = 1.0  # Closer to 0 makes the distribution sparse.

    @classmethod
    def from_bunched_params(cls, params: torch.FloatTensor, **kwargs):
        alphas = params.exp()
        return cls(alphas)

    def __init__(self, alphas):
        self.parameters = {"alphas": self.alphas}

    def rsample(self, size=None):
        if size is not None:
            size = torch.Size(list(size) + list(self.alphas.size()))
        else:
            size = self.alphas.size()
        unif = torch.rand(size)
        unif = unif.to(self.alphas.device)
        gumbel = -torch.log(-torch.log(unif + self.eps) + self.eps)
        log_alphas = torch.log(self.alphas + self.eps)
        logits = (log_alphas + gumbel) / self.temperature
        return torch.nn.functional.log_softmax(logits, dim=-1)

    def sample(self, size):
        return self.rsample(size)

    def kl_loss(self):
        """
        KL divergence between this distribution and the uniform.
        """
        log_dim = torch.log(torch.tensor(self.alphas.size(-1)))
        neg_entropy = self.alphas * torch.log(self.alphas + self.eps)
        kl_divergence = log_dim + neg_entropy
        # Scale kl by number of dims otherwise it takes over and
        # suppresses the content latent.
        return kl_divergence.mean(0).sum() / self.alphas.size(-1)

    def predict(self):
        return self.rsample().argmax(1)


@register_distribution("sle-beta")
class SLEBeta(sle.SLBeta):
    """
    Beta distribution reparameterized to use (b)elief, (d)isbelief,
    and (u)ncertainty parameters as described in Subjective Logic.
    """
    num_params = 3
    discrete = True

    @classmethod
    def from_bunched_params(cls, params: torch.FloatTensor,
                            softmax=True, **kwargs):
        if softmax is True:
            params = params.softmax(-1)
        # softmax() to ensure params are positive and b + d + u = 1
        b, d, u = params.chunk(3, dim=-1)
        return cls(b, d, u)

    def __init__(self, b, d, u):
        self.parameters = {'b': b, 'd': d, 'u': u}
        super().__init__(b, d, u)

    def kl_loss(self):
        standard_dist = sle.SLBeta(0.0, 0.0, 1.0).to(self.b.device)
        return D.kl_divergence(self, standard_dist).mean(0).sum()

    def predict(self):
        return (self.rsample() >= 0.5).long()


@register_distribution("sle-dirichlet")
class SLEDirichlet(sle.SLDirichlet):
    """
    Dirichlet distribution reparameterized to use (b)elief and (u)ncertainty
    parameters as described in Subjective Logic.
    """
    num_params = 1
    discrete = True

    @classmethod
    def from_bunched_params(cls, params: torch.FloatTensor,
                            softmax=True, **kwargs):
        if softmax is True:
            params = params.softmax(-1)
        b, u = params.tensor_split([-1], dim=-1)
        return cls(b, u)

    def __init__(self, b, u):
        self.parameters = {'b': b, 'u': u}
        super().__init__(b, u)

    def kl_loss(self):
        loc = torch.zeros_like(self.b)
        scale = torch.ones((loc.size(0),)).to(loc.device)
        standard_dist = sle.SLDirichlet(loc, scale)
        return D.kl_divergence(self, standard_dist).mean(0).sum()

    def predict(self):
        return self.rsample().argmax(1)
