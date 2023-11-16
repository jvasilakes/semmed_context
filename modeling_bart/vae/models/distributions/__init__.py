import torch
import torch.distributions as D


import sle
from ..util import register_distribution
from .helpers import Kumaraswamy, StretchedAndRectifiedDistribution


@register_distribution("normal")
class Normal(D.Normal):
    num_params = 2
    discrete = False

    def __init__(self, params: torch.FloatTensor, **kwargs):
        loc, scale = params.chunk(2, dim=-1)
        scale = torch.exp(scale)
        self.parameters = {"loc": loc, "scale": scale}
        super().__init__(loc, scale)

    def kl_loss(self):
        standard_dist = D.Normal(0, 1)
        return D.kl_divergence(self, standard_dist).mean(0).sum()


@register_distribution("hardkuma")
class HardKumaraswamy(StretchedAndRectifiedDistribution):
    """
    Approximately binomial.
    """

    num_params = 2
    discrete = True
    arg_constraints = {}

    def __init__(self, params: torch.FloatTensor, **kwargs):
        a, b = params.chunk(2, dim=-1)
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

    def __init__(self, params: torch.FloatTensor):
        self.alphas = params.exp()
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


@register_distribution("sle-beta")
class SLBeta(sle.SLBeta):
    """
    Beta distribution reparameterized to use (b)elief, (d)isbelief,
    and (u)ncertainty parameters as described in Subjective Logic.
    """
    num_params = 3
    discrete = True

    def __init__(self, params: torch.FloatTensor, **kwargs):
        # softmax() to ensure params are positive and b + d + u = 1
        b, d, u = params.softmax(-1).chunk(3, dim=-1)
        self.parameters = {'b': b, 'd': d, 'u': u}
        super().__init__(b, d, u)

    def kl_loss(self):
        standard_dist = sle.SLBeta(0.0, 0.0, 1.0)
        return D.kl_divergence(self, standard_dist).mean(0).sum()


@register_distribution("sle-dirichlet")
class SLDirichlet(sle.SLDirichlet):
    """
    Dirichlet distribution reparameterized to use (b)elief and (u)ncertainty
    parameters as described in Subjective Logic.
    """
    num_params = 1
    discrete = True

    def __init__(self, params: torch.FloatTensor, **kwargs):
        b, u = params.softmax(-1).tensor_split([-1], dim=-1)
        self.parameters = {'b': b, 'u': u}
        super().__init__(b, u)

    def kl_loss(self):
        standard_dist = sle.SLDirichlet(torch.zeros_like(self.b), 1.0)
        return D.kl_divergence(self, standard_dist).mean(0).sum()
