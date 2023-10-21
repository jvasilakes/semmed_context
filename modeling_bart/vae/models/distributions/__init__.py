import torch
import torch.distributions as D


from ..util import register_distribution
from .helpers import Kumaraswamy, StretchedAndRectifiedDistribution


@register_distribution("normal")
class Normal(D.Normal):
    num_params = 2
    discrete = False

    def __init__(self, params: torch.FloatTensor, **kwargs):
        mu, logvar = params.chunk(2, dim=-1)
        var = torch.exp(logvar)
        self.parameters = {"mu": mu, "var": var}
        super().__init__(mu, var)

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
