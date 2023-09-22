import torch
import numpy as np
from torchtext.data.metrics import bleu_score

from .utils import CLUB, CLUBSample, sequence_sparse_softmax_cross_entropy  # noqa


def compute_bleu(Xbatch, pred_batch, tokenizer):
    Xtext = [[tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)]
             for ids in Xbatch.cpu().detach()]
    pred_text = [tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
                 for ids in pred_batch.cpu().detach()]
    bleu = bleu_score(pred_text, Xtext)
    return bleu


def reconstruction_loss(targets, logits, target_lengths):
    recon_loss = sequence_sparse_softmax_cross_entropy(
            labels=targets, logits=logits, sequence_length=target_lengths)
    return {"reconstruction_loss": recon_loss}


def get_cyclic_kl_weight(step, total_steps, cycles=4, rate=0.5):
    denom = total_steps / cycles
    numer = step % np.ceil(denom)
    tau = numer / denom
    if tau <= rate:
        return tau / rate
    else:
        return 1


def kl_divergence(mu, logvar):
    kl = 0.5 * (torch.exp(logvar) + torch.pow(mu, 2) - 1 - logvar)
    kl = kl.mean(0).sum()
    return kl


def compute_kl_divergence_losses(model, latent_params, kl_weights_dict):
    # KL for each latent space
    idv_kls = dict()
    # total kl over all latent spaces
    total_kl = 0.0  # scalar for logging
    # tensor scalar for backward pass
    total_weighted_kl = torch.tensor(0.0).to(model.device)
    for (latent_name, latent_params) in latent_params.items():
        kl = kl_divergence(latent_params.mu, latent_params.logvar)
        idv_kls[latent_name] = kl.item()
        total_kl += kl.item()
        try:
            weight = kl_weights_dict[latent_name]
        except KeyError:
            weight = kl_weights_dict["default"]
        total_weighted_kl += weight * kl
    return {"total_weighted_kl": total_weighted_kl,
            "total_kl": total_kl,
            "idv_kls": idv_kls}


def compute_discriminator_losses(model, discriminator_logits, Ybatch):
    # Loss and accuracy for each discriminator
    idv_dsc_losses = dict()
    idv_dsc_accs = dict()
    # total loss over all discriminators
    total_dsc_loss = torch.tensor(0.0).to(model.device)
    for (dsc_name, dsc_logits) in discriminator_logits.items():
        dsc = model.discriminators[dsc_name]
        targets = Ybatch[dsc_name].to(model.device)
        dsc_loss = dsc.compute_loss(dsc_logits, targets)
        dsc_acc = dsc.compute_accuracy(dsc_logits, targets)
        idv_dsc_losses[dsc_name] = dsc_loss.item()
        idv_dsc_accs[dsc_name] = dsc_acc.item()
        total_dsc_loss += dsc_loss
    return {"total_dsc_loss": total_dsc_loss,
            "idv_dsc_losses": idv_dsc_losses,
            "idv_dsc_accs": idv_dsc_accs}


def compute_adversarial_losses(model, adversary_logits, Ybatch):
    # Adversarial loss for each individual adversary
    idv_adv_losses = dict()
    # Discriminator loss for each individual adversary
    idv_dsc_losses = dict()
    # Accuracies of the discriminators
    idv_dsc_accs = dict()
    # total loss over all adversarial discriminators
    total_adv_loss = torch.tensor(0.0).to(model.device)
    for (adv_name, adv_logits) in adversary_logits.items():
        adv = model.adversaries[adv_name]
        latent_name, label_name = adv_name.split('-')
        targets = Ybatch[label_name].to(model.device)
        adv_loss = adv.compute_adversarial_loss(adv_logits)
        idv_adv_losses[adv_name] = adv_loss.item()
        total_adv_loss += adv_loss
        # This will be used to update the adversaries
        dsc_loss = adv.compute_discriminator_loss(adv_logits, targets)
        idv_dsc_losses[adv_name] = dsc_loss
        dsc_acc = adv.compute_accuracy(adv_logits, targets)
        idv_dsc_accs[adv_name] = dsc_acc.item()
    return {"total_adv_loss": total_adv_loss,
            "idv_adv_losses": idv_adv_losses,
            "idv_adv_dsc_losses": idv_dsc_losses,
            "idv_adv_dsc_accs": idv_dsc_accs}


def compute_mi_losses(model, latent_params, beta=1.0):
    idv_mi_estimates = dict()
    total_mi = torch.tensor(0.0).to(model.device)
    for (latent_name_1, params1) in latent_params.items():
        for (latent_name_2, params2) in latent_params.items():
            if latent_name_1 == latent_name_2:
                continue
            try:
                name = f"{latent_name_1}-{latent_name_2}"
                mi_estimator = model.mi_estimators[name]
            except KeyError:
                continue
            mi_estimate = mi_estimator(params1.z, params2.z) * beta
            idv_mi_estimates[name] = mi_estimate.item()
            total_mi += mi_estimate
    return {"total_mi": total_mi,
            "idv_mi_estimates": idv_mi_estimates}
