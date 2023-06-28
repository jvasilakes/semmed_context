import torch
import torchvision


from .util import register_loss


register_loss("cross_entropy")(torch.nn.CrossEntropyLoss)


@register_loss("focal_loss")
class InverseFocalLoss(torch.nn.Module):

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1 - targets because this loss assumes 0 is the minority class
        return torchvision.ops.sigmoid_focal_loss(
                inputs[:, 0], 1 - targets, reduction=self.reduction)
