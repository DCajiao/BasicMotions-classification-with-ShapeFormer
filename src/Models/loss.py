import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_module():
    """
    Returns a non-reduced instance of NoFussCrossEntropyLoss.

    This function instantiates a modified cross-entropy loss that avoids internal
    type constraints and returns individual loss values per sample.

    Returns:
        nn.Module: Instance of NoFussCrossEntropyLoss with 'none' reduction.
    """
    return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample


def l2_reg_loss(model):
    """
    Computes the L2 regularization term for the model's output layer.

    Specifically, this function returns the squared L2 norm of the weights
    from the final output layer, if found.

    Args:
        model (nn.Module): PyTorch model with named parameters.

    Returns:
        torch.Tensor or None: L2 norm of the output layer's weights, or None if not found.
    """

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    A relaxed version of CrossEntropyLoss that ensures compatibility with input types.

    This subclass of PyTorch's CrossEntropyLoss automatically converts the target
    tensor to `LongTensor` type and bypasses the strict 1D shape constraint.

    Useful for custom training loops where input formats may vary.
    """

    def forward(self, inp, target):
        """
        Computes cross-entropy loss between predictions and targets.

        Args:
            inp (Tensor): Predicted logits of shape (B, C), where C is the number of classes.
            target (Tensor): Target labels of any compatible type and shape.

        Returns:
            Tensor: Element-wise loss values according to the configured reduction.
        """
        return F.cross_entropy(inp, target.long(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
