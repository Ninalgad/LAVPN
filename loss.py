import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class DiceBCELoss(nn.Module):
    def __init__(self, from_logits=False):
        super(DiceBCELoss, self).__init__()
        self.from_logits = from_logits

    def forward(self, inputs, targets, smooth=0.1):
        if self.from_logits:
            inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return bce_loss + dice_loss


class FocalLoss(nn.Module):
    def __init__(self, from_logits=False):
        super(FocalLoss, self).__init__()
        self.from_logits = from_logits

    def forward(self, inputs, targets, alpha=0.8, gamma=2):
        if self.from_logits:
            inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        bce_loss_exp = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - bce_loss_exp) ** gamma * bce_loss

        return focal_loss


def loss_func(outputs, tar):
    return DiceBCELoss(from_logits=True)(outputs, tar) + \
           3 * FocalLoss(from_logits=True)(outputs, tar)
