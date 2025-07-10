# -*- coding: utf-8 -*-

import torch
from torch import nn

from torchvision.models import resnet34, resnet18, vgg13, vgg11, vgg19


__all__ = [
    "generate_model",
    "CassavaModel",
]


PRETRAINED_FUNCTION_MAP = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "vgg13": vgg13,
    "vgg11": vgg11,
    "vgg19": vgg19,
}


def generate_model(out_features, freeze_weights = True, pretrained_model_function = "resnet18"):
    pretrained_model_function = PRETRAINED_FUNCTION_MAP[pretrained_model_function]
    base_model = pretrained_model_function(weights = True)

    if freeze_weights:
        for param in base_model.parameters():
            param.requires_grad_(False)

    try:
        in_features = base_model.fc.in_features
        new_layer = nn.Linear(in_features = in_features, out_features = out_features)
        base_model.fc = new_layer
    except:
        in_features = base_model.classifier[0].in_features
        new_layer = nn.Linear(in_features = in_features, out_features = out_features)
        base_model.classifier = new_layer

    return base_model


class CassavaModel(nn.Module):
    def __init__(self, out_features, freeze_weights = True, pretrained_model_function = "resnet18"):
        super(CassavaModel, self).__init__()

        self.base = generate_model(
            out_features = out_features,
            freeze_weights = freeze_weights,
            pretrained_model_function = pretrained_model_function
        )

    def forward(self, x):
        x = self.base(x)
        return torch.log_softmax(x, dim = -1)
