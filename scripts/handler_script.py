# -*- coding: utf-8 -*-

"""Handler script for inference."""

import torch
from ts.torch_handler.image_classifier import ImageClassifier

from torchvision import transforms as T
from imageio.v3 import imread

__all__ = [
    "CassavaClassifier"
]


class CassavaClassifier(ImageClassifier):
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.Normalize(mean = [.485, .456, .406], std = [.229, .224, .225])
        ]
    )

    class_map = {
        "cbsd": 0,
        "cbb": 1,
        "cmd": 2,
        "cgm": 3,
        "healthy": 4
    }
    class_map = {v: k for k, v in class_map.items()}

    def __init__(self, ):
        super().__init__()

    @torch.no_grad()
    def preprocess(self, data):
        data = data[0]
        data = bytes(data["body"])

        data = imread(data)
        data = torch.tensor(data, dtype = torch.float32)

        data = data.permute(2, 0, 1)
        data = self.transform(data).unsqueeze(0)

        return data.cuda()

    @torch.no_grad()
    def postprocess(self, data):
        class_indices = data.max(dim = -1).indices.cpu().numpy().tolist()
        class_names = [f"\nPredicted Class: {self.class_map[class_index]}\n" for class_index in class_indices]
        return class_names
