# -*- coding: utf-8 -*-

"""Handler script for inference."""

import numpy as np
import cv2

import torch
from ts.torch_handler.image_classifier import ImageClassifier

from torchvision import transforms as T

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


    def initialize(self, context):
        self._context = context
        self.manifest = context.manifest

        model_file = self.manifest['model']['serializedFile']
        model_dir = context.system_properties.get("model_dir")

        import os
        model_path = os.path.join(model_dir, model_file)


        self.model = torch.load(model_path)
        self.model.eval()
        self.initialized = True

    @torch.no_grad()
    def preprocess(self, data):

        print(data)
        data = data[0]

        try:
            data = np.frombuffer(data["data"], dtype=np.uint8)
        except:
            data = np.frombuffer(data["body"], dtype=np.uint8)

        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        data = np.asarray(data, dtype="uint8")

        data = torch.tensor(data, dtype = torch.float32)

        data = data.permute(2, 0, 1)
        data = self.transform(data).unsqueeze(0)

        return data.cuda()

    @torch.no_grad()
    def inference(self, data):
        if len(data.shape) < 4:
            data = data.unsqueeze(0)

        prediction = self.model(data)
        return prediction

    @torch.no_grad()
    def postprocess(self, data):
        class_indices = data.max(dim = -1).indices.cpu().numpy().tolist()
        class_names = [f"\nPredicted Class: {self.class_map[class_index]}\n" for class_index in class_indices]
        return class_names
