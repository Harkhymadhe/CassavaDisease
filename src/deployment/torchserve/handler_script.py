# -*- coding: utf-8 -*-

"""Handler script for inference."""

import numpy as np
import cv2

import torch
from ts.torch_handler.image_classifier import ImageClassifier

from torchvision import transforms as T

__all__ = [
    "TorchServeCassavaClassifier",
]


class TorchServeCassavaClassifier(ImageClassifier):
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


        self.model = torch.load(model_path, weights_only=False)
        self.model.eval()
        self.initialized = True

    @torch.no_grad()
    def preprocess(self, data):
        data = data[0]

        print(data.keys())

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
        prediction = self.model(data)
        return prediction

    @torch.no_grad()
    def postprocess(self, data):
        probs = torch.softmax(data, dim=-1).cpu().squeeze(0).numpy().tolist()
        prob_map = {self.class_map[i]: probs[i] for i in range(len(probs))}

        pred_class = data.max(dim = -1).indices.cpu().numpy().item()

        results = {
            "probabilities": prob_map,
            "prediction": self.class_map[pred_class]
        }

        return [results]
