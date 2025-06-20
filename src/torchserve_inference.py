# -*- coding: utf-8 -*-

import argparse
import requests
from decouple import config

__all__ = [
    "predict_cassava_disease"
]

MODEL_NAME = config("MODEL_NAME", cast=str)
URL = f"http://localhost:8080/predictions/{MODEL_NAME}"


def predict_cassava_disease(image_path):
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    files = {
        "data": ("image_name.jpg", image_data)
    }

    headers = {
        ""
    }

    response = requests.request(
        "GET",
        url=URL,
        # headers=headers,
        files=files
    )
    
    return response.text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-path", type=str)

    args = parser.parse_args()
    print(predict_cassava_disease(args.image_path))



