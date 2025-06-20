# -*- coding: utf-8 -*-

import requests

__all__ = [
    "predict_cassava_disease"
]

URL = "http://localhost:8080/predictions/cassava_model"


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
    print(predict_cassava_disease("test_img.jpg"))



