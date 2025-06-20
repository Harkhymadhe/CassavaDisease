### Cassava Disease Prediction

---

### Overview

---
This project explores the design and training of convolutional neural networks for predicting cassava diseases. To achieve this,  prerained CNN models were leveraged, such as VGG and ResNet.

Different options were leveraged for deployment, including:

1. TorchServe ✅
2. FastAPI (under development 🗓️)
3. Onnx    (under development 🗓️)

### Dependencies

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi) ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)

Other tools leveraged for this project may be found in the <a hef="pyproject.toml">pyproject.toml</a> file.

### Repository Structure

```
.
├── README.md
├── artefacts
│   └── models                 # Exported model repository
│       └── model_store        # Packaged model repository for Torchserve
├── data
│   └── train                  # Image folder (split into train and test set)
│       ├── cbb                # Disease 1
│       ├── cbsd               # Disease 2
│       ├── cgm                # Disease 3
│       ├── cmd                # Disease 4
│       └── healthy            # Disease 5
├── example.env
├── logs                       # Torchserve deployment logs
├── notebook.ipynb             # Experimentation notebook
├── pyproject.toml
├── src                        # Script files
│   ├── data_ops.py
│   ├── deployment
│   │   └── torchserve
│   │       ├── deployment.py
│   │       └── handler_script.py
│   ├── main.py
│   ├── models.py
│   ├── torchserve_inference.py
│   ├── training_ops.py
│   └── utils.py
├── torchserve_deployment_setup.sh  # Bash script to setup model archive for Torchserve
└── torchserve_inference_setup.sh   # Bash script to setup inference endpoints for Torchserve
```

---

## Deployment (TorchServe)

---

Deploying with TorchServe is easy. To test, run the example code below to package the trained model as a model archive. Be sure to first set the `MODEL_NAME` variable:

```bash
$ torch-model-archiver --model-name $MODEL_NAME --version 1.0 --model-file src/models.py --serialized-file artefacts/models/scripted_model.pt --handler src/deployment/torchserve/handler_script.py --export-path artefacts/models/model_store
```
The final deployment will be served from this archive. To start the deployment from the model archive, run the code below:

```bash
$ torchserve --start --model-store artefacts/models/model_store --models $MODEL_NAME=$MODEL_NAME.mar --ncs --disable-token-auth
```

This will expose deployment endpoints as shown below:

| Port    | Purpose    |
|---------|------------|
| 8080    | Inference  |
| 8081    | Management |
| 8082    | Metrics    |

To simplify this process, just:

1. Set the `MODEL_NAME` variable in the `.env` file at base project directory and
2. Run the torchserve bash scripts provided.

See below:

```bash
$ source .env && ./torchserve_deployment_setup.sh && ./torchserve_inference_setup.sh
```

---

## Inference (TorchServe)

---

There are two options to perform inference.

1️⃣ Python 🐍

Run the following from the command line:

```bash
$ python3 src/torchserve_inference.py  --image-path < PATH_TO_IMAGE >
```

2️⃣ via CuRL

Run the following from the command line:

```bash
$ source .env && curl http://localhost:8080/predictions/$MODEL_NAME  -T < PATH_TO_IMAGE >
```

When inference is done, the server can be stopped with:

```bash
$ torchserve --stop
```