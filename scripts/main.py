# -* coding: utf-8 -*-

import os
from argparse import ArgumentParser

import torch

from data_ops import (
    prepare_cassava_datasets,
    load_cassava_data,
)

from training_ops import training_loop, prepare_optimizer
from architectures import CassavaModel

from utils import visualize_accuracy, visualize_loss

NUM_CLASSES = 5  # Number of categories


def parse_args():
    args = ArgumentParser()

    args.add_argument("--device", default="cpu", type=str, help="Computation device")

    args.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs to train"
    )

    args.add_argument(
        "--batch_size", default=8, type=int, help="Batch size for training"
    )

    args.add_argument(
        "--size", default=224, type=int, help="Target size for training images"
    )

    args.add_argument("--lr", default=0.001, type=float, help="Learning rate")

    args.add_argument("--factor", default=10, type=int, help="Learning rate factor")

    args.add_argument("--amsgrad", "-a", action="store_true", help="Use Amsgrad Adam variant?")

    args.add_argument("--beta1",  default=0.9, type=float, help="Beta 1")

    args.add_argument("--beta2",  default=0.999, type=float, help="Beta 2")

    args.add_argument(
        "--out_features", default=1, type=int, help="Number of output features"
    )

    args.add_argument(
        "--freeze", action="store_true", help="Freeze pretrained weights?"
    )

    args.add_argument(
        "--pf",
        default="resnet18",
        choices=["resnet18", "resnet34", "vgg11", "vgg13", "vgg19"],
        help="Function to return pretrained model",
    )
    return args


def main(args):
    args = args.parse_args()
    device = torch.device(args.device)

    fpath = os.getcwd().replace("scripts", "data")

    data = load_cassava_data(data_path=fpath, size = args.size,)
    train_dl, test_dl = prepare_cassava_datasets(data, batch_size=args.batch_size)

    model = CassavaModel(
        out_features=NUM_CLASSES,
        freeze_weights=args.freeze,
        pretrained_model_function=args.pf,
    ).to(device)

    model_opt = prepare_optimizer(
        model, lr=args.lr, factor=args.factor,
        amsgrad=args.amsgrad, betas=(args.beta1, args.beta2), device=device
    )

    model, optimizer = model_opt["model"], model_opt["optimizer"]

    results = training_loop(
        model=model,
        data=[train_dl, test_dl],
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
    )

    visualize_accuracy(results)
    visualize_loss(results)
    return  # results


if __name__ == "__main__":
    args = parse_args()
    main(args)
