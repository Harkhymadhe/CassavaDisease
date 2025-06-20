# -*- coding: utf-8 -*-

from torch import nn

from matplotlib import pyplot as plt
import seaborn as sns


__all__ = [
    "initialize_model_weights",
    "visualize_loss",
    "visualize_accuracy",
    "visualize_metrics",
]


def initialize_model_weights(model, init_func = nn.init.normal_):
    for name, parameter in model.named_parameters():
        if name in ["fc", "classifier"]:
            init_func(parameter)
        else:
            continue

    return model


def visualize_loss(results):
    sns.set()

    epochs = results["epochs"]
    loss = results["performance"]["loss"]
    train_loss, test_loss = loss["train"], loss["test"]

    plt.plot([i for i in range(len(epochs))], train_loss, label = "Train Loss", color = "r")
    plt.plot([i for i in range(len(epochs))], test_loss, label="Test Loss", color="r")

    plt.title(f"Loss Evolution over {epochs} Epochs")

    plt.show(); plt.close("all")

    return


def visualize_accuracy(results):
    sns.set()

    epochs = results["epochs"]
    loss = results["performance"]["accuracy"]
    train_accuracy, test_accuracy = loss["train"], loss["test"]

    plt.plot([i for i in range(len(epochs))], train_accuracy, label = "Train Accuracy", color = "r")
    plt.plot([i for i in range(len(epochs))], test_accuracy, label="Test Accuracy", color="r")

    plt.title(f"Accuracy Evolution over {epochs} Epochs")

    plt.show(); plt.close("all")

    return


def visualize_metrics(results):
    visualize_loss(results=results)
    visualize_accuracy(results=results)

    return
