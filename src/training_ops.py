# -*- coding: utf-8 -*-

import torch
from torch import nn, optim

from sklearn.metrics import accuracy_score


__all__ = [
    "prepare_optimizer",
    "training_loop",
]


def prepare_optimizer(model, lr=1e-3, factor=10, amsgrad = False, betas = (.9, .999), device="cpu"):
    if isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)

    try:
        # Adam optimizer
        opt = optim.Adam(
            params=[
                {
                    "params": model.base.classifier.parameters(),
                    "lr": lr
                }
            ],
            lr=lr / factor,
            amsgrad=amsgrad,
            betas=betas
        )
    except:
        # Adam optimizer
        opt = optim.Adam(
            params=[
                {
                    "params": model.base.classifier.parameters(),
                    "lr": lr
                }
            ],
            lr=lr / factor,
            amsgrad=amsgrad,
            betas=betas
        )

    return {
        "optimizer": opt,
        "model": model
    }


def training_loop(
    model,
    data,
    optimizer,
    epochs=1000,
    device="cpu",
):
    train_dl, test_dl = data

    criterion = nn.CrossEntropyLoss()

    TRAIN_LOSSES, TEST_LOSSES = [], []
    TRAIN_ACCS, TEST_ACCS = [], []

    for epoch in range(1, epochs + 1):
        train_losses, test_losses = [], []
        train_accs, test_accs = [], []

        model.train()  # Set up training mode

        for X, y in iter(train_dl):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            train_loss = criterion(y_pred, y)  # Compare actual targets and predicted targets to get the loss
            train_loss.backward()  # Backpropagate the loss

            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(train_loss.item())

            train_acc = accuracy_score(y.cpu().numpy(), y_pred.max(dim=-1).indices.cpu().numpy())
            train_accs.append(train_acc)

        with torch.no_grad():  # Turn off computational graph
            model.eval()  # Set model to evaluation mode
            for X_, y_ in iter(test_dl):
                X_, y_ = X_.to(device), y_.to(device)

                y_pred_ = model(X_)

                test_loss = criterion(y_pred_, y_)  # Compare actual targets and predicted targets to get the loss
                test_losses.append(test_loss.item())

                test_acc = accuracy_score(y_.cpu().numpy(), y_pred_.max(dim=-1).indices.cpu().numpy())
                test_accs.append(test_acc)

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_test_loss = sum(test_losses) / len(test_losses)

        avg_train_acc = sum(train_accs) / len(train_accs)
        avg_test_acc = sum(test_accs) / len(test_accs)

        print(
            f"Epoch: {epoch} | Train loss: {avg_train_loss: .3f} | Test loss: {avg_test_loss: .3f} |",
            f"Train accuracy: {avg_train_acc: .3f} | Test accuracy: {avg_test_acc: .3f} |"
        )

        TRAIN_LOSSES.append(avg_train_loss)
        TEST_LOSSES.append(avg_test_loss)

        TRAIN_ACCS.append(avg_train_acc)
        TEST_ACCS.append(avg_test_acc)

    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.clear_autocast_cache()

    return {
        "epochs": epochs,
        "optimizer": optimizer,
        "model": model,
        "performance": {
            "loss": {
                "train": TRAIN_LOSSES,
                "test": TEST_LOSSES,
            },
            "accuracy": {
                "train": TRAIN_ACCS,
                "test": TEST_ACCS,
            },
        },
    }
