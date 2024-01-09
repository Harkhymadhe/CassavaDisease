# -*- coding: utf-8 -*-

import os
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T

from random import shuffle
from PIL import Image


__all__ = [
    "CassavaDataset",
    "load_cassava_data",
    "prepare_cassava_datasets",
]


class CassavaDataset(Dataset):
    def __init__(self, path, size = None, transform=None):
        self.path = path
        self.size = size if size is not None else (224, 224)

        if transform is None:
            transform = T.Compose(
                [
                    T.Resize(self.size),
                    T.ToTensor(),
                    T.Normalize(mean=[.485, 0.456, .406], std=[.229, .224, .225]),
                    # T.RandomAdjustSharpness(sharpness_factor=0, p=.3),
                    # T.RandomAdjustSharpness(sharpness_factor=2, p=.3),
                    # T.RandomHorizontalFlip(p=.4),
                ]

            )

        self.transform = transform

        classes = os.listdir(path)
        self.class_map = dict(zip(classes, [_ for _ in range(len(classes))]))

        self.files = []

        for class_ in classes:
            class_files = os.listdir(os.path.join(path, class_))
            self.files += [
                (os.path.join(path, class_, f), class_) for f in class_files
            ]

        shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img, class_ = self.files[index]
        img = Image.open(img)

        return self.transform(img), self.class_map[class_]


def load_cassava_data(data_path="data", test_size=.3, size = 224, transform=None):
    # Instantiate dataset
    dataset = CassavaDataset(
        path=data_path,
        size=size,
        transform=transform
    )

    # Split into train and test sets
    train_ds, test_ds = random_split(dataset=dataset, lengths=[1 - test_size, test_size])

    return train_ds, test_ds


def prepare_cassava_datasets(
    data,
    batch_size=32,
):
    train_ds, test_ds = data

    # Generate DataLoaders
    train_dl, test_dl = (
        DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)
    )

    return train_dl, test_dl
