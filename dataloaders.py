import numpy as np
from torch.utils.data import Dataset

from data import *


class UnlabeldDataset(Dataset):
    def __init__(self, x, transform=None):
        self.x = x
        self.n = len(self.x)
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = self.x[idx]

        if self.transform is not None:
            transformed = self.transform(image=image)

            image = transformed["image"]
            del transformed

        batch = dict()
        batch['image'] = np.transpose(image, (2, 0, 1))

        return batch


class ImageOutcomeDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x, self.y = x, y
        self.n = len(self.x)
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image, label = self.x[idx], self.y[idx]

        if self.transform is not None:
            transformed = self.transform(image=image)

            image = transformed["image"]
            del transformed

        batch = dict()
        batch['image'] = np.transpose(image, (2, 0, 1))
        batch['label'] = label

        return batch


class ImageDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x, self.y = x, y
        self.n = len(self.x)
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image, label = self.x[idx], self.y[idx]

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)

            image = transformed["image"]
            label = transformed["mask"]
            del transformed

        batch = dict()
        batch['image'] = np.transpose(image, (2, 0, 1))
        batch['label'] = np.clip(np.transpose(label, (2, 0, 1)), 0, 1)

        return batch


class DenosingDataset(Dataset):
    def __init__(self, img_ids, data_dir, noising_transform, config, transform=None):
        self.x, self.y = load_data_bonbidhie2023(img_ids, data_dir, config)
        self.noise_transform = noising_transform
        self.transform = transform
        self.config = config

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        clean, mask = self.x[idx], self.y[idx]
        noise_transformed = self.noise_transform.apply(clean, mask)

        if self.transform is not None:
            inp, tar = noise_transformed["input"], noise_transformed["target"]
            transformed = self.transform(image=inp, mask=tar)
            noise_transformed["input"] = transformed["image"]
            noise_transformed["target"] = transformed["mask"]
            del transformed

        batch = {k: np.transpose(v, (2, 0, 1)) for k, v in noise_transformed.items()}
        batch['clean'] = np.transpose(clean, (2, 0, 1))

        return batch


class VolumeDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x, self.y = x, y
        self.n = len(self.x)
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image, label = self.x[idx], self.y[idx]

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)

            image = transformed["image"]
            label = transformed["mask"]
            del transformed

        batch = dict()
        batch['image'] = np.transpose(image, (3, 1, 2, 0))
        batch['label'] = np.transpose(label, (3, 1, 2, 0))

        return batch


class VolumeDenosingDataset(Dataset):
    def __init__(self, img_ids, data_dir, noising_transform, config, transform=None):
        self.x, self.y = load_data_bonbidhie2023_3d(img_ids, data_dir, config)
        self.noise_transform = noising_transform
        self.transform = transform
        self.config = config

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        clean, mask = self.x[idx], self.y[idx]
        noise_transformed = self.noise_transform.apply(clean, mask)

        if self.transform is not None:
            inp, tar = noise_transformed["input"], noise_transformed["target"]
            transformed = self.transform(image=inp, mask=tar)
            noise_transformed["input"] = transformed["image"]
            noise_transformed["target"] = transformed["mask"]
            del transformed

        batch = {k: np.transpose(v, (3, 1, 2, 0)) for k, v in noise_transformed.items()}
        batch['clean'] = np.transpose(clean, (3, 1, 2, 0))

        return batch
