import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


class QuestionDataset(Dataset):
    def __init__(
        self, data_path, split="train", fold=0, size=(256, 256), transform=None
    ):
        self.split = split
        self.fold = fold
        self.image_paths, self.labels = self.get_data(data_path)

        self.transform = transform
        self.size = size

    def get_data(self, data_path):
        data_folders = data_path.split(",")

        image_paths = []
        labels = []
        for folder in data_folders:
            print(f"Loading data from {folder}")
            csv_data = pd.read_csv(os.path.join(folder, "train-metadata.csv"))

            if "fold" in csv_data.columns:
                if self.split == "train":
                    csv_data = csv_data[csv_data["fold"] != self.fold]
                else:
                    csv_data = csv_data[csv_data["fold"] == self.fold]
            image_folder = os.path.join(folder, "train-image")

            # look for all jpg images recursively
            isic_id_to_image_paths = {}
            for root, _, files in os.walk(image_folder):
                for file in files:
                    if file.endswith(".jpg"):
                        image_path = os.path.join(root, file)

                        isic_id = "_".join(file.split(".")[0].split("_")[:2])
                        isic_id_to_image_paths[isic_id] = image_path

            for isic_id, target in zip(csv_data["isic_id"], csv_data["target"]):
                if isic_id not in isic_id_to_image_paths:
                    continue
                image_path = isic_id_to_image_paths[isic_id]
                image_paths.append(image_path)
                labels.append(int(target))

        print("Number of positive samples:", sum(labels))

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        # channel first
        image = image.transpose(2, 0, 1)
        image = image.astype("float32")
        image /= 255.0

        label = self.labels[idx]

        return torch.tensor(image).float(), torch.tensor(label)


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return {"images": images, "labels": labels}


class RatioSampler(Sampler):
    def __init__(self, dataset, ratio, num_replicas=None, rank=None):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self.dataset = dataset
        self.ratio = ratio
        self.num_replicas = (
            num_replicas if num_replicas is not None else dist.get_world_size()
        )
        self.rank = rank if rank is not None else dist.get_rank()
        self.indices = list(range(len(dataset)))
        self.positive_indices = np.where(np.array(dataset.labels) == 1)[0].tolist()
        self.negative_indices = np.where(np.array(dataset.labels) == 0)[0].tolist()
        self.sampled_indices = self._resample_indices()

    def _resample_indices(self):
        np.random.shuffle(self.negative_indices)
        num_positives = len(self.positive_indices)
        num_negatives = min(int(self.ratio * num_positives), len(self.negative_indices))
        sampled_indices = self.positive_indices + self.negative_indices[:num_negatives]
        np.random.shuffle(sampled_indices)
        return sampled_indices

    def __iter__(self):
        # Split indices for distributed training
        indices = self.sampled_indices[self.rank :: self.num_replicas]
        return iter(indices)

    def __len__(self):
        return len(self.sampled_indices) // self.num_replicas


if __name__ == "__main__":
    dataset = QuestionDataset(data_path="./data", split="train", fold=0)
    print(len(dataset))
    print(dataset[0])
