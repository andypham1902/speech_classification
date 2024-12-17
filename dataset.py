import os

import librosa
import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift


def build_transforms():
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(p=0.5),
    ])
    return augment

class QuestionDataset(Dataset):
    def __init__(self, df, mode="train", n_mels=256, root="/data/lipsync/question", sr=48000):
        self.mode = mode
        self.paths = df.audio.tolist()
        self.labels = df.label.tolist()
        self.transform = build_transforms()
        self.n_mels = n_mels
        self.root = root
        self.sr = sr

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio_path = self.paths[idx]
        label = self.labels[idx]
        audio_path = audio_path.replace("Question-Statement", "Question-Statement_clean")
        data, _ = librosa.load(os.path.join(self.root, audio_path), sr=self.sr)
        if len(data) < self.sr:
            data = np.pad(data, (self.sr - len(data), 0))
        data = data[-self.sr:]
        if self.mode == "train":
            data = self.transform(data, self.sr)
        mels = librosa.feature.melspectrogram(y=data, sr=self.sr, fmax=self.sr//2, n_mels=self.n_mels) # (256, 94)
        mels = np.expand_dims(mels, axis=0)
        return torch.tensor(mels).float(), torch.tensor(label)
    
class EmotionDataset(Dataset):
    def __init__(self, df, mode="train", n_mels=128, sr=48000, length=4):
        self.mode = mode
        self.paths = df["Path"].tolist()
        self.labels = df.label.tolist()
        self.transform = build_transforms()
        self.n_mels = n_mels
        self.sr = sr
        self.length = length

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio_path = self.paths[idx]
        label = self.labels[idx]
        data, _ = librosa.load(audio_path, sr=self.sr)
        if len(data) < self.length * self.sr:
            data = np.pad(data, (self.length * self.sr - len(data), 0))
        data = data[-(self.length * self.sr):]
        if self.mode == "train":
            data = self.transform(data, self.sr)
        mels = librosa.feature.melspectrogram(y=data, sr=self.sr, fmax=self.sr/2, n_mels=self.n_mels, hop_length=512, n_fft=2048) # (256, 94)
        mels = np.expand_dims(mels, axis=0)
        return torch.tensor(mels).float(), torch.tensor(label)


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return {"images": images, "labels": labels}


if __name__ == "__main__":
    df = pd.read_csv("data/train_question.csv")
    dataset = QuestionDataset(df)
    dataset[0]
