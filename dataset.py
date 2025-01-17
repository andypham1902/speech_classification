import os

import librosa
import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, AddColorNoise, LoudnessNormalization, Normalize, RoomSimulator, Aliasing, BitCrush, BandPassFilter
from tqdm import tqdm

def build_transforms(mode="train"):
    if mode == "train":
        augment = Compose([
            TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            Shift(p=0.7, shift_unit="seconds", min_shift=-1.0, max_shift=1.0, rollover=False),
            AddColorNoise(p=0.5),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            BandPassFilter(min_center_freq=50.0, max_center_freq=7000.0, p=0.3),
            # LoudnessNormalization(p=0.5),
            RoomSimulator(p=0.3),
            Aliasing(p=0.3),
            BitCrush(p=0.3),
            Normalize(p=1.0),
        ])
    else:
        augment = Compose([
            Normalize(p=1.0),
        ])
    return augment

class QuestionDataset(Dataset):
    def __init__(self, df, mode="train", n_mels=256, root="/data/lipsync/question", sr=48000):
        self.mode = mode
        self.paths = df.audio.tolist()
        self.labels = df.label.tolist()
        self.transform = build_transforms(mode)
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
        
        # Normalize audio before padding
        data = data / (np.max(np.abs(data)) + 1e-6)
        
        if len(data) < self.sr:
            data = np.pad(data, (self.sr - len(data), 0))
        data = data[-self.sr:]
        
        if self.mode == "train" or self.mode == "val":
            data = self.transform(data, self.sr)
            
        # Updated mel spectrogram parameters
        mels = librosa.feature.melspectrogram(
            y=data, 
            sr=self.sr, 
            fmax=self.sr//2,
            n_mels=self.n_mels,
            hop_length=512,  # Adjusted hop length
            n_fft=2048,      # Adjusted window size
            power=2.0        # Square of magnitude
        )
        
        # Log-scale mel spectrograms
        mels = librosa.power_to_db(mels, ref=np.max)
        
        # Normalize mel spectrograms
        mels = (mels - mels.mean()) / (mels.std() + 1e-8)
        
        mels = np.expand_dims(mels, axis=0)
        return torch.tensor(mels).float(), torch.tensor(label)
    
class EmotionDataset(Dataset):
    def __init__(self, df, mode="train", sr=48000, length=4):
        self.mode = mode
        self.paths = df["Path"].tolist()
        self.labels = df.label.tolist()
        self.transform = build_transforms(mode)
        self.sr = sr
        self.length = length
        # data_max = -1e6
        # data_min = 1e6
        # for i in tqdm(range(len(self.paths))):
        #     data, _ = librosa.load(self.paths[i], sr=self.sr)
        #     S = librosa.feature.melspectrogram(y=data, sr=sr, power=2.0)
        #     S_dB = librosa.power_to_db(S, ref=np.max)
        #     data_max = max(data_max, np.max(S_dB))
        #     data_min = min(data_min, np.min(S_dB))
        # print("Max:", data_max)
        # print("Min:", data_min)
        data_max = 7.6293945e-06
        data_min = -80.0
        self.data_max = data_max
        self.data_min = data_min

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio_path = self.paths[idx]
        label = self.labels[idx]
        data, _ = librosa.load(audio_path, sr=self.sr)

        if len(data) < self.length * self.sr:
            data = np.pad(data, (self.length * self.sr - len(data), 0))
        data = data[-(self.length * self.sr):]

        data = self.transform(data, self.sr)
        
        if len(data) < self.length * self.sr:
            data = np.pad(data, (self.length * self.sr - len(data), 0))
        data = data[-(self.length * self.sr):]

        S = librosa.feature.melspectrogram(y=data, sr=self.sr, power=2.0)
        S_dB = librosa.power_to_db(S, ref=np.max)
        S_dB = (S_dB - self.data_min) / (self.data_max - self.data_min)
        
        # mels = librosa.feature.melspectrogram(y=data, sr=self.sr, fmax=self.sr/2, n_mels=self.n_mels, hop_length=512, n_fft=2048) # (256, 94)
        # mels = librosa.feature.melspectrogram(y=data, sr=self.sr, fmax=self.sr/2, n_mels=self.n_mels, hop_length=256, n_fft=1024)
        
        # mels = np.expand_dims(mels, axis=0)
        # return torch.tensor(mels).float(), torch.tensor(label)
        return torch.tensor(np.expand_dims(S_dB, axis=0)).float(), torch.tensor(label)


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return {"images": images, "labels": labels}


if __name__ == "__main__":
    df = pd.read_csv("data/train_question.csv")
    dataset = QuestionDataset(df)
    dataset[0]
