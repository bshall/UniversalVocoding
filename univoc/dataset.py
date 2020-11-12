from pathlib import Path
import numpy as np
import torch
import json
import random
from torch.utils.data import Dataset


class VocoderDataset(Dataset):
    def __init__(self, root, sample_frames=24, hop_length=200):
        self.root = Path(root)
        self.sample_frames = sample_frames
        self.hop_length = hop_length

        metadata_path = self.root / "train.json"
        with open(metadata_path) as file:
            metadata = json.load(file)
            self.metadata = [Path(path) for _, path in metadata]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = self.metadata[index]
        path = self.root / path

        audio = np.load(path.with_suffix(".wav.npy"))
        mel = np.load(path.with_suffix(".mel.npy"))

        pos = random.randint(0, mel.shape[-1] - self.sample_frames - 1)
        mel = mel[:, pos : pos + self.sample_frames]

        p, q = pos, pos + self.sample_frames
        audio = audio[p * self.hop_length : q * self.hop_length + 1]

        return torch.LongTensor(audio), torch.FloatTensor(mel.T)
