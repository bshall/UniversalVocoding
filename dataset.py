import numpy as np
import torch
import os
from random import randint
from torch.utils.data import Dataset


class VocoderDataset(Dataset):
    def __init__(self, meta_file, sample_frames, hop_length, bits):
        self.sample_frames = sample_frames
        self.hop_length = hop_length
        self.bits = bits

        with open(meta_file, encoding="utf-8") as f:
            self.metadata = [line.strip().split("|") for line in f]
        self.metadata = [m for m in self.metadata if int(m[3]) > self.sample_frames + 1]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        _, audio_path, mel_path, _ = self.metadata[index]

        audio = np.load(os.path.join(audio_path))
        mel = np.load(os.path.join(mel_path))

        rand_pos = randint(0, mel.shape[0] - self.sample_frames - 2)

        audio = audio[rand_pos*self.hop_length:(rand_pos + self.sample_frames) * self.hop_length + 1]
        mel = mel[rand_pos:rand_pos + self.sample_frames, :]

        return torch.LongTensor(audio), torch.FloatTensor(mel)
