import numpy as np
import torch
import os
from random import randint
from torch.utils.data import Dataset


class VocoderDataset(Dataset):
    def __init__(self, meta_file, sample_frames, audio_slice_frames, hop_length, bits):
        self.sample_frames = sample_frames
        self.audio_slice_frames = audio_slice_frames
        self.pad = (sample_frames - audio_slice_frames) // 2
        self.hop_length = hop_length
        self.bits = bits

        with open(meta_file, encoding="utf-8") as f:
            self.metadata = [line.strip().split("|") for line in f]
        self.metadata = [m for m in self.metadata if int(m[3]) > 2 * sample_frames - audio_slice_frames]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        _, audio_path, mel_path, _ = self.metadata[index]

        audio = np.load(os.path.join(audio_path))
        mel = np.load(os.path.join(mel_path))

        pos = randint(0, len(mel) - self.sample_frames)
        mel = mel[pos:pos + self.sample_frames, :]

        p, q = pos + self.pad, pos + self.pad + self.audio_slice_frames
        audio = audio[p * self.hop_length:q * self.hop_length + 1]

        return torch.LongTensor(audio), torch.FloatTensor(mel)
