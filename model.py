import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import mulaw_decode
from tqdm import tqdm


def fold_with_overlap(x, target, overlap):
    _, sample_size, channels = x.size()

    # Calculate variables needed
    num_folds = (sample_size - overlap) // (target + overlap)
    extended_len = num_folds * (overlap + target) + overlap
    remaining = sample_size - extended_len

    # Pad if some time steps poking out
    if remaining != 0:
        num_folds += 1
        padding = target + 2 * overlap - remaining
        x = F.pad(x, [0, 0, 0, padding], mode="constant", value=0)

    folded = torch.zeros(num_folds, target + 2 * overlap, channels, device=x.device)

    # Get the values for the folded tensor
    for i in range(num_folds):
        start = i * (target + overlap)
        end = start + target + 2 * overlap
        folded[i] = x[:, start:end, :]

    return folded


def xfade_and_unfold(y, overlap):
    num_folds, length = y.shape
    target = length - 2 * overlap
    total_len = num_folds * (target + overlap) + overlap

    # Need some silence for the rnn warmup
    silence_len = overlap // 2
    fade_len = overlap - silence_len
    silence = np.zeros(silence_len, dtype=np.float64)

    # Equal power crossfade
    t = np.linspace(-1, 1, fade_len, dtype=np.float64)
    fade_in = np.sqrt(0.5 * (1 + t))
    fade_out = np.sqrt(0.5 * (1 - t))

    # Concat the silence to the fades
    fade_in = np.concatenate([silence, fade_in])
    fade_out = np.concatenate([fade_out, silence])

    # Apply the gain to the overlap samples
    y[:, :overlap] *= fade_in
    y[:, -overlap:] *= fade_out

    unfolded = np.zeros(total_len, dtype=np.float64)

    # Loop to add up all the samples
    for i in range(num_folds):
        start = i * (target + overlap)
        end = start + target + 2 * overlap
        unfolded[start:end] += y[i]

    return unfolded


def get_gru_cell(gru):
    gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    gru_cell.weight_hh.data = gru.weight_hh_l0.data
    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data
    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    return gru_cell


class Vocoder(nn.Module):
    def __init__(self, mel_channels, conditioning_channels, embedding_dim,
                 rnn_channels, fc_channels, bits, hop_length):
        super().__init__()
        self.rnn_channels = rnn_channels
        self.quantization_channels = 2**bits
        self.hop_length = hop_length

        self.rnn1 = nn.GRU(mel_channels, conditioning_channels, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(2 * conditioning_channels, conditioning_channels, batch_first=True, bidirectional=True)
        self.embedding = nn.Embedding(self.quantization_channels, embedding_dim)
        self.rnn3 = nn.GRU(embedding_dim + 2 * conditioning_channels, rnn_channels, batch_first=True)
        self.fc1 = nn.Linear(rnn_channels, fc_channels)
        self.fc2 = nn.Linear(fc_channels, self.quantization_channels)

    def forward(self, x, mels):
        sample_frames = mels.size(1)
        audio_slice_frames = x.size(1) // self.hop_length
        pad = (sample_frames - audio_slice_frames) // 2

        mels, _ = self.rnn1(mels)
        mels, _ = self.rnn2(mels)
        mels = mels[:, pad:pad + audio_slice_frames, :]

        mels = F.interpolate(mels.transpose(1, 2), scale_factor=self.hop_length)
        mels = mels.transpose(1, 2)

        x = self.embedding(x)

        x, _ = self.rnn3(torch.cat((x, mels), dim=2))

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def generate(self, mel, batched, target, overlap):
        self.eval()

        output = []
        cell = get_gru_cell(self.rnn3)

        with torch.no_grad():
            mel, _ = self.rnn1(mel)
            mel, _ = self.rnn2(mel)

            mel = F.interpolate(mel.transpose(1, 2), scale_factor=self.hop_length)
            mel = mel.transpose(1, 2)

            if batched:
                mel = fold_with_overlap(mel, target, overlap)

            batch_size, sample_size, _ = mel.size()

            h = torch.zeros(batch_size, self.rnn_channels, device=mel.device)
            x = torch.zeros(batch_size, device=mel.device).fill_(self.quantization_channels // 2).long()

            for m in tqdm(torch.unbind(mel, dim=1), leave=False):
                x = self.embedding(x)
                h = cell(torch.cat((x, m), dim=1), h)

                x = F.relu(self.fc1(h))
                logits = self.fc2(x)

                posterior = F.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(posterior)

                x = dist.sample()
                output.append(2 * x.float() / (self.quantization_channels - 1.) - 1.)

        output = torch.stack(output).transpose(0, 1)
        output = output.cpu().numpy()
        output = output.astype(np.float64)

        if batched:
            output = xfade_and_unfold(output, overlap)
        else:
            output = output[0]

        output = mulaw_decode(output, self.quantization_channels)

        self.train()
        return output
