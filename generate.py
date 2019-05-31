import argparse
import os
import json

import torch
import numpy as np

from model import Vocoder
from utils import load_wav, save_wav, melspectrogram

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Checkpoint path to resume")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--gen-dir", type=str, default="./generated")
    parser.add_argument("--wav_path", type=str)
    parser.add_argument("--batched", action='store_true')
    args = parser.parse_args()
    with open("config.json") as f:
        params = json.load(f)
    os.makedirs(args.gen_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Vocoder(mel_channels=params["preprocessing"]["num_mels"],
                    conditioning_channels=params["vocoder"]["conditioning_channels"],
                    embedding_dim=params["vocoder"]["embedding_dim"],
                    rnn_channels=params["vocoder"]["rnn_channels"],
                    fc_channels=params["vocoder"]["fc_channels"],
                    bits=params["preprocessing"]["bits"],
                    hop_length=params["preprocessing"]["hop_length"])
    model.to(device)

    print("Resume checkpoint from: {}:".format(args.resume))
    checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])
    model_step = checkpoint["steps"]

    wav = load_wav(args.wav_path, params["preprocessing"]["sample_rate"])
    utterance_id = os.path.basename(args.wav_path).split(".")[0]
    wav = wav / np.abs(wav).max() * 0.999
    mel = melspectrogram(wav, sample_rate=params["preprocessing"]["sample_rate"],
                         num_mels=params["preprocessing"]["num_mels"],
                         num_fft=params["preprocessing"]["num_fft"],
                         preemph=params["preprocessing"]["preemph"],
                         min_level_db=params["preprocessing"]["min_level_db"],
                         hop_length=params["preprocessing"]["hop_length"],
                         win_length=params["preprocessing"]["win_length"],
                         fmin=params["preprocessing"]["fmin"])
    mel = torch.FloatTensor(mel).unsqueeze(0).to(device)
    output = model.generate(mel, args.batched,
                            params["vocoder"]["generate"]["target"],
                            params["vocoder"]["generate"]["overlap"])
    path = os.path.join(args.gen_dir, "gen_{}_model_steps_{}.wav".format(utterance_id, model_step))
    save_wav(path, output, params["preprocessing"]["sample_rate"])
