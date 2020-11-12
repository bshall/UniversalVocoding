import argparse
import os
import json

import torch
import numpy as np
import soundfile as sf

from univoc.model import Vocoder

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--checkpoint", type=str, help="Checkpoint path to resume")
    # parser.add_argument("--data-dir", type=str, default="./data")
    # parser.add_argument("--gen-dir", type=str, default="./generated")
    # parser.add_argument("--wav-path", type=str)
    # args = parser.parse_args()
    # with open("config.json") as f:
    #     params = json.load(f)
    # os.makedirs(args.gen_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Vocoder(
        n_mels=80,
        conditioning_size=128,
        embedding_dim=256,
        rnn_size=896,
        fc_size=1024,
        bits=10,
        hop_length=200,
    ).cuda()
    model.to(device)
    model.eval()

    # print("Load checkpoint from: {}:".format(args.checkpoint))
    checkpoint = torch.load(
        "vocoder/model.ckpt-50000.pt", map_location=lambda storage, loc: storage
    )
    model.load_state_dict(checkpoint["model"])
    model_step = checkpoint["step"]

    # wav = load_wav(args.wav_path, params["preprocessing"]["sample_rate"])
    # utterance_id = os.path.basename(args.wav_path).split(".")[0]
    # wav = wav / np.abs(wav).max() * 0.999
    # mel = melspectrogram(
    #     wav,
    #     sample_rate=params["preprocessing"]["sample_rate"],
    #     preemph=params["preprocessing"]["preemph"],
    #     num_mels=params["preprocessing"]["num_mels"],
    #     num_fft=params["preprocessing"]["num_fft"],
    #     min_level_db=params["preprocessing"]["min_level_db"],
    #     hop_length=params["preprocessing"]["hop_length"],
    #     win_length=params["preprocessing"]["win_length"],
    #     fmin=params["preprocessing"]["fmin"],
    # )
    mel = np.load("datasets/train/p232/p232_002.mel.npy")
    mel = torch.FloatTensor(mel.T).unsqueeze(0).to(device)
    with torch.no_grad():
        wav = model.generate(mel)
    sf.write("test.wav", wav, 16000)
    path = os.path.join(
        args.gen_dir, "gen_{}_model_steps_{}.wav".format(utterance_id, model_step)
    )
    save_wav(path, output, params["preprocessing"]["sample_rate"])
