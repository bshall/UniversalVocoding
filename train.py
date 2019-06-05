import argparse
import os
import json

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import save_wav
from dataset import VocoderDataset
from model import Vocoder


def save_checkpoint(model, step, checkpoint_dir):
    checkpoint_state = {
        "model": model.state_dict(),
        "step": step}
    checkpoint_path = os.path.join(
        checkpoint_dir, "model.ckpt-{}.pt".format(step))
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path))


def train_fn(args, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Vocoder(mel_channels=params["preprocessing"]["num_mels"],
                    conditioning_channels=params["vocoder"]["conditioning_channels"],
                    embedding_dim=params["vocoder"]["embedding_dim"],
                    rnn_channels=params["vocoder"]["rnn_channels"],
                    fc_channels=params["vocoder"]["fc_channels"],
                    bits=params["preprocessing"]["bits"],
                    hop_length=params["preprocessing"]["hop_length"])
    model.to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=params["vocoder"]["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, params["vocoder"]["schedule"]["step_size"], params["vocoder"]["schedule"]["gamma"])

    if args.resume is not None:
        print("Resume checkpoint from: {}:".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    train_dataset = VocoderDataset(meta_file=os.path.join(args.data_dir, "train.txt"),
                                   sample_frames=params["vocoder"]["sample_frames"],
                                   audio_slice_frames=params["vocoder"]["audio_slice_frames"],
                                   hop_length=params["preprocessing"]["hop_length"],
                                   bits=params["preprocessing"]["bits"])

    train_dataloader = DataLoader(train_dataset, batch_size=params["vocoder"]["batch_size"],
                                  shuffle=True, num_workers=1,
                                  pin_memory=True)

    num_epochs = params["vocoder"]["num_steps"] // len(train_dataloader) + 1
    start_epoch = global_step // len(train_dataloader) + 1

    for epoch in range(start_epoch, num_epochs + 1):
        running_loss = 0

        for i, (audio, mels) in enumerate(tqdm(train_dataloader), 1):
            audio, mels = audio.to(device), mels.to(device)

            output = model(audio[:, :-1], mels)
            loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            average_loss = running_loss / i

            global_step += 1

            if global_step % params["vocoder"]["checkpoint_interval"] == 0:
                save_checkpoint(model, global_step, args.checkpoint_dir)

                with open(os.path.join(args.data_dir, "test.txt"), encoding="utf-8") as f:
                    test_mel_paths = [line.strip().split("|")[2] for line in f]

                for mel_path in test_mel_paths:
                    utterance_id = os.path.basename(mel_path).split(".")[0]
                    mel = torch.FloatTensor(np.load(mel_path)).unsqueeze(0).to(device)
                    output = model.generate(mel, params["vocoder"]["generate"]["batched"],
                                            params["vocoder"]["generate"]["target"],
                                            params["vocoder"]["generate"]["overlap"])
                    path = os.path.join(args.gen_dir, "gen_{}_model_steps_{}.wav".format(utterance_id, global_step))
                    save_wav(path, output, params["preprocessing"]["sample_rate"])

        print("epoch:{}, loss:{:.3f}".format(epoch, average_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Directory to save checkpoints.")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--gen_dir", type=str, default="./generated")
    args = parser.parse_args()
    with open("config.json") as f:
        params = json.load(f)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.gen_dir, exist_ok=True)
    train_fn(args, params)
