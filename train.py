from pathlib import Path

import hydra
import hydra.utils as utils

import numpy as np
from tqdm import tqdm
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from univoc import Vocoder, VocoderDataset


def save_checkpoint(vocoder, optimizer, scheduler, scaler, step, checkpoint_dir):
    checkpoint_state = {
        "model": vocoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / f"model-{step}.pt"
    torch.save(checkpoint_state, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path.stem}")


def load_checkpoint(vocoder, optimizer, scaler, scheduler, load_path):
    print(f"Loading checkpoint from {load_path}")
    checkpoint = torch.load(load_path)
    vocoder.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["step"]


@hydra.main(config_path="univoc/config", config_name="train")
def train_model(cfg):
    tensorboard_path = Path(utils.to_absolute_path("tensorboard")) / cfg.checkpoint_dir
    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
    writer = SummaryWriter(tensorboard_path)

    vocoder = Vocoder(**cfg.model).cuda()
    optimizer = optim.Adam(vocoder.parameters(), lr=cfg.train.optimizer.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        cfg.train.scheduler.step_size,
        cfg.train.scheduler.gamma,
    )
    scaler = amp.GradScaler()

    if cfg.resume:
        resume_path = utils.to_absolute_path(cfg.resume)
        global_step = load_checkpoint(
            vocoder=vocoder,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            load_path=resume_path,
        )
    else:
        global_step = 0

    dataset_root = Path(utils.to_absolute_path("datasets"))
    dataset = VocoderDataset(
        dataset_root,
        sample_frames=cfg.train.sample_frames,
        hop_length=cfg.preprocess.hop_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.n_workers,
        pin_memory=True,
        drop_last=True,
    )

    n_epochs = cfg.train.n_steps // len(dataloader) + 1
    start_epoch = global_step // len(dataloader) + 1

    for epoch in range(start_epoch, n_epochs + 1):
        average_loss = 0

        for i, (audio, mels) in enumerate(tqdm(dataloader), 1):
            audio, mels = audio.cuda(), mels.cuda()

            optimizer.zero_grad()

            with amp.autocast():
                wav = vocoder(audio[:, :-1], mels)
                loss = F.cross_entropy(wav.transpose(1, 2), audio[:, 1:])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1

            average_loss += (loss.item() - average_loss) / i

            if global_step % cfg.train.checkpoint_interval == 0:
                save_checkpoint(
                    vocoder, optimizer, scheduler, scaler, global_step, checkpoint_dir
                )

        writer.add_scalar("loss", average_loss, global_step)
        print(f"epoch:{epoch}, loss:{average_loss:.3f}, {scheduler.get_last_lr()}")


if __name__ == "__main__":
    train_model()