"""Samuel V2 training loop (fixed version)."""

import argparse
import math
import time
from pathlib import Path
from typing import Optional
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from samuel.configs.model_config import ModelConfig
from samuel.configs.training_config import TrainingConfig
from samuel.model.transformer import SamuelModel
from samuel.data.dataset import StreamingSamuelDataset


class Trainer:
    """Training loop for Samuel."""

    def __init__(
        self,
        model: SamuelModel,
        train_config: TrainingConfig,
        train_dataset: StreamingSamuelDataset,
        val_dataset: Optional[StreamingSamuelDataset] = None,
    ):
        self.model = model
        self.config = train_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Device
        if train_config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(train_config.device)

        print(f"Training on: {self.device}")

        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # DataLoader
        num_workers = 0 if sys.platform == "win32" else 2

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=train_config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(self.device.type == "cuda"),
                drop_last=True,
            )

        # State
        self.step = 0
        self.best_val_loss = float("inf")
        self.last_val_loss = float("inf")
        self.train_losses = []

        self.output_dir = Path(train_config.output_dir)
        self.checkpoint_dir = self.output_dir / train_config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- OPTIMIZER ----------------

    def _create_optimizer(self):
        decay, no_decay = [], []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2 or "bias" in name or "norm" in name:
                no_decay.append(p)
            else:
                decay.append(p)

        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )

    # ---------------- LR ----------------

    def _lr(self):
        if self.step < self.config.warmup_steps:
            return self.config.learning_rate * self.step / self.config.warmup_steps

        decay_steps = max(1, self.config.max_steps - self.config.warmup_steps)
        t = (self.step - self.config.warmup_steps) / decay_steps
        t = min(max(t, 0.0), 1.0)

        return self.config.min_learning_rate + (
            0.5 * (1 + math.cos(math.pi * t))
        ) * (self.config.learning_rate - self.config.min_learning_rate)

    def _update_lr(self):
        lr = self._lr()
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    # ---------------- CHECKPOINTS ----------------

    def save_checkpoint(self, name="latest", is_best=False):
        ckpt = {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(ckpt, path)

        if is_best:
            torch.save(ckpt, self.checkpoint_dir / "best.pt")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed at step {self.step}")

    # ---------------- TRAIN ----------------

    def train(self):
        print("\n=== TRAIN START ===")
        print(self.model.summary())

        # IMPORTANT: ensure we ALWAYS start loop
        self.save_checkpoint("init")

        self.model.train()
        data_iter = iter(self.train_loader)

        print("🔥 TRAIN LOOP ENTERED")

        while self.step < self.config.max_steps:

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            x = batch["input_ids"].to(self.device)
            y = batch["targets"].to(self.device)

            out = self.model(x, y)
            loss = out["loss"]

            loss.backward()

            if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            lr = self._update_lr()
            self.step += 1

            # LOGGING (FIXED: always runs)
            if self.step % self.config.log_interval == 0:
                print(
                    f"step {self.step} | "
                    f"loss {loss.item():.4f} | "
                    f"lr {lr:.2e}"
                )

            # EVAL
            if self.val_loader and self.step % self.config.eval_interval == 0:
                val_loss = self.evaluate()
                print(f"eval: {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best", is_best=True)

            # SAVE
            if self.step % self.config.save_interval == 0:
                self.save_checkpoint("latest")

        self.save_checkpoint("final")
        print("=== TRAIN DONE ===")

    # ---------------- EVAL ----------------

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        losses = []

        for i, batch in enumerate(self.val_loader):
            if i > self.config.eval_steps:
                break

            x = batch["input_ids"].to(self.device)
            y = batch["targets"].to(self.device)

            out = self.model(x, y)
            losses.append(out["loss"].item())

        self.model.train()
        return float(np.mean(losses)) if losses else float("nan")


# ---------------- CLI ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="foundation")
    parser.add_argument("--model-size", default="medium")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    size_map = {
        "small": ModelConfig.small,
        "medium": ModelConfig.medium,
        "large": ModelConfig.large,
    }

    model_config = size_map[args.model_size]()
    model = SamuelModel(model_config)

    train_config = TrainingConfig.for_stage(args.stage, device=args.device)

    train_dataset = StreamingSamuelDataset(
        data_dir="data/processed",
        seq_len=model_config.max_seq_len,
        stage=args.stage,
        split="train",
    )

    val_dataset = StreamingSamuelDataset(
        data_dir="data/processed",
        seq_len=model_config.max_seq_len,
        stage=args.stage,
        split="val",
    )

    trainer = Trainer(model, train_config, train_dataset, val_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
