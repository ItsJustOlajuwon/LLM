"""Samuel V2 training loop.

Supports:
- Multi-stage training (foundation → instruction → samuel)
- Gradient accumulation
- Learning rate scheduling (cosine with warmup)
- Mixed precision training
- Checkpointing and resumption
- Wandb logging
"""

import argparse
import math
import time
from pathlib import Path
from typing import Optional

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

        # Device setup
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

        # Move model to device
        self.model = self.model.to(self.device)

        # Compile model if requested (PyTorch 2.0+)
        if train_config.compile_model and hasattr(torch, "compile"):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=train_config.batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=(self.device.type == "cuda"),
                drop_last=True,
            )
        else:
            self.val_loader = None

        # Training state
        self.step = 0
        self.best_val_loss = float("inf")
        self.train_losses = []

        # Mixed precision
        self.use_amp = train_config.dtype in ("float16", "bfloat16") and self.device.type == "cuda"
        if self.use_amp:
            dtype = torch.float16 if train_config.dtype == "float16" else torch.bfloat16
            self.scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
            self.amp_dtype = dtype
        else:
            self.scaler = None
            self.amp_dtype = None

        # Wandb
        self.wandb_run = None
        if train_config.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=train_config.wandb_project,
                    name=train_config.wandb_run_name or f"samuel-{train_config.stage}",
                    config={
                        "model": model.config.__dict__,
                        "training": train_config.__dict__,
                    },
                )
            except ImportError:
                print("wandb not installed, skipping logging")

        # Output directories
        self.output_dir = Path(train_config.output_dir)
        self.checkpoint_dir = self.output_dir / train_config.checkpoint_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay only on non-bias parameters."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )

    def _get_lr(self, step: int) -> float:
        """Cosine learning rate schedule with warmup."""
        if step < self.config.warmup_steps:
            # Linear warmup
            return self.config.learning_rate * step / self.config.warmup_steps

        if self.config.lr_schedule == "constant":
            return self.config.learning_rate

        # Cosine decay
        decay_steps = self.config.max_steps - self.config.warmup_steps
        current_step = step - self.config.warmup_steps
        progress = current_step / decay_steps

        if self.config.lr_schedule == "linear":
            return self.config.min_learning_rate + (
                self.config.learning_rate - self.config.min_learning_rate
            ) * (1 - progress)

        # Cosine
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_learning_rate + (
            self.config.learning_rate - self.config.min_learning_rate
        ) * coeff

    def _update_lr(self):
        """Update learning rate for current step."""
        lr = self._get_lr(self.step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation and return average loss."""
        if self.val_loader is None:
            return float("nan")

        self.model.eval()
        losses = []

        for i, batch in enumerate(self.val_loader):
            if i >= self.config.eval_steps:
                break

            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    result = self.model(input_ids, targets)
            else:
                result = self.model(input_ids, targets)

            losses.append(result["loss"].item())

        self.model.train()
        return np.mean(losses) if losses else float("nan")

    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model.config.__dict__,
            "training_config": self.config.__dict__,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses[-100:],  # Last 100 losses
        }

        # Save latest
        path = self.checkpoint_dir / f"checkpoint_step_{self.step}.pt"
        torch.save(checkpoint, path)

        # Save as latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"  New best model saved (val_loss: {self.best_val_loss:.4f})")

    def load_checkpoint(self, path: str):
        """Resume training from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resumed from step {self.step}")

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print(f"Samuel V2 Training — Stage: {self.config.stage}")
        print("=" * 60)
        print(self.model.summary())
        print(f"Effective batch size: {self.config.effective_batch_size}")
        print(f"Max steps: {self.config.max_steps}")
        print(f"Device: {self.device}")
        print("=" * 60 + "\n")

        # Resume if specified
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)

        self.model.train()
        data_iter = iter(self.train_loader)
        running_loss = 0.0
        start_time = time.time()

        while self.step < self.config.max_steps:
            # Get batch (loop back if data exhausted)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Forward pass
            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    result = self.model(input_ids, targets)
                    loss = result["loss"] / self.config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                result = self.model(input_ids, targets)
                loss = result["loss"] / self.config.gradient_accumulation_steps
                loss.backward()

            running_loss += loss.item()

            # Gradient accumulation
            if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

            # Update learning rate
            lr = self._update_lr()
            self.step += 1

            # Logging
            if self.step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (
                    self.config.log_interval
                    * self.config.batch_size
                    * self.model.config.max_seq_len
                    / elapsed
                )
                avg_loss = running_loss / self.config.log_interval * self.config.gradient_accumulation_steps

                print(
                    f"  step {self.step:>6d} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"tokens/s {tokens_per_sec:.0f} | "
                    f"elapsed {elapsed:.1f}s"
                )

                self.train_losses.append(avg_loss)

                if self.wandb_run:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/step": self.step,
                    })

                running_loss = 0.0
                start_time = time.time()

            # Evaluation
            if self.step % self.config.eval_interval == 0:
                val_loss = self.evaluate()
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                print(f"  ─── eval step {self.step}: val_loss={val_loss:.4f} {'(best!)' if is_best else ''}")

                if self.wandb_run:
                    import wandb
                    wandb.log({"val/loss": val_loss, "val/step": self.step})

            # Checkpointing
            if self.step % self.config.save_interval == 0:
                self.save_checkpoint(is_best=(val_loss < self.best_val_loss) if 'val_loss' in dir() else False)

        # Final save
        self.save_checkpoint()
        print(f"\nTraining complete! Final step: {self.step}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        if self.wandb_run:
            self.wandb_run.finish()


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train Samuel V2")
    parser.add_argument(
        "--stage", type=str, default="foundation",
        choices=["foundation", "instruction", "samuel"],
        help="Training stage"
    )
    parser.add_argument("--model-config", type=str, default=None, help="Path to model config YAML")
    parser.add_argument("--train-config", type=str, default=None, help="Path to training config YAML")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Override common settings
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-size", type=str, default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--use-wandb", action="store_true")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Load or create model config
    if args.model_config:
        model_config = ModelConfig.load(args.model_config)
    else:
        size_map = {"small": ModelConfig.small, "medium": ModelConfig.medium, "large": ModelConfig.large}
        model_config = size_map[args.model_size]()

    # Load or create training config
    if args.train_config:
        train_config = TrainingConfig.load(args.train_config)
    else:
        overrides = {"output_dir": args.output_dir, "device": args.device}
        if args.batch_size:
            overrides["batch_size"] = args.batch_size
        if args.max_steps:
            overrides["max_steps"] = args.max_steps
        if args.lr:
            overrides["learning_rate"] = args.lr
        if args.use_wandb:
            overrides["use_wandb"] = True
        if args.resume:
            overrides["resume_from"] = args.resume
        train_config = TrainingConfig.for_stage(args.stage, **overrides)

    # Create model
    model = SamuelModel(model_config)
    print(model.summary())

    # Create datasets
    print(f"\nLoading data for stage: {args.stage}")
    train_dataset = StreamingSamuelDataset(
        data_dir=args.data_dir,
        seq_len=model_config.max_seq_len,
        stage=args.stage,
        split="train",
    )

    val_dataset = StreamingSamuelDataset(
        data_dir=args.data_dir,
        seq_len=model_config.max_seq_len,
        stage=args.stage,
        split="val",
    )

    # Train
    trainer = Trainer(model, train_config, train_dataset, val_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
