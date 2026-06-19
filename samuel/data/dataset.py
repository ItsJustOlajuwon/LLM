"""PyTorch dataset for Samuel training."""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np


class SamuelDataset(Dataset):
    """Memory-mapped dataset for efficient training.

    Reads pre-tokenized data stored as numpy memmap files.
    Each sample is a fixed-length sequence of token IDs.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int,
        split: str = "train",
    ):
        self.seq_len = seq_len
        self.data_path = Path(data_path)
        self.split = split

        mmap_path = self.data_path / f"{split}.bin"
        if not mmap_path.exists():
            raise FileNotFoundError(
                f"Processed data not found at {mmap_path}. "
                f"Run 'samuel-prepare-data' first."
            )

        try:
            self.data = np.memmap(str(mmap_path), dtype=np.uint16, mode="r")
            self.n_tokens = len(self.data)
            self.n_samples = self.n_tokens // seq_len
        except Exception as e:
            raise RuntimeError(f"Failed to load memmap from {mmap_path}: {e}")

        if self.n_samples == 0:
            raise ValueError(
                f"Not enough tokens ({self.n_tokens}) for seq_len={seq_len}. "
                f"Need at least {seq_len + 1} tokens."
            )

        print(f"Loaded {split} split: {self.n_tokens:,} tokens, {self.n_samples:,} samples")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.n_samples})")

        start = idx * self.seq_len
        end = start + self.seq_len + 1

        chunk = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return {"input_ids": x, "targets": y}


class StreamingSamuelDataset(Dataset):
    """Dataset that samples from multiple category files with given weights.

    Used during training:
    - General Language
    - Conversation
    - Engineering
    - Samuel Identity
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        stage: str = "foundation",
        split: str = "train",
    ):
        self.seq_len = seq_len
        self.data_dir = Path(data_dir)
        self.stage = stage

        stage_categories = {
            "foundation": {
                "general": 0.55,
                "engineering": 0.45,
            },
            "instruction": {
                "general": 0.20,
                "conversation": 0.50,
                "engineering": 0.30,
            },
            "samuel": {
                "conversation": 0.30,
                "engineering": 0.20,
                "samuel": 0.50,
            },
        }

        if stage not in stage_categories:
            raise ValueError(f"Unknown stage: {stage}")

        self.categories = stage_categories[stage]
        self.category_data = {}
        self.category_sizes = {}

        total_tokens = 0

        for cat, weight in self.categories.items():
            cat_path = self.data_dir / cat / f"{split}.bin"

            if not cat_path.exists():
                print(f"  [{cat}] Missing: {cat_path}")
                continue

            try:
                data = np.memmap(str(cat_path), dtype=np.uint16, mode="r")
                n_samples = len(data) // seq_len

                if n_samples <= 0:
                    print(f"  [{cat}] Too small for seq_len={seq_len}")
                    continue

                self.category_data[cat] = data
                self.category_sizes[cat] = n_samples
                total_tokens += len(data)

                print(f"  [{cat}] {len(data):,} tokens ({weight:.0%})")

            except Exception as e:
                print(f"  [{cat}] Failed loading: {e}")

        if not self.category_data:
            raise FileNotFoundError(
                f"No valid datasets found in {self.data_dir}"
            )

        available_weight = sum(
            w for c, w in self.categories.items() if c in self.category_data
        )

        self.weights = {
            c: w / available_weight
            for c, w in self.categories.items()
            if c in self.category_data
        }

        self.total_samples = sum(self.category_sizes.values())

        print(f"  Total: {total_tokens:,} tokens, {self.total_samples:,} samples")

        # 🔥 single RNG for speed + stability
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> dict:
        categories = list(self.weights.keys())
        weights = list(self.weights.values())

        cat = self.rng.choice(categories, p=weights)
        data = self.category_data[cat]

        max_start = len(data) - self.seq_len - 1

        if max_start <= 0:
            start = 0
        else:
            start = self.rng.integers(0, max_start)

        chunk = data[start:start + self.seq_len + 1].astype(np.int64)

        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return {"input_ids": x, "targets": y}
