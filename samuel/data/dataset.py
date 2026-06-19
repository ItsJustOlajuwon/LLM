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

        # Load memory-mapped data
        mmap_path = self.data_path / f"{split}.bin"
        if not mmap_path.exists():
            raise FileNotFoundError(
                f"Processed data not found at {mmap_path}. "
                f"Run 'samuel-prepare-data' first."
            )

        # Memory map the file for efficient random access
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
        end = start + self.seq_len + 1  # +1 for target shift

        try:
            chunk = self.data[start:end].astype(np.int64)
            x = torch.from_numpy(chunk[:-1])
            y = torch.from_numpy(chunk[1:])

            return {"input_ids": x, "targets": y}
        except Exception as e:
            raise RuntimeError(f"Error loading sample {idx}: {e}")


class StreamingSamuelDataset(Dataset):
    """Dataset that samples from multiple category files with given weights.

    Used during training to implement the layered dataset strategy:
    - General Language: 40%
    - Conversation: 25%
    - Engineering: 25%
    - Samuel Identity: 10%
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

        # Define which categories are used in each stage
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

        # Load available categories
        total_tokens = 0
        for cat, weight in self.categories.items():
            cat_path = self.data_dir / cat / f"{split}.bin"
            if cat_path.exists():
                try:
                    data = np.memmap(str(cat_path), dtype=np.uint16, mode="r")
                    n_samples = len(data) // seq_len
                    if n_samples > 0:
                        self.category_data[cat] = data
                        self.category_sizes[cat] = n_samples
                        total_tokens += len(data)
                        print(f"  [{cat}] {len(data):,} tokens ({weight:.0%} weight)")
                    else:
                        print(f"  [{cat}] Insufficient tokens ({len(data)}) for seq_len={seq_len}, skipping")
                except Exception as e:
                    print(f"  [{cat}] Error loading {cat_path}: {e}, skipping")
            else:
                print(f"  [{cat}] Not found at {cat_path}, skipping")

        if not self.category_data:
            raise FileNotFoundError(
                f"No processed data found in {data_dir}. Run 'samuel-prepare-data' first."
            )

        # Normalize weights to available categories
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

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> dict:
        # Sample a category based on weights
        categories = list(self.weights.keys())
        weights = list(self.weights.values())
        cat = np.random.choice(categories, p=weights)

        # Random sample from the chosen category
        data = self.category_data[cat]
        max_start = len(data) - self.seq_len - 1
        if max_start <= 0:
            start = 0
        else:
            rng = np.random.default_rng()
            start = int(rng.integers(0, max_start, dtype=np.int64))

        try:
            chunk = data[start: start + self.seq_len + 1].astype(np.int64)
            x = torch.from_numpy(chunk[:-1])
            y = torch.from_numpy(chunk[1:])

            return {"input_ids": x, "targets": y}
        except Exception as e:
            raise RuntimeError(f"Error loading sample from {cat}: {e}")
