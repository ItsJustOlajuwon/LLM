"""Prepare and tokenize datasets for Samuel training.

This script:
1. Downloads datasets from HuggingFace
2. Processes them into text
3. Tokenizes with the Samuel tokenizer
4. Saves as memory-mapped numpy arrays for efficient training
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer


def load_and_process_source(
    source_config: dict,
    max_samples: int = None,
) -> Iterator[str]:
    """Load a dataset source and yield processed text samples."""
    from datasets import load_dataset

    name = source_config["name"]
    path = source_config["path"]
    subset = source_config.get("subset")
    text_field = source_config.get("text_field", "text")
    max_s = max_samples or source_config.get("max_samples")

    if path == "local":
        # Local data source — look for it in data/sources/{name}/
        local_path = Path(f"data/sources/{name}")
        if local_path.exists():
            for txt_file in sorted(local_path.glob("*.txt")):
                with open(txt_file) as f:
                    yield f.read()
            for jsonl_file in sorted(local_path.glob("*.jsonl")):
                with open(jsonl_file) as f:
                    for line in f:
                        data = json.loads(line)
                        yield data.get("text", data.get("content", json.dumps(data)))
        else:
            print(f"  Warning: Local source '{name}' not found at {local_path}")
        return

    print(f"  Loading {name} from {path}...")
    try:
        if subset:
            ds = load_dataset(path, subset, split="train", streaming=True)
        else:
            ds = load_dataset(path, split="train", streaming=True)
    except Exception as e:
        print(f"  Error loading {name}: {e}")
        return

    count = 0
    for example in ds:
        if max_s and count >= max_s:
            break

        text = example.get(text_field, "")

        # Handle different data formats
        if isinstance(text, list):
            # Conversation format (list of messages)
            parts = []
            for msg in text:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "user":
                        parts.append(f"<|user|>{content}<|end|>")
                    elif role == "assistant":
                        parts.append(f"<|samuel|>{content}<|end|>")
                    else:
                        parts.append(f"<|system|>{content}<|end|>")
                else:
                    parts.append(str(msg))
            text = "\n".join(parts)
        elif not isinstance(text, str):
            text = str(text)

        if len(text) > 50:
            yield text
            count += 1

    print(f"  {name}: yielded {count} samples")


def tokenize_and_save(
    texts: Iterator[str],
    tokenizer: Tokenizer,
    output_path: Path,
    seq_len: int,
    split: str = "train",
    max_tokens: int = None,
):
    """Tokenize texts and save as memory-mapped binary file.

    Uses chunked writing to avoid loading all tokens into memory at once.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    bin_path = output_path / f"{split}.bin"
    tmp_path = output_path / f"{split}.tmp.bin"

    # Write tokens in chunks to a temporary file to avoid MemoryError
    CHUNK_SIZE = 500_000  # flush every 500k tokens
    chunk = []
    total = 0

    with open(tmp_path, "wb") as f:
        for text in texts:
            encoded = tokenizer.encode(text)
            tokens = encoded.ids
            chunk.extend(tokens)
            total += len(tokens)

            # Flush chunk to disk
            if len(chunk) >= CHUNK_SIZE:
                arr = np.array(chunk, dtype=np.uint16)
                f.write(arr.tobytes())
                chunk = []

                # Progress indicator
                if total % 5_000_000 == 0:
                    print(f"    ... {total:,} tokens processed")

            if max_tokens and total >= max_tokens:
                # Trim to max
                chunk = chunk[:max(0, max_tokens - (total - len(chunk)))]
                total = max_tokens
                break

        # Write remaining chunk
        if chunk:
            arr = np.array(chunk, dtype=np.uint16)
            f.write(arr.tobytes())

    if total == 0:
        print(f"  Warning: No tokens generated for {output_path}")
        tmp_path.unlink(missing_ok=True)
        return 0

    # Rename tmp to final
    if bin_path.exists():
        bin_path.unlink()
    tmp_path.rename(bin_path)

    print(f"  Saved {total:,} tokens to {bin_path}")
    return total


def prepare_dataset(
    tokenizer_path: str = "outputs/tokenizer/tokenizer.json",
    output_dir: str = "data/processed",
    config_path: str = None,
    seq_len: int = 512,
    val_ratio: float = 0.05,
):
    """Prepare all datasets for training.

    Args:
        tokenizer_path: Path to trained tokenizer
        output_dir: Where to save processed data
        config_path: Path to data config YAML (optional)
        seq_len: Sequence length for training
        val_ratio: Fraction of data to use for validation
    """
    from samuel.configs.data_config import DataConfig

    print("=" * 60)
    print("Samuel V2 — Data Preparation")
    print("=" * 60)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"Loaded tokenizer: {tokenizer.get_vocab_size()} tokens")

    # Load data config
    if config_path:
        config = DataConfig.load(config_path)
    else:
        config = DataConfig()

    output_path = Path(output_dir)
    categories = ["general", "conversation", "engineering", "samuel"]
    stats = {}

    for category in categories:
        print(f"\n{'─' * 40}")
        print(f"Processing: {category} (weight: {config.get_category_weight(category):.0%})")
        print(f"{'─' * 40}")

        sources = config.get_sources(category)
        cat_output = output_path / category

        # Collect all texts from sources in this category
        def category_texts():
            for source in sources:
                source_dict = {
                    "name": source.name,
                    "path": source.path,
                    "subset": source.subset,
                    "text_field": source.text_field,
                    "max_samples": source.max_samples,
                }
                yield from load_and_process_source(source_dict)

        # Tokenize and save
        n_tokens = tokenize_and_save(
            texts=category_texts(),
            tokenizer=tokenizer,
            output_path=cat_output,
            seq_len=seq_len,
            split="train",
        )
        stats[category] = n_tokens

        # Create a small validation split by taking the last val_ratio of tokens
        if n_tokens > 0:
            train_path = cat_output / "train.bin"
            file_size = train_path.stat().st_size
            total_tokens = file_size // 2  # uint16 = 2 bytes per token
            val_size = int(total_tokens * val_ratio)

            if val_size > seq_len:
                val_path = cat_output / "val.bin"
                train_size = total_tokens - val_size

                # Read only the val portion from the end of the file
                val_offset = train_size * 2  # byte offset
                with open(train_path, "rb") as f:
                    f.seek(val_offset)
                    val_bytes = f.read()

                # Write val split
                with open(val_path, "wb") as f:
                    f.write(val_bytes)

                # Truncate train file to remove val portion
                with open(train_path, "r+b") as f:
                    f.truncate(val_offset)

                print(f"  Split: {train_size:,} train, {val_size:,} val tokens")

    # Summary
    print(f"\n{'=' * 60}")
    print("Data Preparation Complete")
    print(f"{'=' * 60}")
    total = sum(stats.values())
    for cat, n in stats.items():
        pct = n / total * 100 if total > 0 else 0
        print(f"  {cat:15s}: {n:>12,} tokens ({pct:.1f}%)")
    print(f"  {'Total':15s}: {total:>12,} tokens")
    print(f"\nProcessed data saved to: {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Prepare Samuel training data")
    parser.add_argument("--tokenizer", type=str, default="outputs/tokenizer/tokenizer.json")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--config", type=str, default=None, help="Path to data config YAML")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    prepare_dataset(
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        config_path=args.config,
        seq_len=args.seq_len,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
