"""Train a custom BPE tokenizer for Samuel.

Why train your own tokenizer:
- Vocabulary is tailored to your training data distribution
- Engineering/robotics terms get proper tokens
- Samuel-specific vocabulary is efficiently encoded
- Smaller vocabulary = fewer parameters in embeddings
"""

import argparse
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


def get_training_corpus(data_paths: list[str], max_texts: int = 500000) -> Iterator[str]:
    """Yield text samples for tokenizer training.

    Pulls from multiple sources to ensure vocabulary covers all domains.
    """
    from datasets import load_dataset

    count = 0

    for path_spec in data_paths:
        if count >= max_texts:
            break

        parts = path_spec.split(":")
        dataset_path = parts[0]
        subset = parts[1] if len(parts) > 1 else None
        text_field = parts[2] if len(parts) > 2 else "text"

        try:
            if subset:
                ds = load_dataset(dataset_path, subset, split="train", streaming=True)
            else:
                ds = load_dataset(dataset_path, split="train", streaming=True)

            for example in ds:
                if count >= max_texts:
                    break

                text = example.get(text_field, "")
                if isinstance(text, str) and len(text) > 50:
                    yield text
                    count += 1
                elif isinstance(text, list):
                    # Handle conversation format (list of messages)
                    combined = " ".join(
                        msg.get("content", str(msg)) if isinstance(msg, dict) else str(msg)
                        for msg in text
                    )
                    if len(combined) > 50:
                        yield combined
                        count += 1
        except Exception as e:
            print(f"Warning: Could not load {dataset_path}: {e}")
            continue

    print(f"Tokenizer training: used {count} texts")


def train_tokenizer(
    vocab_size: int = 32000,
    output_dir: str = "outputs/tokenizer",
    data_paths: list[str] = None,
    max_texts: int = 500000,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a BPE tokenizer from scratch.

    Args:
        vocab_size: Target vocabulary size
        output_dir: Where to save the trained tokenizer
        data_paths: List of dataset paths in format "path:subset:text_field"
        max_texts: Maximum number of texts for training
        min_frequency: Minimum frequency for a token to be included

    Returns:
        Trained Tokenizer instance
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Special tokens that Samuel needs
    special_tokens = [
        "<pad>",       # Padding
        "<s>",         # Beginning of sequence
        "</s>",        # End of sequence
        "<unk>",       # Unknown token
        "<|user|>",    # User turn marker
        "<|samuel|>",  # Samuel's turn marker
        "<|system|>",  # System message marker
        "<|end|>",     # End of turn
    ]

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Pre-tokenization: split on whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor for adding special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Default data sources for tokenizer training
    if data_paths is None:
        data_paths = [
            "roneneldan/TinyStories::text",
            "HuggingFaceH4/ultrachat_200k::messages",
        ]

    print(f"Training BPE tokenizer with vocab_size={vocab_size}...")
    print(f"Using {len(data_paths)} data source(s)")

    # Train
    corpus = get_training_corpus(data_paths, max_texts)
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    # Save
    tokenizer.save(str(output_path / "tokenizer.json"))

    # Save config for reference
    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": special_tokens,
        "data_paths": data_paths,
        "max_texts": max_texts,
    }
    import json
    with open(output_path / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Tokenizer saved to {output_path}")
    print(f"Final vocab size: {tokenizer.get_vocab_size()}")

    # Quick test
    test_texts = [
        "Hello, I am Samuel. How can I help you today?",
        "The Arduino Uno uses an ATmega328P microcontroller.",
        "def train_model(config):\n    model = SamuelModel(config)",
        "PWM stands for Pulse Width Modulation.",
    ]
    print("\nTokenizer test:")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        print(f"  \"{text[:50]}...\" -> {len(encoded.ids)} tokens")

    return tokenizer


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train Samuel's tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--output-dir", type=str, default="outputs/tokenizer")
    parser.add_argument("--max-texts", type=int, default=500000, help="Max training texts")
    parser.add_argument(
        "--data-paths", nargs="+", default=None,
        help="Dataset paths in format 'path:subset:text_field'"
    )
    args = parser.parse_args()

    train_tokenizer(
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        data_paths=args.data_paths,
        max_texts=args.max_texts,
    )


if __name__ == "__main__":
    main()
