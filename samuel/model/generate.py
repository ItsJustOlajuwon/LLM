"""Text generation with Samuel model."""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from samuel.model.transformer import SamuelModel
from samuel.configs.model_config import ModelConfig


@torch.no_grad()
def generate(
    model: SamuelModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> torch.Tensor:
    """Generate text tokens autoregressively.

    Args:
        model: The Samuel model
        input_ids: Starting token IDs, shape (1, T)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_k: Keep only top-k tokens for sampling
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeating tokens

    Returns:
        Generated token IDs, shape (1, T + max_new_tokens)
    """
    model.eval()
    max_seq_len = model.config.max_seq_len

    for _ in range(max_new_tokens):
        # Crop input to max sequence length
        idx_cond = input_ids if input_ids.shape[1] <= max_seq_len else input_ids[:, -max_seq_len:]

        # Forward pass
        result = model(idx_cond)
        logits = result["logits"][:, -1, :]  # (B, vocab_size)

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(input_ids[0].tolist()):
                logits[0, token_id] /= repetition_penalty

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        else:
            # Greedy decoding
            next_token = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            continue

        # Top-k filtering
        if top_k > 0:
            top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_k_values[:, [-1]]] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids


def main():
    """CLI entry point for text generation."""
    parser = argparse.ArgumentParser(description="Generate text with Samuel")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Hello", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model config and create model
    config = ModelConfig(**checkpoint["model_config"])
    model = SamuelModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = Path("outputs/tokenizer/tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Encode prompt
    encoded = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    print(f"\nSamuel V2 — Generating from: \"{args.prompt}\"")
    print("-" * 50)

    # Generate
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # Decode
    generated_ids = output_ids[0].tolist()
    text = tokenizer.decode(generated_ids)
    print(text)
    print("-" * 50)


if __name__ == "__main__":
    main()
