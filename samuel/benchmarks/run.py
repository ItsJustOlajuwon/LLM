"""Benchmark runner for Samuel model evaluation.

Runs the same set of prompts across different model checkpoints
and saves results for comparison.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from samuel.model.transformer import SamuelModel
from samuel.model.generate import generate
from samuel.configs.model_config import ModelConfig


# Standard benchmark prompts — same across all Samuel versions
BENCHMARK_PROMPTS = [
    # Basic interaction
    {"id": "greeting", "prompt": "Hello", "category": "basic"},
    {"id": "how_are_you", "prompt": "How are you?", "category": "basic"},
    {"id": "who_are_you", "prompt": "Who are you?", "category": "identity"},

    # Technical knowledge
    {"id": "servo_motor", "prompt": "What is a servo motor?", "category": "engineering"},
    {"id": "pwm", "prompt": "Explain PWM.", "category": "engineering"},
    {"id": "arduino_blink", "prompt": "How do I blink an LED with Arduino?", "category": "engineering"},
    {"id": "esp32_wifi", "prompt": "How do I connect an ESP32 to WiFi?", "category": "engineering"},

    # Programming
    {"id": "python_list", "prompt": "How do I sort a list in Python?", "category": "programming"},
    {"id": "train_llm", "prompt": "How do I train an LLM?", "category": "programming"},

    # Creative
    {"id": "story", "prompt": "Tell me a story.", "category": "creative"},
    {"id": "project_idea", "prompt": "Give me a robotics project idea.", "category": "creative"},

    # Samuel-specific
    {"id": "itsy", "prompt": "What is Itsy?", "category": "samuel"},
    {"id": "saturn", "prompt": "Tell me about Saturn Robotics.", "category": "samuel"},

    # Reasoning
    {"id": "reasoning_1", "prompt": "If I have 3 servos and each needs 500mA, what power supply do I need?", "category": "reasoning"},
    {"id": "reasoning_2", "prompt": "Why would an LED not light up in a circuit?", "category": "reasoning"},
]


def run_benchmarks(
    checkpoint_path: str,
    tokenizer_path: str,
    output_dir: str = "outputs/benchmarks",
    version_name: str = "samuel-v2",
    max_tokens: int = 200,
    temperature: float = 0.7,
    device: str = "auto",
) -> dict:
    """Run all benchmark prompts and save results.

    Args:
        checkpoint_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer
        output_dir: Where to save benchmark results
        version_name: Name for this Samuel version
        max_tokens: Max tokens per generation
        temperature: Sampling temperature
        device: Device to use

    Returns:
        Dictionary of benchmark results
    """
    from tokenizers import Tokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nSamuel Benchmark — Version: {version_name}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = SamuelModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model: {config.n_layers}L-{config.n_heads}H-{config.d_model}D")
    print(f"Parameters: {model.num_parameters:,}")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")
    print("=" * 60)

    # Run benchmarks
    results = {
        "version": version_name,
        "timestamp": datetime.now().isoformat(),
        "model_config": config.__dict__,
        "parameters": model.num_parameters,
        "checkpoint": str(checkpoint_path),
        "generation_config": {
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        "results": [],
    }

    for benchmark in BENCHMARK_PROMPTS:
        prompt = benchmark["prompt"]

        # Encode
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

        # Generate
        start_time = time.time()
        output_ids = generate(
            model,
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
        )
        gen_time = time.time() - start_time

        # Decode
        generated_ids = output_ids[0].tolist()
        response = tokenizer.decode(generated_ids[len(encoded.ids):])

        result = {
            "id": benchmark["id"],
            "category": benchmark["category"],
            "prompt": prompt,
            "response": response.strip(),
            "tokens_generated": len(generated_ids) - len(encoded.ids),
            "generation_time_s": round(gen_time, 3),
            "tokens_per_second": round((len(generated_ids) - len(encoded.ids)) / gen_time, 1),
        }
        results["results"].append(result)

        # Print
        print(f"\n[{benchmark['category']}] {prompt}")
        print(f"  → {response.strip()[:150]}{'...' if len(response.strip()) > 150 else ''}")
        print(f"  ({result['tokens_generated']} tokens, {gen_time:.2f}s)")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"{version_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Benchmark results saved to: {result_file}")

    # Summary stats
    categories = {}
    for r in results["results"]:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["tokens_per_second"])

    print(f"\nGeneration speed by category:")
    for cat, speeds in sorted(categories.items()):
        avg = sum(speeds) / len(speeds)
        print(f"  {cat}: {avg:.1f} tokens/s (avg)")

    return results


def compare_versions(benchmark_dir: str = "outputs/benchmarks"):
    """Compare benchmark results across Samuel versions."""
    benchmark_path = Path(benchmark_dir)
    if not benchmark_path.exists():
        print("No benchmark results found.")
        return

    results = []
    for f in sorted(benchmark_path.glob("*.json")):
        with open(f) as fp:
            results.append(json.load(fp))

    if not results:
        print("No benchmark results found.")
        return

    print("\nSamuel Version Comparison")
    print("=" * 60)

    for prompt in BENCHMARK_PROMPTS[:5]:  # Show first 5
        print(f"\nPrompt: \"{prompt['prompt']}\"")
        print("-" * 40)
        for result in results:
            version = result["version"]
            for r in result["results"]:
                if r["id"] == prompt["id"]:
                    response = r["response"][:100]
                    print(f"  [{version}] {response}")
                    break


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run Samuel benchmarks")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--tokenizer", type=str, default="outputs/tokenizer/tokenizer.json")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmarks")
    parser.add_argument("--version", type=str, default="samuel-v2")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compare", action="store_true", help="Compare existing results")
    args = parser.parse_args()

    if args.compare:
        compare_versions(args.output_dir)
    else:
        run_benchmarks(
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            version_name=args.version,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device=args.device,
        )


if __name__ == "__main__":
    main()
