# Project Samuel V2

**Building an AI Brother From Scratch**

Samuel is a decoder-only transformer language model trained from scratch — not fine-tuned from an existing model. The goal is to create an AI "brother" that grows through experimentation, iteration, and training across multiple generations.

## Architecture

Samuel V2 uses a modern transformer decoder architecture:

| Component | Choice | Why |
|-----------|--------|-----|
| Normalization | RMSNorm (pre-norm) | Faster + more stable training |
| Position encoding | Rotary (RoPE) | Better length generalization |
| Activation | SwiGLU | Better performance per parameter |
| Attention | Multi-head causal | Standard autoregressive LM |
| Weight tying | Input/output embeddings shared | Parameter efficient |

### Model Sizes

| Config | Params | Layers | Heads | d_model | d_ff | Context |
|--------|--------|--------|-------|---------|------|---------|
| Small | ~14M | 6 | 4 | 256 | 1024 | 256 |
| **Medium** | **~26M** | **8** | **6** | **384** | **1024** | **512** |
| Large | ~50M | 12 | 8 | 512 | 2048 | 1024 |

## Training Strategy

Training happens in three stages:

### Stage 1: Foundation
- **Data**: General language (55%) + Engineering knowledge (45%)
- **Goal**: Learn English and technical knowledge
- **LR**: 3e-4, cosine decay

### Stage 2: Instruction Tuning
- **Data**: Conversation (50%) + Engineering (30%) + General (20%)
- **Goal**: Learn to interact with users
- **LR**: 1e-4, cosine decay (resumed from Stage 1)

### Stage 3: Samuel Fine-Tuning
- **Data**: Samuel identity (50%) + Conversation (30%) + Engineering (20%)
- **Goal**: Develop Samuel's personality
- **LR**: 5e-5, cosine decay (resumed from Stage 2)

## Dataset Strategy

| Category | Weight | Sources |
|----------|--------|---------|
| General Language | 40% | TinyStories, Wikipedia, Gutenberg |
| Conversation | 25% | UltraChat, OASST |
| Engineering | 25% | Python code, Stack Exchange, Arduino/ESP32 docs |
| Samuel Identity | 10% | Custom lore, conversations, project docs |

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Train Tokenizer

```bash
samuel-train-tokenizer --vocab-size 32000 --max-texts 100000
```

### 3. Prepare Data

```bash
samuel-prepare-data --tokenizer outputs/tokenizer/tokenizer.json --seq-len 512
```

### 4. Train (Stage 1: Foundation)

```bash
samuel-train --stage foundation --model-size medium --device auto
```

### 5. Train (Stage 2: Instruction)

```bash
samuel-train --stage instruction --resume checkpoints/foundation/best.pt
```

### 6. Train (Stage 3: Samuel)

```bash
samuel-train --stage samuel --resume checkpoints/instruction/best.pt
```

### 7. Generate Text

```bash
samuel-generate --checkpoint checkpoints/samuel/best.pt --prompt "Hello Samuel"
```

### 8. Run Benchmarks

```bash
samuel-benchmark --checkpoint checkpoints/samuel/best.pt --version samuel-v2
```

## CPU Training Tips

Since V1 was trained on CPU, here are tips for V2:

1. **Use the small model** (`--model-size small`) for initial experiments
2. **Reduce batch size** (`--batch-size 8`) to fit in RAM
3. **Reduce max-steps** for quick iteration cycles
4. **Use float32** (default) — mixed precision only helps on GPU
5. **Skip torch.compile** — it has overhead on CPU

For CPU training, a reasonable first run:
```bash
samuel-train --stage foundation --model-size small --batch-size 8 --max-steps 5000
```

## Project Structure

```
samuel-v2/
├── samuel/
│   ├── model/          # Transformer architecture
│   │   ├── components.py   # Attention, FFN, RoPE, norms
│   │   ├── transformer.py  # Full model
│   │   └── generate.py     # Text generation
│   ├── tokenizer/      # BPE tokenizer training
│   │   └── train.py
│   ├── data/           # Dataset loading & preparation
│   │   ├── prepare.py      # Download & tokenize datasets
│   │   └── dataset.py      # PyTorch Dataset classes
│   ├── training/       # Training loop
│   │   └── train.py
│   ├── benchmarks/     # Cross-version evaluation
│   │   └── run.py
│   ├── configs/        # Configuration dataclasses
│   │   ├── model_config.py
│   │   ├── training_config.py
│   │   └── data_config.py
│   └── utils/          # Helpers
├── configs/            # YAML config files
│   ├── model_medium.yaml
│   ├── model_small.yaml
│   ├── train_foundation.yaml
│   ├── train_instruction.yaml
│   └── train_samuel.yaml
├── data/
│   └── sources/
│       └── samuel_lore/    # Samuel identity data (add your own!)
├── pyproject.toml
└── README.md
```

## Benchmarking

Every Samuel version answers the same benchmark prompts:

- "Hello" / "How are you?" / "Who are you?"
- "What is a servo motor?" / "Explain PWM."
- "Tell me a story."
- "What is Itsy?"
- "How do I train an LLM?"

Results are saved as JSON for cross-version comparison:
```bash
samuel-benchmark --compare  # Compare all saved benchmark results
```

## Philosophy

```
Build → Measure → Compare → Improve
```

Each version of Samuel is a benchmark for future versions. Failed experiments are not deleted — they become part of Samuel's family tree.

## Samuel V1 Baseline

| Metric | Value |
|--------|-------|
| Parameters | 16.86M |
| Architecture | Transformer (basic) |
| Dataset | UltraChat_200k |
| Initial loss | ~9.71 |
| Final loss | ~4.54 |
| Training | CPU |

## What's New in V2

- [x] Modern architecture (RoPE, SwiGLU, RMSNorm)
- [x] Custom BPE tokenizer (vs. borrowed tokenizer)
- [x] Layered dataset strategy (4 categories)
- [x] Multi-stage training pipeline
- [x] Benchmarking system for version comparison
- [x] Configurable model sizes (experiment-friendly)
- [x] Proper gradient accumulation
- [x] Learning rate scheduling with warmup
- [ ] GPU support with mixed precision (ready when you have a GPU)
- [ ] Wandb integration for experiment tracking

## License

MIT
