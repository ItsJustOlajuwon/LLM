#!/usr/bin/env python3
"""Quick validation script to verify the Samuel V2 codebase works.

Run this after installation to confirm everything is set up correctly.
"""

import sys
import torch
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print()

# Test 1: Model creation
print("=" * 50)
print("Test 1: Create model")
print("=" * 50)
from samuel.configs.model_config import ModelConfig
from samuel.model.transformer import SamuelModel

for size_name, factory in [("small", ModelConfig.small), ("medium", ModelConfig.medium)]:
    config = factory()
    model = SamuelModel(config)
    print(f"\n{size_name.upper()} model:")
    print(model.summary())

# Test 2: Forward pass
print("=" * 50)
print("Test 2: Forward pass")
print("=" * 50)
config = ModelConfig.small()
model = SamuelModel(config)

batch_size = 2
seq_len = 64
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

result = model(input_ids, targets)
print(f"Input shape: {input_ids.shape}")
print(f"Logits shape: {result['logits'].shape}")
print(f"Loss: {result['loss'].item():.4f}")
print("Forward pass: OK")

# Test 3: Generation
print("\n" + "=" * 50)
print("Test 3: Generation")
print("=" * 50)
from samuel.model.generate import generate

input_ids = torch.randint(0, config.vocab_size, (1, 10))
output = generate(model, input_ids, max_new_tokens=20, temperature=0.8)
print(f"Input tokens: {input_ids.shape[1]}")
print(f"Output tokens: {output.shape[1]}")
print(f"Generated {output.shape[1] - input_ids.shape[1]} new tokens")
print("Generation: OK")

# Test 4: Config save/load
print("\n" + "=" * 50)
print("Test 4: Config serialization")
print("=" * 50)
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "test_config.yaml"
    config.save(str(path))
    loaded = ModelConfig.load(str(path))
    assert loaded.d_model == config.d_model
    assert loaded.n_layers == config.n_layers
    print(f"Saved and loaded config: OK")

# Test 5: Training step
print("\n" + "=" * 50)
print("Test 5: Training step")
print("=" * 50)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

train_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
train_target = torch.randint(0, config.vocab_size, (batch_size, seq_len))
result = model(train_input, train_target)
loss = result["loss"]
loss.backward()
optimizer.step()
optimizer.zero_grad()
print(f"Training step loss: {loss.item():.4f}")
print("Training step: OK")

print("\n" + "=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
print("\nSamuel V2 codebase is ready!")
print("Next steps:")
print("  1. Train tokenizer: samuel-train-tokenizer")
print("  2. Prepare data:    samuel-prepare-data")
print("  3. Train model:     samuel-train --stage foundation")
