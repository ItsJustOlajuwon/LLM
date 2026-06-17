"""Model architecture configuration."""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    """Configuration for the Samuel transformer model.

    Default config produces ~25M parameters, a step up from V1's 16.86M.
    All values are configurable for experimentation across Samuel versions.
    """

    # Architecture
    vocab_size: int = 32000
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 8
    d_ff: int = 1024  # Feed-forward inner dimension
    max_seq_len: int = 512
    dropout: float = 0.1

    # Positional encoding
    pos_encoding: str = "rotary"  # "rotary" | "learned" | "sinusoidal"

    # Normalization
    norm_type: str = "rmsnorm"  # "rmsnorm" | "layernorm"
    norm_eps: float = 1e-6

    # Activation
    activation: str = "swiglu"  # "swiglu" | "gelu" | "relu"

    # Optional
    tie_embeddings: bool = True  # Tie input/output embeddings (saves params)
    bias: bool = False  # Use bias in linear layers

    # Computed properties
    d_head: int = field(init=False)

    def __post_init__(self):
        self.d_head = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )

    @property
    def num_parameters_estimate(self) -> int:
        """Rough estimate of total parameters."""
        embed = self.vocab_size * self.d_model
        attn_per_layer = 4 * self.d_model * self.d_model  # Q, K, V, O projections
        if self.activation == "swiglu":
            ff_per_layer = 3 * self.d_model * self.d_ff  # gate, up, down
        else:
            ff_per_layer = 2 * self.d_model * self.d_ff
        norm_per_layer = 2 * self.d_model
        layers = self.n_layers * (attn_per_layer + ff_per_layer + norm_per_layer)
        output = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        return embed + layers + output

    def save(self, path: str):
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        data.pop("d_head", None)  # computed field
        return cls(**data)

    @classmethod
    def small(cls) -> "ModelConfig":
        """Small config for quick experiments (~10M params)."""
        return cls(d_model=256, n_heads=4, n_layers=6, d_ff=1024, max_seq_len=256)

    @classmethod
    def medium(cls) -> "ModelConfig":
        """Medium config — default Samuel V2 (~26M params)."""
        return cls()

    @classmethod
    def large(cls) -> "ModelConfig":
        """Larger config for GPU training (~50M params)."""
        return cls(d_model=512, n_heads=8, n_layers=12, d_ff=2048, max_seq_len=1024)
