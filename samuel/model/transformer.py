"""Samuel transformer model: full decoder-only architecture."""

import torch
import torch.nn as nn
from samuel.configs.model_config import ModelConfig
from samuel.model.components import TransformerBlock, RMSNorm


class SamuelModel(nn.Module):
    """Decoder-only transformer language model for Project Samuel.

    Architecture choices (vs V1):
    - Pre-norm (more stable training)
    - RoPE positional encoding (better length generalization)
    - SwiGLU activation (better performance per param)
    - RMSNorm (faster than LayerNorm)
    - Optional weight tying (parameter efficient)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Learned positional embedding (only if not using RoPE)
        if config.pos_encoding == "learned":
            self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        else:
            self.pos_embedding = None

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm
        if config.norm_type == "rmsnorm":
            self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        else:
            self.final_norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)

        # Output head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> dict:
        """Forward pass.

        Args:
            input_ids: Token IDs, shape (B, T)
            targets: Target token IDs for loss computation, shape (B, T)

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"
        )

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Add learned positional embeddings if applicable
        if self.pos_embedding is not None:
            positions = torch.arange(T, device=input_ids.device)
            x = x + self.pos_embedding(positions)

        x = self.dropout(x)

        # Causal mask
        mask = self._make_causal_mask(T, input_ids.device)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final norm and output projection
        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Compute loss if targets provided
        result = {"logits": logits}
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,  # padding token
            )
            result["loss"] = loss

        return result

    @property
    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_parameters_non_embedding(self) -> int:
        """Count parameters excluding embeddings."""
        n = self.num_parameters
        n -= self.token_embedding.weight.numel()
        return n

    def summary(self) -> str:
        """Print model summary."""
        total = self.num_parameters
        non_emb = self.num_parameters_non_embedding
        return (
            f"Samuel Model Summary:\n"
            f"  Architecture: {self.config.n_layers}L-{self.config.n_heads}H-{self.config.d_model}D\n"
            f"  Total parameters: {total:,} ({total / 1e6:.2f}M)\n"
            f"  Non-embedding parameters: {non_emb:,} ({non_emb / 1e6:.2f}M)\n"
            f"  Vocab size: {self.config.vocab_size:,}\n"
            f"  Max sequence length: {self.config.max_seq_len}\n"
            f"  Position encoding: {self.config.pos_encoding}\n"
            f"  Activation: {self.config.activation}\n"
            f"  Norm: {self.config.norm_type}\n"
            f"  Weight tying: {self.config.tie_embeddings}\n"
        )
