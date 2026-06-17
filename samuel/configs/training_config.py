"""Training configuration."""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class TrainingConfig:
    """Configuration for training loop and optimization."""

    # Training stages
    stage: str = "foundation"  # "foundation" | "instruction" | "samuel"

    # Optimization
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 500
    max_steps: int = 50000

    # Batch
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = field(init=False)

    # Schedule
    lr_schedule: str = "cosine"  # "cosine" | "linear" | "constant"

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 2000
    eval_steps: int = 100

    # Checkpointing
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None

    # Hardware
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    dtype: str = "float32"  # "float32" | "float16" | "bfloat16"
    compile_model: bool = False  # torch.compile (requires PyTorch 2.0+)

    # Wandb
    wandb_project: str = "samuel-v2"
    wandb_run_name: Optional[str] = None
    use_wandb: bool = False

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps

    @classmethod
    def for_stage(cls, stage: str, **overrides) -> "TrainingConfig":
        """Get default config for a training stage."""
        defaults = {
            "foundation": {
                "stage": "foundation",
                "learning_rate": 3e-4,
                "max_steps": 50000,
                "warmup_steps": 500,
            },
            "instruction": {
                "stage": "instruction",
                "learning_rate": 1e-4,
                "max_steps": 20000,
                "warmup_steps": 200,
            },
            "samuel": {
                "stage": "samuel",
                "learning_rate": 5e-5,
                "max_steps": 5000,
                "warmup_steps": 100,
            },
        }
        if stage not in defaults:
            raise ValueError(f"Unknown stage: {stage}. Choose from {list(defaults.keys())}")
        config = defaults[stage]
        config.update(overrides)
        return cls(**config)

    def save(self, path: str):
        """Save config to YAML file."""
        data = {k: v for k, v in self.__dict__.items() if k != "effective_batch_size"}
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        data.pop("effective_batch_size", None)
        return cls(**data)
