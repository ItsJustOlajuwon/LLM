"""Dataset configuration."""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class DatasetSource:
    """A single dataset source with its properties."""

    name: str
    path: str  # HuggingFace dataset path or local path
    subset: Optional[str] = None  # dataset subset/config
    split: str = "train"
    text_field: str = "text"  # field containing the text
    weight: float = 1.0  # sampling weight within its category
    max_samples: Optional[int] = None  # cap on samples to use


@dataclass
class DataConfig:
    """Configuration for the layered dataset strategy."""

    # Category weights (must sum to 1.0)
    general_weight: float = 0.40
    conversation_weight: float = 0.25
    engineering_weight: float = 0.25
    samuel_weight: float = 0.10

    # Processing
    tokenizer_path: str = "outputs/tokenizer"
    max_seq_len: int = 512
    num_workers: int = 4

    # Output
    processed_data_dir: str = "data/processed"

    # Sources per category
    general_sources: list = field(default_factory=lambda: [
        {
            "name": "tinystories",
            "path": "roneneldan/TinyStories",
            "text_field": "text",
            "weight": 0.4,
            "max_samples": 200000,
        },
        {
            "name": "wikipedia",
            "path": "wikimedia/wikipedia",
            "subset": "20231101.en",
            "text_field": "text",
            "weight": 0.4,
            "max_samples": 100000,
        },
        {
            "name": "gutenberg",
            "path": "sedthh/gutenberg_english",
            "text_field": "TEXT",
            "weight": 0.2,
            "max_samples": 20000,
        },
    ])

    conversation_sources: list = field(default_factory=lambda: [
        {
            "name": "ultrachat",
            "path": "HuggingFaceH4/ultrachat_200k",
            "text_field": "messages",
            "weight": 0.6,
            "max_samples": 100000,
        },
        {
            "name": "oasst",
            "path": "OpenAssistant/oasst2",
            "text_field": "text",
            "weight": 0.4,
            "max_samples": 50000,
        },
    ])

    engineering_sources: list = field(default_factory=lambda: [
        {
            "name": "python_code",
            "path": "bigcode/starcoderdata",
            "subset": "python",
            "text_field": "content",
            "weight": 0.4,
            "max_samples": 50000,
        },
        {
            "name": "stack_exchange",
            "path": "HuggingFaceH4/stack-exchange-preferences",
            "text_field": "question",
            "weight": 0.3,
            "max_samples": 50000,
        },
        {
            "name": "arduino_docs",
            "path": "local",  # placeholder for local engineering docs
            "text_field": "text",
            "weight": 0.3,
        },
    ])

    samuel_sources: list = field(default_factory=lambda: [
        {
            "name": "samuel_lore",
            "path": "local",  # user-provided Samuel identity data
            "text_field": "text",
            "weight": 1.0,
        },
    ])

    def get_sources(self, category: str) -> list:
        """Get dataset sources for a category."""
        mapping = {
            "general": self.general_sources,
            "conversation": self.conversation_sources,
            "engineering": self.engineering_sources,
            "samuel": self.samuel_sources,
        }
        if category not in mapping:
            raise ValueError(f"Unknown category: {category}")
        return [DatasetSource(**s) for s in mapping[category]]

    def get_category_weight(self, category: str) -> float:
        """Get the weight for a dataset category."""
        weights = {
            "general": self.general_weight,
            "conversation": self.conversation_weight,
            "engineering": self.engineering_weight,
            "samuel": self.samuel_weight,
        }
        return weights[category]

    def save(self, path: str):
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "DataConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
