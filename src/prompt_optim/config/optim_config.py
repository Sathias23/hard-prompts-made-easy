"""
Optimization Configuration

Handles all configuration parameters for prompt optimization.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import json
from pathlib import Path

@dataclass
class OptimConfig:
    """Configuration for prompt optimization."""
    
    # Model settings
    clip_model: str = "ViT-B/32"
    clip_pretrain: str = "openai"
    device: Optional[str] = None
    
    # Optimization settings
    learning_rate: float = 0.1
    num_iterations: int = 1000
    prompt_length: int = 8
    weight_decay: float = 0.0
    seed: int = 42  # Random seed for reproducibility
    
    # Output settings
    print_step: Optional[int] = 100
    print_new_best: bool = True
    
    # Advanced settings
    initialization_cfg: Dict[str, Any] = field(default_factory=dict)
    optimization_cfg: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "OptimConfig":
        """Load configuration from a JSON file."""
        with open(config_path) as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
