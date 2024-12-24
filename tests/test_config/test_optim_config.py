"""
Tests for the OptimConfig class.
"""

import pytest
import json
from pathlib import Path
from prompt_optim.config.optim_config import OptimConfig

def test_default_config():
    """Test default configuration values."""
    config = OptimConfig()
    assert config.clip_model == "ViT-B/32"
    assert config.clip_pretrain == "openai"
    assert config.learning_rate == 0.1
    assert config.num_iterations == 1000
    assert config.prompt_length == 8

def test_config_update():
    """Test configuration update method."""
    config = OptimConfig()
    config.update(
        clip_model="ViT-L/14",
        learning_rate=0.2
    )
    assert config.clip_model == "ViT-L/14"
    assert config.learning_rate == 0.2
    # Other values should remain default
    assert config.clip_pretrain == "openai"

def test_invalid_update():
    """Test handling of invalid configuration parameters."""
    config = OptimConfig()
    with pytest.raises(ValueError):
        config.update(invalid_param=123)

def test_config_file_io(tmp_path):
    """Test saving and loading configuration from file."""
    config = OptimConfig()
    config.update(clip_model="ViT-L/14", learning_rate=0.2)
    
    # Save config
    config_path = tmp_path / "test_config.json"
    config.to_file(config_path)
    
    # Load config
    loaded_config = OptimConfig.from_file(config_path)
    
    # Check values match
    assert loaded_config.clip_model == config.clip_model
    assert loaded_config.learning_rate == config.learning_rate
    assert loaded_config.num_iterations == config.num_iterations
