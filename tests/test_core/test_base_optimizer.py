"""
Tests for the base PromptOptimizer class.
"""

import pytest
import torch
from pathlib import Path
from prompt_optim.core.base_optimizer import PromptOptimizer

def test_optimizer_init(device):
    """Test basic initialization of optimizer."""
    optimizer = PromptOptimizer(device=device)
    assert optimizer.model_name == "ViT-B/32"
    assert optimizer.pretrained == "openai"
    assert optimizer.device == device
    assert optimizer.model is not None
    assert optimizer.preprocess is not None

def test_custom_model():
    """Test initialization with custom model."""
    optimizer = PromptOptimizer(
        model_name="ViT-L/14",
        pretrained="openai"
    )
    assert optimizer.model_name == "ViT-L/14"
    assert optimizer.pretrained == "openai"

def test_device_auto_detection():
    """Test automatic device detection."""
    optimizer = PromptOptimizer()
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert optimizer.device == expected_device

def test_optimize_not_implemented():
    """Test that base optimize method raises NotImplementedError."""
    optimizer = PromptOptimizer()
    with pytest.raises(NotImplementedError):
        optimizer.optimize(target_images=["test.jpg"])

class TestOptimizer(PromptOptimizer):
    """Test implementation of PromptOptimizer."""
    def optimize(self, target_images=None, target_prompts=None, **kwargs):
        return "test prompt"

def test_subclass_implementation(device):
    """Test that subclass can implement optimize method."""
    optimizer = TestOptimizer(device=device)
    result = optimizer.optimize()
    assert isinstance(result, str)
    assert result == "test prompt"

def test_model_on_correct_device(device):
    """Test that model is on the correct device."""
    optimizer = PromptOptimizer(device=device)
    # Check if at least one parameter is on the correct device
    param = next(optimizer.model.parameters())
    assert str(param.device) == device
