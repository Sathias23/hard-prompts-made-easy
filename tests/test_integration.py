"""
Integration tests for prompt optimization.

These tests verify that our new implementation produces similar results
to the original implementation.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from prompt_optim.optimizers.basic import BasicPromptOptimizer
from prompt_optim.config.optim_config import OptimConfig

def create_test_image(path: Path, color: tuple) -> None:
    """Create a test image with a solid color."""
    size = (224, 224)  # CLIP's expected size
    img = Image.new('RGB', size, color)
    img.save(path)

@pytest.fixture(scope="module")
def test_images(tmp_path_factory):
    """Create a set of test images with different colors."""
    test_dir = tmp_path_factory.mktemp("test_images")
    
    # Create images with different colors
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
    ]
    
    image_paths = []
    for i, color in enumerate(colors):
        path = test_dir / f"test_image_{i}.png"
        create_test_image(path, color)
        image_paths.append(path)
    
    return image_paths

def test_basic_optimization_single_image(test_images):
    """Test basic optimization with a single image."""
    config = OptimConfig(
        num_iterations=100,
        learning_rate=0.1,
        print_new_best=True,
        print_step=20
    )
    
    optimizer = BasicPromptOptimizer(config=config)
    result = optimizer.optimize(test_images[0])
    
    assert isinstance(result, str)
    assert "optimized_prompt" in result

def test_optimization_convergence(test_images):
    """Test that optimization converges by checking loss values."""
    config = OptimConfig(
        num_iterations=200,
        learning_rate=0.1,
        print_new_best=False,
        print_step=None
    )
    
    optimizer = BasicPromptOptimizer(config=config)
    
    # Track losses during optimization
    losses = []
    
    def loss_callback(step: int, loss: float):
        losses.append(loss)
    
    # Add loss tracking to optimizer
    optimizer.optimize(test_images[0], callback=loss_callback)
    
    # Check that loss decreases
    assert len(losses) > 0
    assert losses[-1] < losses[0]  # Final loss should be lower than initial
    
    # Check convergence (loss should stabilize)
    final_losses = losses[-10:]  # Last 10 losses
    assert max(final_losses) - min(final_losses) < 0.1  # Loss should not vary much

def test_multiple_image_optimization(test_images):
    """Test optimization with multiple target images."""
    config = OptimConfig(
        num_iterations=150,
        learning_rate=0.1,
        print_new_best=True,
        print_step=50
    )
    
    optimizer = BasicPromptOptimizer(config=config)
    result = optimizer.optimize(test_images[:2])  # Use first two images
    
    assert isinstance(result, str)
    assert "optimized_prompt" in result

def test_feature_consistency():
    """Test that feature extraction is consistent."""
    # Create two identical images
    test_dir = Path("tests/fixtures")
    img_path1 = test_dir / "identical1.png"
    img_path2 = test_dir / "identical2.png"
    
    create_test_image(img_path1, (128, 128, 128))
    create_test_image(img_path2, (128, 128, 128))
    
    optimizer = BasicPromptOptimizer()
    
    # Extract features from both images
    images1 = optimizer._load_images(img_path1)
    images2 = optimizer._load_images(img_path2)
    
    features1 = optimizer._get_target_features(images1)
    features2 = optimizer._get_target_features(images2)
    
    # Features should be identical
    assert torch.allclose(features1, features2)

def test_device_consistency(test_images):
    """Test that optimization works consistently across devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    config = OptimConfig(num_iterations=50)
    
    # Run on CPU
    cpu_optimizer = BasicPromptOptimizer(device="cpu", config=config)
    cpu_result = cpu_optimizer.optimize(test_images[0])
    
    # Run on GPU
    gpu_optimizer = BasicPromptOptimizer(device="cuda", config=config)
    gpu_result = gpu_optimizer.optimize(test_images[0])
    
    # Results should be similar (not exactly equal due to hardware differences)
    cpu_loss = float(cpu_result.split("_")[-1])
    gpu_loss = float(gpu_result.split("_")[-1])
    
    assert abs(cpu_loss - gpu_loss) < 0.1  # Losses should be reasonably close
