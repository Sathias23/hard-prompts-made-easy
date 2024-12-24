"""
Tests for the BasicPromptOptimizer.
"""

import pytest
import torch
from pathlib import Path
from prompt_optim.optimizers.basic import BasicPromptOptimizer
from prompt_optim.config.optim_config import OptimConfig
from PIL import Image
import numpy as np

def test_basic_optimizer_init():
    """Test basic initialization."""
    optimizer = BasicPromptOptimizer()
    assert isinstance(optimizer.config, OptimConfig)

def test_custom_config():
    """Test initialization with custom config."""
    config = OptimConfig(
        learning_rate=0.2,
        num_iterations=500
    )
    optimizer = BasicPromptOptimizer(config=config)
    assert optimizer.config.learning_rate == 0.2
    assert optimizer.config.num_iterations == 500

def test_load_single_image(test_image_path):
    """Test loading a single image."""
    optimizer = BasicPromptOptimizer()
    images = optimizer._load_images(test_image_path)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].mode == "RGB"

def test_load_multiple_images(tmp_path):
    """Test loading multiple images."""
    # Create test images
    import numpy as np
    from PIL import Image
    
    # Create two test images
    img1_path = tmp_path / "test_image1.png"
    img2_path = tmp_path / "test_image2.png"
    
    for path in [img1_path, img2_path]:
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(path)
    
    optimizer = BasicPromptOptimizer()
    images = optimizer._load_images([img1_path, img2_path])
    assert len(images) == 2
    for img in images:
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

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
    result = optimizer.optimize(test_images[0], callback=loss_callback)
    
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
    assert len(result) > 0  # Should return a non-empty prompt

def test_feature_consistency():
    """Test that feature extraction is consistent."""
    # Create two identical images
    test_dir = Path("tests/fixtures")
    img_path1 = test_dir / "identical1.png"
    img_path2 = test_dir / "identical2.png"
    
    # Create test images
    import numpy as np
    from PIL import Image
    
    img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128)
    img.save(img_path1)
    img.save(img_path2)
    
    optimizer = BasicPromptOptimizer()
    
    # Optimize with each image
    result1 = optimizer.optimize(img_path1)
    result2 = optimizer.optimize(img_path2)
    
    # Results should be identical since the images are identical
    assert result1 == result2

def test_device_consistency(test_images):
    """Test that optimization works consistently across devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    config = OptimConfig(
        num_iterations=50,
        seed=42  # Use same seed for reproducibility
    )
    
    # Run on CPU
    cpu_optimizer = BasicPromptOptimizer(device="cpu", config=config)
    cpu_result = cpu_optimizer.optimize(test_images[0])
    
    # Run on GPU
    gpu_optimizer = BasicPromptOptimizer(device="cuda", config=config)
    gpu_result = gpu_optimizer.optimize(test_images[0])
    
    # Results should be identical since we use the same seed
    assert cpu_result == gpu_result
