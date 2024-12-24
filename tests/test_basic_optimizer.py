"""Tests for the BasicPromptOptimizer class."""

import pytest
import torch
from pathlib import Path
from PIL import Image
import tempfile
import os
import sys
import io
import numpy as np

from prompt_optim.optimizers.basic import BasicPromptOptimizer
from prompt_optim.config.optim_config import OptimConfig

@pytest.fixture
def test_image():
    """Create a test image for optimization."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='red')
        img.save(f.name)
        img.close()  # Close the image to avoid file handle issues
        path = Path(f.name)
        yield path
        # Cleanup with error handling
        try:
            if path.exists():
                os.remove(path)
        except Exception as e:
            pytest.skip(f"Could not cleanup test file: {e}")

@pytest.fixture
def optimizer():
    """Create a basic optimizer instance."""
    config = OptimConfig(
        prompt_length=8,
        num_iterations=2,  # Small number for testing
        learning_rate=0.1,
        weight_decay=0.0,
        print_step=1,
        print_new_best=True,
        seed=42
    )
    return BasicPromptOptimizer(config=config)

def test_optimizer_initialization(optimizer):
    """Test that the optimizer initializes correctly."""
    assert optimizer.model is not None
    assert optimizer.tokenizer is not None
    assert optimizer.device in ['cuda', 'cpu']
    assert optimizer.config is not None

def test_load_images(optimizer, test_image):
    """Test image loading functionality."""
    images = optimizer._load_images(test_image)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].mode == 'RGB'

def test_optimize_single_image(optimizer, test_image):
    """Test optimization with a single image."""
    result = optimizer.optimize(target_images=test_image)
    assert isinstance(result, str)
    assert len(result) > 0

def test_optimize_multiple_images(optimizer, test_image):
    """Test optimization with multiple copies of the same image."""
    result = optimizer.optimize(
        target_images=[test_image, test_image]  # Use same image twice for testing
    )
    assert isinstance(result, str)
    assert len(result) > 0

def test_optimize_invalid_input(optimizer):
    """Test optimizer behavior with invalid inputs."""
    with pytest.raises(ValueError):
        optimizer.optimize()  # No images provided
        
    with pytest.raises(NotImplementedError):
        optimizer.optimize(target_prompts=["test prompt"])  # Text optimization not implemented

def test_optimize_with_config_override(optimizer, test_image):
    """Test optimization with parameter overrides."""
    result = optimizer.optimize(
        target_images=test_image,
        prompt_len=12,  # Override default length
        lr=0.2  # Override default learning rate
    )
    assert isinstance(result, str)
    assert len(result) > 0

def test_optimize_real_image():
    """Test optimization with a real image using production-like settings."""
    config = OptimConfig(
        prompt_length=75,  # Target slightly under SDXL's 77 token limit
        num_iterations=500,  # Full optimization
        learning_rate=0.1,
        weight_decay=0.1,
        print_step=100,  # More frequent updates
        print_new_best=True,
        seed=42
    )
    
    # Initialize optimizer with the same model as run.py
    optimizer = BasicPromptOptimizer(
        model_name="ViT-H-14",  # Same as sample_config.json
        pretrained="laion2b_s32b_b79k",
        config=config
    )
    
    # Use the test fixture image
    image_path = Path(__file__).parent / "fixtures" / "F2cOgMma0AA0NyS.jpg"
    assert image_path.exists(), f"Test image not found at {image_path}"
    
    # Redirect stdout to stderr which handles Unicode better
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    
    try:
        result = optimizer.optimize(target_images=image_path)
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"\nOptimized prompt for test image: {result}", file=sys.stderr)
        
        # Print token count and raw string representation
        tokens = optimizer.tokenizer(result)
        token_ids = [t for t in tokens[0] if t != 0]  # Filter out padding
        print(f"\nToken count: {len(token_ids)}", file=sys.stderr)
        print(f"Tokens: {token_ids}", file=sys.stderr)
        
        # Print the prompt with preserved Unicode
        print("\nPrompt with preserved Unicode:", file=sys.stderr)
        print(repr(result)[1:-1], file=sys.stderr)  # Remove quotes but keep escapes
    finally:
        sys.stdout = old_stdout

def test_optimize_multiple_images():
    """Test optimization with multiple images using production-like settings."""
    config = OptimConfig(
        prompt_length=40,  # Much more conservative limit
        num_iterations=500,
        learning_rate=0.1,
        weight_decay=0.1,
        print_step=100,
        print_new_best=True,
        seed=42
    )
    
    # Initialize optimizer with the same model as run.py
    optimizer = BasicPromptOptimizer(
        model_name="ViT-H-14",
        pretrained="laion2b_s32b_b79k",
        config=config
    )
    
    # Use multiple test fixture images
    image_dir = Path(__file__).parent / "fixtures"
    image_paths = [
        image_dir / "molly.png",
        image_dir / "molly1.png",
        image_dir / "molly2.png"
    ]
    
    for path in image_paths:
        assert path.exists(), f"Test image not found at {path}"
    
    # Redirect stdout to stderr which handles Unicode better
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    
    try:
        # First get individual embeddings for each image
        individual_results = []
        print("\nAnalyzing individual images:", file=sys.stderr)
        for i, path in enumerate(image_paths):
            single_result = optimizer.optimize(target_images=[path])
            print(f"\nImage {i+1} ({path.name}) optimized prompt:", file=sys.stderr)
            print(repr(single_result)[1:-1], file=sys.stderr)
            individual_results.append(single_result)
        
        # Now optimize for all images together
        print("\nOptimizing for all images together:", file=sys.stderr)
        result = optimizer.optimize(target_images=image_paths)
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"\nOptimized prompt for multiple images: {result}", file=sys.stderr)
        
        # Print token count and raw string representation
        tokens = optimizer.tokenizer(result)
        token_ids = [t for t in tokens[0] if t != 0]  # Filter out padding
        print(f"\nToken count: {len(token_ids)}", file=sys.stderr)
        print(f"Tokens: {token_ids}", file=sys.stderr)
        
        # Print the prompt with preserved Unicode
        print("\nPrompt with preserved Unicode:", file=sys.stderr)
        print(repr(result)[1:-1], file=sys.stderr)
        
        # Compare common elements between individual and combined results
        print("\nAnalyzing common elements:", file=sys.stderr)
        combined_words = set(result.lower().split())
        for i, single_result in enumerate(individual_results):
            single_words = set(single_result.lower().split())
            common_words = combined_words.intersection(single_words)
            print(f"\nImage {i+1} ({image_paths[i].name}) shared terms: {sorted(common_words)}", file=sys.stderr)
            
    finally:
        sys.stdout = old_stdout

if __name__ == "__main__":
    pytest.main([__file__])
