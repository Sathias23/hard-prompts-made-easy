"""
PyTest configuration and fixtures.
"""

import pytest
import torch
from pathlib import Path
from prompt_optim.config.optim_config import OptimConfig

@pytest.fixture
def config():
    """Provide a basic configuration for testing."""
    return OptimConfig()

@pytest.fixture
def device():
    """Provide a device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def test_image_path(tmp_path):
    """Create a test image for testing."""
    from PIL import Image
    import numpy as np
    
    # Create a simple test image
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    path = tmp_path / "test_image.png"
    img.save(path)
    return path

@pytest.fixture
def test_images(tmp_path):
    """Create multiple test images for testing."""
    from PIL import Image
    import numpy as np
    
    # Create test images with different colors
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
    ]
    
    image_paths = []
    for i, color in enumerate(colors):
        # Create a solid color image
        img = Image.new('RGB', (224, 224), color)
        path = tmp_path / f"test_image_{i}.png"
        img.save(path)
        image_paths.append(path)
    
    return image_paths
