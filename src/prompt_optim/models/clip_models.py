"""
CLIP Models

This module contains the list of available CLIP models and their pretrained weights.
These models are provided by the open_clip library.

Notable models used in Stable Diffusion versions:
- SDXL: Uses two CLIP models
  1. Primary: openai/clip-vit-large-patch14 (ViT-L-14)
  2. Secondary: openai/clip-vit-big-patch14 (ViT-bigG-14)
- SD3: Uses OpenCLIP model
  - laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
"""

from typing import List, Tuple

# List of available CLIP models and their pretrained weights
AVAILABLE_MODELS: List[Tuple[str, str]] = [
    # ResNet Models
    ('RN50', 'openai'),
    ('RN50', 'yfcc15m'),
    ('RN50', 'cc12m'),
    ('RN50-quickgelu', 'openai'),
    ('RN50-quickgelu', 'yfcc15m'),
    ('RN50-quickgelu', 'cc12m'),
    ('RN101', 'openai'),
    ('RN101', 'yfcc15m'),
    ('RN101-quickgelu', 'openai'),
    ('RN101-quickgelu', 'yfcc15m'),
    ('RN50x4', 'openai'),
    ('RN50x16', 'openai'),
    ('RN50x64', 'openai'),
    
    # Vision Transformer Models
    ('ViT-B-32', 'openai'),
    ('ViT-B-32', 'laion400m_e31'),
    ('ViT-B-32', 'laion400m_e32'),
    ('ViT-B-32', 'laion2b_e16'),
    ('ViT-B-32', 'laion2b_s34b_b79k'),
    ('ViT-B-32-quickgelu', 'openai'),
    ('ViT-B-32-quickgelu', 'laion400m_e31'),
    ('ViT-B-32-quickgelu', 'laion400m_e32'),
    ('ViT-B-16', 'openai'),
    ('ViT-B-16', 'laion400m_e31'),
    ('ViT-B-16', 'laion400m_e32'),
    ('ViT-B-16-plus-240', 'laion400m_e31'),
    ('ViT-B-16-plus-240', 'laion400m_e32'),
    ('ViT-L-14', 'openai'),  # Used in SDXL as primary text encoder
    ('ViT-L-14', 'laion400m_e31'),
    ('ViT-L-14', 'laion400m_e32'),
    ('ViT-L-14', 'laion2b_s32b_b82k'),
    ('ViT-L-14-336', 'openai'),
    ('ViT-H-14', 'laion2b_s32b_b79k'),
    ('ViT-g-14', 'laion2b_s12b_b42k'),  # Base architecture for SD3's text encoder
    ('ViT-bigG-14', 'laion2b_39b_b160k'),  # Used in SDXL as secondary text encoder
    
    # Multilingual Models
    ('roberta-ViT-B-32', 'laion2b_s12b_b32k'),
    ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'),
    ('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k'),
]

def get_model_names() -> List[str]:
    """Get a list of unique model architectures."""
    return sorted(list(set(model for model, _ in AVAILABLE_MODELS)))

def get_pretrained_weights(model_name: str) -> List[str]:
    """Get available pretrained weights for a given model.
    
    Args:
        model_name: Name of the CLIP model
        
    Returns:
        List of available pretrained weight names
    """
    return [weight for model, weight in AVAILABLE_MODELS if model == model_name]

def is_valid_model(model_name: str, pretrained: str) -> bool:
    """Check if a model and pretrained weight combination is valid.
    
    Args:
        model_name: Name of the CLIP model
        pretrained: Name of the pretrained weights
        
    Returns:
        True if the combination is valid, False otherwise
    """
    return (model_name, pretrained) in AVAILABLE_MODELS
