"""
Base Prompt Optimizer

Provides the foundation for all prompt optimization implementations with integrated
utilities for embedding manipulation, image processing, and optimization.
"""

from typing import List, Optional, Union, Tuple, Any
from pathlib import Path
import logging
import torch
import numpy as np
import open_clip
from PIL import Image

from ..models import is_valid_model
from .optim_utils import (
    set_random_seed,
    decode_ids,
    download_image,
    get_target_feature,
    nn_project
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptOptimizer:
    """Base class for prompt optimization with integrated CLIP utilities."""
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
        seed: int = 42
    ):
        """Initialize the prompt optimizer.
        
        Args:
            model_name: Name of the CLIP model to use
            pretrained: Name of the pretrained weights
            device: Device to run on ('cuda' or 'cpu')
            seed: Random seed for reproducibility
            
        Raises:
            ValueError: If model_name or pretrained weights are invalid
            RuntimeError: If model initialization fails
        """
        if not is_valid_model(model_name, pretrained):
            raise ValueError(f"Invalid model name '{model_name}' or pretrained weights '{pretrained}'")
            
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed for reproducibility
        set_random_seed(seed)
        
        try:
            # Initialize model and tokenizer
            self.model, self.preprocess = self._init_model()
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            logger.info(f"Initialized {model_name} model with {pretrained} weights on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
    
    def _init_model(self) -> Tuple[Any, Any]:
        """Initialize the CLIP model and preprocessing transform.
        
        Returns:
            Tuple of (model, preprocess_transform)
            
        Raises:
            RuntimeError: If model creation fails
        """
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device
        )
        return model, preprocess
    
    def _process_images(
        self,
        images: Union[str, Path, List[str], List[Path], List[Image.Image]]
    ) -> List[Image.Image]:
        """Process input images into PIL Image objects.
        
        Args:
            images: Single image path or list of image paths/PIL Images
            
        Returns:
            List of PIL Image objects
            
        Raises:
            ValueError: If image loading fails
        """
        if isinstance(images, (str, Path)):
            images = [images]
            
        processed_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                if str(img).startswith(('http://', 'https://')):
                    img_obj = download_image(str(img))
                else:
                    try:
                        img_obj = Image.open(img).convert('RGB')
                    except Exception as e:
                        raise ValueError(f"Failed to load image {img}: {str(e)}")
            elif isinstance(img, Image.Image):
                img_obj = img
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
                
            if img_obj is None:
                raise ValueError(f"Failed to load image: {img}")
            processed_images.append(img_obj)
            
        return processed_images
    
    def get_embeddings(
        self,
        images: Optional[Union[str, Path, List[str], List[Path], List[Image.Image]]] = None,
        prompts: Optional[Union[str, List[str]]] = None
    ) -> torch.Tensor:
        """Get CLIP embeddings for images or prompts.
        
        Args:
            images: Input images
            prompts: Input text prompts
            
        Returns:
            Tensor of embeddings
            
        Raises:
            ValueError: If neither images nor prompts are provided
        """
        if images is None and prompts is None:
            raise ValueError("Must provide either images or prompts")
            
        with torch.no_grad():
            if images is not None:
                processed_images = self._process_images(images)
                image_inputs = torch.stack([self.preprocess(img) for img in processed_images])
                return self.model.encode_image(image_inputs.to(self.device))
            else:
                if isinstance(prompts, str):
                    prompts = [prompts]
                text_tokens = self.tokenizer(prompts).to(self.device)
                return self.model.encode_text(text_tokens)
    
    def optimize(
        self,
        target_images: Optional[Union[str, Path, List[str], List[Path]]] = None,
        target_prompts: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> str:
        """Optimize a prompt based on target images or prompts.
        
        Args:
            target_images: Path(s) to target image(s)
            target_prompts: Target prompt(s)
            **kwargs: Additional optimizer-specific parameters
            
        Returns:
            Optimized prompt string
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement optimize()")
