"""
Basic Prompt Optimizer

Implements the core prompt optimization functionality by wrapping optim_utils.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Callable
from pathlib import Path
from PIL import Image
from dataclasses import asdict, dataclass
import logging

from ..core.base_optimizer import PromptOptimizer
from ..config.optim_config import OptimConfig
from ..core.optim_utils import optimize_prompt, set_random_seed

logger = logging.getLogger(__name__)

@dataclass
class OptimArgs:
    """Arguments for the optimization process."""
    prompt_len: int = 8
    prompt_bs: int = 1
    iter: int = 100
    lr: float = 0.1
    weight_decay: float = 0.0
    loss_weight: float = 1.0
    batch_size: Optional[int] = None
    print_step: Optional[int] = None
    print_new_best: bool = False
    tokenizer: Optional[object] = None
    seed: int = 42
    clip_model: str = "ViT-B-32"  # Added for compatibility with optim_utils


class BasicPromptOptimizer(PromptOptimizer):
    """Basic prompt optimizer that wraps the core optimization functionality."""
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",  # Changed from ViT-B/32 to match open_clip names
        pretrained: str = "openai",
        device: Optional[str] = None,
        config: Optional[OptimConfig] = None
    ):
        """Initialize the basic prompt optimizer.
        
        Args:
            model_name: Name of the CLIP model to use
            pretrained: Name of the pretrained weights
            device: Device to run on ('cuda' or 'cpu')
            config: Optional configuration object
        """
        super().__init__(model_name, pretrained, device)
        self.config = config or OptimConfig()
        self.model_name = model_name
        
    def _load_images(self, image_paths: Union[str, Path, List[str], List[Path]]) -> List[Image.Image]:
        """Load and preprocess images."""
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
            
        images = []
        for path in image_paths:
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        
        return images
    
    def optimize(
        self,
        target_images: Optional[Union[str, Path, List[str], List[Path]]] = None,
        target_prompts: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> str:
        """Optimize a prompt based on target images or prompts.
        
        Args:
            target_images: Path(s) to target image(s)
            target_prompts: Target prompt(s) to optimize towards
            **kwargs: Additional optimizer-specific parameters that override config
            
        Returns:
            Optimized prompt string
            
        Raises:
            ValueError: If neither target_images nor target_prompts are provided
        """
        if target_images is None and target_prompts is None:
            raise ValueError("Must provide either target_images or target_prompts")
        
        # Load images if provided
        images = self._load_images(target_images) if target_images is not None else None
        
        # Process text prompts if provided
        if target_prompts is not None and isinstance(target_prompts, str):
            target_prompts = [target_prompts]
        
        # Set random seed for reproducibility
        set_random_seed(self.config.seed)
        
        # Convert config to args and override with kwargs
        args_dict = {
            "prompt_len": self.config.prompt_length,
            "prompt_bs": 1,  # We only support single prompt for now
            "iter": self.config.num_iterations,
            "lr": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "loss_weight": 1.0,
            "batch_size": None,  # Use all images/prompts
            "print_step": self.config.print_step,
            "print_new_best": self.config.print_new_best,
            "tokenizer": self.tokenizer,
            "seed": self.config.seed,
            "clip_model": self.model_name  # Pass the CLIP model name
        }
        
        # Update with any provided kwargs
        args_dict.update(kwargs)
        args = OptimArgs(**args_dict)
        
        # Run optimization using optim_utils
        logger.info("Starting prompt optimization...")
        try:
            best_text = optimize_prompt(
                model=self.model,
                preprocess=self.preprocess,
                args=args,
                device=self.device,
                target_images=images,
                target_prompts=target_prompts
            )
            # Remove or replace problematic Unicode characters
            best_text = best_text.encode('ascii', 'ignore').decode('ascii')
            logger.info("Optimization complete.")
            return best_text
        except UnicodeEncodeError as e:
            # Handle Unicode encoding errors by stripping problematic characters
            if hasattr(e, 'object'):
                best_text = e.object.encode('ascii', 'ignore').decode('ascii')
                logger.info("Optimization complete with character encoding fix.")
                return best_text
            else:
                logger.error("Failed to handle Unicode characters in prompt")
                raise
        except Exception as e:
            logger.error("Error during optimization: %s", str(e).encode('ascii', 'ignore').decode('ascii'))
            raise
