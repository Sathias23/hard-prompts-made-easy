"""
Basic usage example of the prompt optimization package.
"""

from prompt_optim.core.base_optimizer import PromptOptimizer
from prompt_optim.config.optim_config import OptimConfig

def main():
    # Create configuration
    config = OptimConfig(
        clip_model="ViT-B/32",
        num_iterations=1000,
        print_step=100
    )
    
    # Initialize optimizer
    optimizer = PromptOptimizer(
        model_name=config.clip_model,
        pretrained=config.clip_pretrain
    )
    
    # Example usage (to be implemented)
    print("Basic prompt optimizer initialized")
    print(f"Using device: {optimizer.device}")
    print(f"Model: {optimizer.model_name}")

if __name__ == "__main__":
    main()
