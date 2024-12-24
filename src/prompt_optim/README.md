# Prompt Optimization Package

A comprehensive toolkit for optimizing prompts using CLIP and other models.

## Structure

```
prompt_optim/
├── config/           # Configuration management
├── core/            # Core optimization functionality
├── optimizers/      # Specialized optimizers
└── utils/           # Utility functions
```

## Installation

```bash
pip install -e .[dev]  # Install with development dependencies
```

## Basic Usage

```python
from prompt_optim.core.base_optimizer import PromptOptimizer
from prompt_optim.config.optim_config import OptimConfig

# Initialize optimizer
optimizer = PromptOptimizer(model_name="ViT-B/32")

# Optimize prompt
prompt = optimizer.optimize(target_images=["image.jpg"])
```

## Development

1. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```

2. Run tests:
   ```bash
   pytest tests/
   ```

3. Format code:
   ```bash
   black src/ tests/
   isort src/ tests/
   ```

## License

MIT License
