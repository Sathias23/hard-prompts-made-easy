# Example Scripts Analysis

This document analyzes the example scripts in the repository and their interactions with the core library functionality.

## Core Library (optim_utils.py)

The `optim_utils.py` file provides core functionality for prompt optimization and manipulation:

- Neural network projection and embedding manipulation
- Image downloading and processing
- Prompt initialization and optimization
- Similarity measurement between images
- Various utility functions for handling tokens and random seeds

## Main Runner (run.py)

**Use Case**: Serves as the primary command-line interface for prompt optimization tasks. It provides two ways to run prompt optimization:

1. Using configuration from `sample_config.json`
2. Direct command-line usage with image paths as arguments

The script handles:
- Loading and validating input images
- Setting up CLIP model and device configuration
- Executing the prompt optimization process
- Outputting the learned prompt

This provides a convenient way to run prompt optimization without writing custom scripts, making it ideal for quick experiments or batch processing.

**Library Usage Compared to Examples**:
- **Simplified Integration**: Unlike the example scripts which each implement specific optimization strategies, `run.py` uses only the basic optimization pipeline through a single call to `optimize_prompt()`
- **Configuration Handling**: While example scripts hardcode their parameters or use specialized argument parsing, `run.py` provides flexible configuration through both JSON and command-line
- **Minimal Customization**: The example scripts each modify aspects of the optimization process (like blocklists or negative prompts), but `run.py` uses the default optimization behavior
- **Error Handling**: Includes more robust error handling for image loading and validation compared to the example scripts
- **Feature Usage**: Only uses core features (model loading, basic optimization) without the specialized features used in examples (no token blocking, style transfer, or negative prompting)

## Example Scripts

### 1. Blocklist Example (blocklist_example.py)

**Use Case**: Demonstrates how to optimize prompts while excluding certain tokens/concepts from being used in the generation process.

**Library Usage**:
- Uses the core optimization loop from `optim_utils.py`
- Leverages token embedding manipulation
- Implements custom token blocking during optimization

### 2. Negative Prompt (negative_prompt.py)

**Use Case**: Shows how to use negative prompting to steer the optimization away from certain characteristics while maintaining desired attributes.

**Library Usage**:
- Uses target feature extraction
- Employs the main optimization loop with negative conditioning
- Utilizes similarity measurement functions

### 3. Prompt Distillation (prompt_distillation.py)

**Use Case**: Demonstrates how to distill multiple target prompts or images into a single optimized prompt that captures shared characteristics.

**Library Usage**:
- Heavy use of target feature extraction
- Uses the core optimization loop
- Employs similarity measurement for validation

### 4. Prompt Inversion (prompt_inversion.py)

**Use Case**: Shows how to invert CLIP embeddings back into text prompts that capture the essence of target images or concepts.

**Library Usage**:
- Focuses on the core prompt optimization functionality
- Uses initialization utilities
- Employs embedding manipulation functions

### 5. Prompt Inversion SD (prompt_inversion_sd.py)

**Use Case**: A specialized version of prompt inversion specifically tailored for Stable Diffusion models, with additional parameters and optimizations.

**Library Usage**:
- Extended use of the optimization loop
- Custom model loading and processing
- More complex token embedding handling

### 6. Style Transfer (style_transfer.py)

**Use Case**: Demonstrates how to transfer stylistic elements between images using prompt optimization.

**Library Usage**:
- Combines multiple target feature extractions
- Uses similarity measurement extensively
- Employs the optimization loop with style-specific parameters

## Key Differences in Library Usage

1. **Optimization Strategy**:
   - Blocklist example focuses on token manipulation
   - Negative prompt uses contrastive optimization
   - Distillation combines multiple targets
   - Inversion focuses on accuracy to single targets
   - Style transfer balances content and style features

2. **Feature Extraction**:
   - Some scripts use single target extraction (inversion)
   - Others use multiple targets (distillation, style transfer)
   - Negative prompt uses both positive and negative targets

3. **Token Handling**:
   - Blocklist has specific token exclusion logic
   - SD inversion has specialized token initialization
   - Basic inversion uses simpler token management

4. **Model Integration**:
   - Most examples use standard CLIP models
   - SD inversion has specific Stable Diffusion integration
   - Style transfer may use multiple model features
