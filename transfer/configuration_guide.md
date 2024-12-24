# Configuration Guide

## Configuration Parameters

### Core Parameters

1. **prompt_len** (default: 16)
   - Length of the optimized prompt
   - Longer prompts allow more detail but slower optimization
   - Recommended range: 8-32

2. **iter** (default: 3000)
   - Number of optimization iterations
   - Higher values may give better results but take longer
   - Recommended range: 1000-5000

3. **lr** (default: 0.1)
   - Learning rate for optimization
   - Controls how quickly tokens are updated
   - Recommended range: 0.01-0.5

4. **weight_decay** (default: 0.1)
   - Regularization parameter
   - Prevents overfitting
   - Recommended range: 0.01-0.5

### Model Parameters

1. **clip_model** (default: "ViT-H-14")
   - CLIP model architecture
   - Options:
     - "ViT-H-14"
     - "ViT-L-14"
     - "ViT-B-32"

2. **clip_pretrain** (default: "laion2b_s32b_b79k")
   - Pretrained weights source
   - Affects model performance and token space

### Batch Processing

1. **batch_size** (default: 1)
   - Number of images processed together
   - Higher values use more memory
   - Recommended: Based on GPU memory

2. **prompt_bs** (default: 1)
   - Prompt batch size for optimization
   - Multiple prompts optimized simultaneously

### Monitoring

1. **print_step** (default: 100)
   - Frequency of progress updates
   - Set to None to disable
   - Recommended: 100-500

## Sample Configuration

```json
{
    "prompt_len": 16,
    "iter": 3000,
    "lr": 0.1,
    "weight_decay": 0.1,
    "prompt_bs": 1,
    "loss_weight": 1.0,
    "print_step": 100,
    "batch_size": 1,
    "clip_model": "ViT-H-14",
    "clip_pretrain": "laion2b_s32b_b79k",
    "image_paths": ["path/to/image.png"]
}
```

## Configuration Tips

1. **Memory Management**
   - Reduce batch_size if out of memory
   - Monitor GPU usage during optimization
   - Consider gradient checkpointing for large models

2. **Optimization Balance**
   - Higher lr: Faster convergence but less stable
   - Lower lr: More stable but slower convergence
   - Adjust weight_decay based on prompt quality

3. **Model Selection**
   - ViT-H-14: Best quality, highest memory usage
   - ViT-L-14: Good balance of quality and speed
   - ViT-B-32: Fastest, lower quality

4. **Iteration Planning**
   - Start with lower iterations for testing
   - Increase for final results
   - Monitor loss convergence
