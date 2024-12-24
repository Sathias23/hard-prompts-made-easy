# Dependencies and Setup Guide

## Core Dependencies

### Required Packages
```
PyTorch >= 1.13.0
transformers >= 4.23.1
diffusers >= 0.11.1
sentence-transformers >= 2.2.2
ftfy >= 6.1.1
mediapy >= 1.1.2
```

## Installation Steps

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. CLIP Setup

1. **Model Download**
   - Automatic with transformers
   - Cached in ~/.cache/huggingface
   - Requires internet connection first run

2. **Model Variants**
   - ViT-H-14 (default)
   - ViT-L-14
   - ViT-B-32

### 3. GPU Support

1. **CUDA Requirements**
   - CUDA toolkit >= 11.6
   - cuDNN compatible version
   - Appropriate GPU drivers

2. **Memory Requirements**
   - ViT-H-14: ~16GB VRAM
   - ViT-L-14: ~8GB VRAM
   - ViT-B-32: ~4GB VRAM

## Optional Dependencies

### 1. Development Tools
```
pytest
black
flake8
mypy
```

### 2. Visualization
```
matplotlib
tensorboard
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller model
   - Enable gradient checkpointing

2. **Import Errors**
   - Check Python version (3.8+ recommended)
   - Verify package versions
   - Check virtual environment activation

3. **Model Download Issues**
   - Check internet connection
   - Verify huggingface token if needed
   - Check disk space

## Version Compatibility

### Tested Configurations

1. **Windows**
   - Python 3.8-3.10
   - CUDA 11.6, 11.7
   - PyTorch 1.13.0, 2.0.0

2. **Linux**
   - Python 3.8-3.11
   - CUDA 11.6+
   - PyTorch 1.13.0+

### Known Issues

1. **Version Conflicts**
   - transformers/diffusers version mismatch
   - CUDA version compatibility
   - Python version requirements

2. **Performance Issues**
   - Memory leaks with certain versions
   - Speed regression in specific configs
   - GPU utilization variations
