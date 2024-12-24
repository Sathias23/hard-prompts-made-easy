# Prompt Optimization Module Refactor Strategy

## Overview

This document outlines a strategy for consolidating the functionality from all example scripts and the main runner into a unified, object-oriented module. The goal is to create a flexible, maintainable class structure that supports all current use cases while allowing for future extensions.

## Proposed Class Structure

### 1. Base Class: `PromptOptimizer`

```python
class PromptOptimizer:
    def __init__(self, model_name: str, device: str = None):
        # Core initialization
        # Model loading
        # Basic configuration
```

Core responsibilities:
- Model initialization and management
- Basic configuration handling
- Common utility methods
- Base optimization pipeline

### 2. Configuration Class: `OptimConfig`

```python
class OptimConfig:
    def __init__(self):
        # Basic optimization parameters
        # Model configuration
        # Output settings
```

Responsibilities:
- Parameter validation
- Configuration persistence
- Default value management
- Configuration merging

### 3. Specialized Optimizer Classes

#### `BasicPromptOptimizer(PromptOptimizer)`
- Implements the basic optimization from run.py
- Handles command-line and config file inputs
- Provides simple API for basic use cases

#### `BlocklistOptimizer(PromptOptimizer)`
- Adds token blocking functionality
- Custom token filtering
- Blocklist management

#### `NegativePromptOptimizer(PromptOptimizer)`
- Handles negative prompting
- Manages contrast between positive/negative targets
- Custom scoring for negative conditions

#### `DistillationOptimizer(PromptOptimizer)`
- Handles multiple target prompts/images
- Implements distillation logic
- Manages feature aggregation

#### `InversionOptimizer(PromptOptimizer)`
- Basic CLIP inversion functionality
- Optimization for single targets
- Quality metrics for inversion

#### `SDInversionOptimizer(InversionOptimizer)`
- Extends basic inversion for Stable Diffusion
- SD-specific parameter handling
- Custom initialization for SD

#### `StyleTransferOptimizer(PromptOptimizer)`
- Style transfer specific functionality
- Balance between content/style features
- Custom scoring for style transfer

## Key Design Principles

1. **Inheritance Hierarchy**
   - Base class provides core functionality
   - Specialized classes extend only what they need
   - Avoid deep inheritance chains

2. **Configuration Management**
   - Centralized configuration through OptimConfig
   - Easy parameter override system
   - Validation at configuration level

3. **Interface Consistency**
   ```python
   # Common interface across all optimizers
   def optimize(self, targets, **kwargs) -> str:
       pass
   ```

4. **Extensibility Points**
   - Hook methods for customization
   - Event system for optimization steps
   - Plugin architecture for new features

## Implementation Strategy

### Phase 1: Core Infrastructure
1. Create base PromptOptimizer class
2. Implement OptimConfig
3. Set up basic testing framework

### Phase 2: Basic Functionality
1. Implement BasicPromptOptimizer
2. Port run.py functionality
3. Add CLI interface

### Phase 3: Specialized Optimizers
1. Implement each specialized optimizer
2. Port existing example functionality
3. Add specific tests for each

### Phase 4: Integration
1. Create unified API
2. Add documentation
3. Create example notebooks

## Usage Examples

```python
# Basic usage
optimizer = BasicPromptOptimizer("ViT-B/32")
prompt = optimizer.optimize(images=["image1.jpg"])

# Blocklist usage
optimizer = BlocklistOptimizer("ViT-B/32")
prompt = optimizer.optimize(
    images=["image1.jpg"],
    blocklist=["unwanted", "terms"]
)

# Style transfer
optimizer = StyleTransferOptimizer("ViT-B/32")
prompt = optimizer.optimize(
    content_image="content.jpg",
    style_image="style.jpg"
)
```

## Benefits of This Approach

1. **Modularity**
   - Each optimization strategy is self-contained
   - Easy to add new optimization methods
   - Clear separation of concerns

2. **Maintainability**
   - Centralized configuration management
   - Consistent interface across optimizers
   - Reduced code duplication

3. **Usability**
   - Simple interface for basic use cases
   - Advanced features available when needed
   - Consistent API across all use cases

4. **Testability**
   - Each component can be tested independently
   - Clear boundaries for unit tests
   - Easy to mock dependencies

## Future Extensions

1. **Additional Optimizers**
   - Multi-modal optimization
   - Batch optimization
   - Custom scoring functions

2. **Pipeline Customization**
   - Custom optimization steps
   - User-defined callbacks
   - Progress monitoring

3. **Integration Features**
   - REST API wrapper
   - Async optimization
   - Distributed optimization

## Migration Path

1. Create new module structure
2. Implement base classes
3. Port each example one at a time
4. Add tests for each port
5. Update documentation
6. Deprecate old scripts
7. Create new examples
