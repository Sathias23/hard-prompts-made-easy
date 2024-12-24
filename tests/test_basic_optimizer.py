"""Tests for the BasicPromptOptimizer class."""

import pytest
import torch
from pathlib import Path
from PIL import Image
import tempfile
import os
import sys
import io
import numpy as np
from prompt_optim.models.clip_models import AVAILABLE_MODELS
from prompt_optim.optimizers.basic import BasicPromptOptimizer
from prompt_optim.config.optim_config import OptimConfig

@pytest.fixture
def test_image():
    """Create a test image for optimization."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='red')
        img.save(f.name)
        img.close()  # Close the image to avoid file handle issues
        path = Path(f.name)
        yield path
        # Cleanup with error handling
        try:
            if path.exists():
                os.remove(path)
        except Exception as e:
            pytest.skip(f"Could not cleanup test file: {e}")

@pytest.fixture
def optimizer():
    """Create a basic optimizer instance."""
    config = OptimConfig(
        prompt_length=8,
        num_iterations=2,  # Small number for testing
        learning_rate=0.1,
        weight_decay=0.0,
        print_step=1,
        print_new_best=True,
        seed=42
    )
    return BasicPromptOptimizer(config=config)

def test_optimizer_initialization(optimizer):
    """Test that the optimizer initializes correctly."""
    assert optimizer.model is not None
    assert optimizer.tokenizer is not None
    assert optimizer.device in ['cuda', 'cpu']
    assert optimizer.config is not None

def test_load_images(optimizer, test_image):
    """Test image loading functionality."""
    images = optimizer._load_images(test_image)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].mode == 'RGB'

def test_optimize_single_image(optimizer, test_image):
    """Test optimization with a single image."""
    result = optimizer.optimize(target_images=test_image)
    assert isinstance(result, str)
    assert len(result) > 0

def test_optimize_multiple_images(optimizer, test_image):
    """Test optimization with multiple copies of the same image."""
    result = optimizer.optimize(
        target_images=[test_image, test_image]  # Use same image twice for testing
    )
    assert isinstance(result, str)
    assert len(result) > 0

def test_optimize_invalid_input(optimizer):
    """Test optimizer behavior with invalid inputs."""
    with pytest.raises(ValueError):
        optimizer.optimize()  # No images provided
        
    with pytest.raises(NotImplementedError):
        optimizer.optimize(target_prompts=["test prompt"])  # Text optimization not implemented

def test_optimize_with_config_override(optimizer, test_image):
    """Test optimization with parameter overrides."""
    result = optimizer.optimize(
        target_images=test_image,
        prompt_len=12,  # Override default length
        lr=0.2  # Override default learning rate
    )
    assert isinstance(result, str)
    assert len(result) > 0

def test_optimize_real_image():
    """Test optimization with a real image using production-like settings."""
    config = OptimConfig(
        prompt_length=75,  # Target slightly under SDXL's 77 token limit
        num_iterations=500,  # Full optimization
        learning_rate=0.1,
        weight_decay=0.1,
        print_step=100,  # More frequent updates
        print_new_best=True,
        seed=42
    )
    
    # Initialize optimizer with the same model as run.py
    optimizer = BasicPromptOptimizer(
        model_name="ViT-H-14",  # Same as sample_config.json
        pretrained="laion2b_s32b_b79k",
        config=config
    )
    
    # Use the test fixture image
    image_path = Path(__file__).parent / "fixtures" / "F2cOgMma0AA0NyS.jpg"
    assert image_path.exists(), f"Test image not found at {image_path}"
    
    # Redirect stdout to stderr which handles Unicode better
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    
    try:
        result = optimizer.optimize(target_images=image_path)
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"\nOptimized prompt for test image: {result}", file=sys.stderr)
        
        # Print token count and raw string representation
        tokens = optimizer.tokenizer(result)
        token_ids = [t for t in tokens[0] if t != 0]  # Filter out padding
        print(f"\nToken count: {len(token_ids)}", file=sys.stderr)
        print(f"Tokens: {token_ids}", file=sys.stderr)
        
        # Print the prompt with preserved Unicode
        print("\nPrompt with preserved Unicode:", file=sys.stderr)
        print(repr(result)[1:-1], file=sys.stderr)  # Remove quotes but keep escapes
    finally:
        sys.stdout = old_stdout

def test_optimize_multiple_images():
    """Test optimization with multiple images using production-like settings."""
    config = OptimConfig(
        prompt_length=40,  # Much more conservative limit
        num_iterations=500,
        learning_rate=0.1,
        weight_decay=0.1,
        print_step=100,
        print_new_best=True,
        seed=42
    )
    
    # Initialize optimizer with the same model as run.py
    optimizer = BasicPromptOptimizer(
        model_name="ViT-H-14",
        pretrained="laion2b_s32b_b79k",
        config=config
    )
    
    # Use multiple test fixture images
    image_dir = Path(__file__).parent / "fixtures"
    image_paths = [
        image_dir / "molly.png",
        image_dir / "molly1.png",
        image_dir / "molly2.png"
    ]
    
    for path in image_paths:
        assert path.exists(), f"Test image not found at {path}"
    
    # Redirect stdout to stderr for better Unicode handling
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    
    try:
        # First get individual embeddings for each image
        individual_results = []
        print("\nAnalyzing individual images:", file=sys.stderr)
        for i, path in enumerate(image_paths):
            single_result = optimizer.optimize(target_images=[path])
            print(f"\nImage {i+1} ({path.name}) optimized prompt:", file=sys.stderr)
            print(repr(single_result)[1:-1], file=sys.stderr)
            individual_results.append(single_result)
        
        # Now optimize for all images together
        print("\nOptimizing for all images together:", file=sys.stderr)
        result = optimizer.optimize(target_images=image_paths)
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"\nOptimized prompt for multiple images: {result}", file=sys.stderr)
        
        # Print token count and raw string representation
        tokens = optimizer.tokenizer(result)
        token_ids = [t for t in tokens[0] if t != 0]  # Filter out padding
        print(f"\nToken count: {len(token_ids)}", file=sys.stderr)
        print(f"Tokens: {token_ids}", file=sys.stderr)
        
        # Print the prompt with preserved Unicode
        print("\nPrompt with preserved Unicode:", file=sys.stderr)
        print(repr(result)[1:-1], file=sys.stderr)
        
        # Compare common elements between individual and combined results
        print("\nAnalyzing common elements:", file=sys.stderr)
        combined_words = set(result.lower().split())
        for i, single_result in enumerate(individual_results):
            single_words = set(single_result.lower().split())
            common_words = combined_words.intersection(single_words)
            print(f"\nImage {i+1} ({image_paths[i].name}) shared terms: {sorted(common_words)}", file=sys.stderr)
            
    finally:
        sys.stdout = old_stdout

def clean_prompt(prompt):
    """Clean up a generated prompt while preserving Unicode characters."""
    import re
    
    # Remove the "best prompt so far:" prefix if present
    if "best prompt so far:" in prompt:
        prompt = prompt.replace("best prompt so far:", "").strip()
    
    # Normalize whitespace without removing Unicode
    prompt = re.sub(r'\s+', ' ', prompt).strip()
    
    return prompt

def test_optimize_with_sd_encoders():
    """Test optimization using CLIP encoders from SDXL and SD3.
    
    This test uses three different CLIP models:
    1. SDXL Primary: ViT-L-14 (openai)
    2. SDXL Secondary: ViT-bigG-14 (laion2b_39b_b160k)
    3. SD3 Base: ViT-g-14 (laion2b_s12b_b42k)
    """
    # Load test image
    image_path = Path("tests/fixtures/molly.png")
    assert image_path.exists(), f"Test image not found at {image_path}"
    
    # Define models to test
    models = [
        ("ViT-L-14", "openai", "SDXL Primary", 42),
        ("ViT-bigG-14", "laion2b_39b_b160k", "SDXL Secondary", 43),
        ("ViT-g-14", "laion2b_s12b_b42k", "SD3 Base", 44)
    ]
    
    results = {}
    
    # Redirect stdout to stderr for better Unicode handling
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    
    try:
        for model_name, pretrained, description, seed in models:
            print(f"\n\nTesting {description}", file=sys.stderr)
            print(f"Model: {model_name}, Pretrained: {pretrained}, Seed: {seed}", file=sys.stderr)
            
            # Create optimizer with specific model config
            config = OptimConfig(
                prompt_length=40,  # Shorter prompt length for faster testing
                num_iterations=500,  # Same as test_optimize_multiple_images
                learning_rate=0.1,
                weight_decay=0.1,  # Same as test_optimize_multiple_images
                print_step=100,  # Print less frequently
                print_new_best=True,
                seed=seed,  # Use different seed for each model
                clip_model=model_name,
                clip_pretrain=pretrained,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            print("Created config:", config, file=sys.stderr)
            optimizer = BasicPromptOptimizer(config=config)
            print("Created optimizer", file=sys.stderr)
            
            # Optimize prompt
            try:
                print(f"Starting optimization for {description}...", file=sys.stderr)
                print(f"Using device: {config.device}", file=sys.stderr)
                
                # Get optimization result
                prompt = optimizer.optimize([str(image_path)])
                print(f"Optimization complete for {description}", file=sys.stderr)
                
                # Clean the prompt
                prompt = clean_prompt(prompt)
                
                results[description] = {
                    "prompt": prompt,
                    "score": None  # Score not available in current implementation
                }
                
                # Print detailed results
                print(f"\n{description} Results:", file=sys.stderr)
                print(f"Prompt: {prompt}", file=sys.stderr)
                
                # Print token info
                tokens = optimizer.tokenizer(prompt)
                token_ids = [t for t in tokens[0] if t != 0]  # Filter out padding
                print(f"Token count: {len(token_ids)}", file=sys.stderr)
                print(f"Tokens: {token_ids}", file=sys.stderr)
                
            except Exception as e:
                print(f"Error with {description}: {str(e)}", file=sys.stderr)
                print(f"Full error details:", str(e), file=sys.stderr)
                continue
    
        # Verify we got results from at least one model
        assert len(results) > 0, "No models succeeded in optimization"
        
        # Print comparison of results
        print("\nComparison of Results:", file=sys.stderr)
        for desc1, result1 in results.items():
            for desc2, result2 in results.items():
                if desc1 < desc2:  # Avoid duplicate comparisons
                    print(f"\nComparing {desc1} vs {desc2}:", file=sys.stderr)
                    print(f"{desc1} prompt: {result1['prompt']}", file=sys.stderr)
                    print(f"{desc2} prompt: {result2['prompt']}", file=sys.stderr)
    
    finally:
        sys.stdout = old_stdout

def test_optimize_with_long_text():
    """Test optimizing prompts using a long target text prompt that exceeds CLIP's 77 token limit."""
    optimizer = BasicPromptOptimizer(
        model_name="ViT-B-32",
        pretrained="openai"
    )
    
    # Long, detailed prompt that exceeds CLIP's token limit
    target_prompt = """A surreal and grotesque humanoid figure with exaggerated features, including a large grinning mouth filled with discolored teeth and bright, bulging blue eyes. The figure's skin is clammy and glossy, with an irregular, mottled texture. Its oversized head is disproportionate to its childlike torso. The scene is set in a dimly lit, cluttered room with a single ceiling light casting harsh shadows. The walls are painted in muted tones, and the background objects are barely visible, creating a claustrophobic atmosphere. The lighting is stark and dramatic, with a yellowish hue, emphasizing the unsettling, horror-themed mood. The style is highly detailed realism, with surreal and grotesque elements, focusing on intricate textures and shadows for an eerie aesthetic."""
    
    results = []
    results.append("Optimizing long prompt...")
    results.append(f"Target prompt ({len(target_prompt.split())} words):\n{target_prompt}\n")
    
    optimized_prompt = optimizer.optimize(
        target_prompts=target_prompt,
        **{
            "prompt_len": 24,  # Shorter length for more coherent output
            "iter": 1000,  # More iterations for better convergence
            "lr": 0.01,  # Lower learning rate for more stable optimization
            "weight_decay": 0.1,  # Add weight decay to prevent repetition
            "loss_weight": 2.0,  # Increase loss weight for stronger optimization
            "print_step": 200,
            "print_new_best": True  # Print when we find better prompts
        }
    )
    
    results.append(f"\nOptimization Results:")
    results.append(f"Original length: {len(target_prompt.split())} words")
    results.append(f"Optimized length: {len(optimized_prompt.split())} words")
    results.append(f"Optimized prompt: {optimized_prompt}")
    
    # Analyze semantic elements preserved
    key_elements = [
        "surreal", "grotesque", "humanoid", "exaggerated", 
        "grinning", "teeth", "eyes", "skin", "texture",
        "lighting", "shadows", "horror", "eerie"
    ]
    preserved = [word for word in key_elements if word.lower() in optimized_prompt.lower()]
    results.append(f"\nKey elements preserved: {', '.join(preserved)}")
    
    # Write results to file
    with open("optimization_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    
    assert isinstance(optimized_prompt, str)
    assert len(optimized_prompt.split()) >= 1, "Optimized prompt should contain at least one word"

def test_analyze_clip_tokenization():
    """Analyze how CLIP tokenizes and truncates long prompts."""
    import open_clip
    import torch
    from transformers import CLIPTokenizer
    
    # Get both open_clip and HF tokenizers for analysis
    open_clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    hf_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Long, detailed prompt
    target_prompt = """A surreal and grotesque humanoid figure with exaggerated features, including a large grinning mouth filled with discolored teeth and bright, bulging blue eyes. The figure's skin is clammy and glossy, with an irregular, mottled texture. Its oversized head is disproportionate to its childlike torso. The scene is set in a dimly lit, cluttered room with a single ceiling light casting harsh shadows. The walls are painted in muted tones, and the background objects are barely visible, creating a claustrophobic atmosphere. The lighting is stark and dramatic, with a yellowish hue, emphasizing the unsettling, horror-themed mood. The style is highly detailed realism, with surreal and grotesque elements, focusing on intricate textures and shadows for an eerie aesthetic."""
    
    # Tokenize with both tokenizers
    open_clip_tokens = open_clip_tokenizer(target_prompt)
    hf_tokens = hf_tokenizer(target_prompt, truncation=True, max_length=77, return_tensors="pt")
    
    # Get token texts from HF tokenizer (which has better inspection tools)
    token_texts = hf_tokenizer.convert_ids_to_tokens(hf_tokens.input_ids[0])
    decoded_text = hf_tokenizer.decode(hf_tokens.input_ids[0])
    
    results = []
    results.append("CLIP Tokenization Analysis")
    results.append("-" * 50)
    results.append(f"\nOriginal prompt ({len(target_prompt.split())} words):\n{target_prompt}")
    results.append(f"\nNumber of tokens: {len(token_texts)}")
    results.append("\nIndividual tokens processed by CLIP:")
    for i, token in enumerate(token_texts, 1):
        results.append(f"{i:2d}. {token}")
    
    results.append(f"\nProcessed text (after tokenization/detokenization):\n{decoded_text}")
    
    # Calculate what portion of the text was dropped
    original_words = set(target_prompt.split())
    processed_words = set(decoded_text.split())
    dropped_words = original_words - processed_words
    
    results.append(f"\nWords that were dropped due to context window limit:")
    results.append(", ".join(sorted(dropped_words)))
    
    # Write analysis to file
    with open("tokenization_analysis.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    
    # Verify the tokenization behavior
    assert len(token_texts) > 0, "Should have processed some tokens"
    assert len(token_texts) <= 77, "Should not exceed CLIP's context window"

if __name__ == "__main__":
    pytest.main([__file__])
