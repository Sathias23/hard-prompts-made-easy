import sys
from PIL import Image
import argparse
import open_clip
from optim_utils import *

config_path = "sample_config.json"

# load args
args = argparse.Namespace()
args.__dict__.update(read_json(config_path))

# Get image paths from config if provided, otherwise from command line arguments
image_paths = args.image_paths if args.image_paths else sys.argv[1:]

if not image_paths:
    sys.exit("""Usage: Either specify image paths in config.json or via command line:
    1. In config.json: Add paths to "image_paths": ["path1", "path2", ...]
    2. Command line: python run.py path-to-image [path-to-image-2 ...]
Passing multiple images will optimize a single prompt across all passed images, useful for style transfer.
""")

# load the target images
try:
    images = [Image.open(image_path) for image_path in image_paths]
except Exception as e:
    sys.exit(f"Error loading images: {str(e)}")

print(f"Loaded {len(images)} images: {image_paths}")

# You may modify the hyperparamters here
args.print_new_best = True

# load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device)

print(f"Running for {args.iter} steps.")
if getattr(args, 'print_new_best', False) and args.print_step is not None:
    print(f"Intermediate results will be printed every {args.print_step} steps.")

# optimize prompt
learned_prompt = optimize_prompt(model, preprocess, args, device, target_images=images)
print(learned_prompt)
