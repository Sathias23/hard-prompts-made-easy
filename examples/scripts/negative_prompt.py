# %%
try:
    import open_clip
    from optim_utils import * 
except ModuleNotFoundError:
    # You only really need this safety net if you want to run these notebooks directly in the examples directory
    # Don't worry about this if you installed the package or moved the notebook to the main directory.
    import os; os.chdir("..")
    import open_clip
    from optim_utils import * 

import torch
import mediapy as media
import argparse

# %% [markdown]
# ## Load Arguments

# %%
args = argparse.Namespace()
args.__dict__.update(read_json("sample_config.json"))

args

# %% [markdown]
# ## Load Clip Model

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device)

# %% [markdown]
# ## Load Diffusion Model

# %%
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

model_id = "stabilityai/stable-diffusion-2-1-base"
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    revision="fp16",
    )
pipe = pipe.to(device)

image_length = 512

# %% [markdown]
# ## Load Target Image

# %%
urls = [
        "https://www.pennington.com/-/media/Project/OneWeb/Pennington/Images/blog/seed/10-Surprising-Facts-About-Grass/grass_10surprising_opengraph.jpg",
       ]

orig_images = list(filter(None,[download_image(url) for url in urls]))
media.show_images(orig_images)

# %% [markdown]
# ## Optimize Prompt

# %%
learned_prompt = optimize_prompt(model, preprocess, args, device, target_images=orig_images)

# %% [markdown]
# ## Generate with Stable Diffusion Model

# %%
# you can customize the learned prompt here
prompt = "two dogs are running"
negative_prompt = learned_prompt

# %%
num_images = 4
guidance_scale = 9
num_inference_steps = 25
seed = 0

print(f"prompt: {prompt}")
print(f"negative prompt: {negative_prompt}")

set_random_seed(seed)
images = pipe(
    prompt,
    num_images_per_prompt=num_images,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    height=image_length,
    width=image_length,
    ).images
print(f"without negative prompt:")
media.show_images(images)

set_random_seed(seed)
images = pipe(
    prompt,
    num_images_per_prompt=num_images,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    height=image_length,
    width=image_length,
    negative_prompt=negative_prompt,
    ).images

print(f"with negative prompt:")
media.show_images(images)

# %%



