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

args.prompt_len = 8

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
# ## Enter Target Prompt

# %%
target_prompts = [
        "realistic car 3 d render sci - fi car and sci - fi robotic factory structure in the coronation of napoleon painting and digital billboard with point cloud in the middle, unreal engine 5, keyshot, octane, artstation trending, ultra high detail, ultra realistic, cinematic, 8 k, 1 6 k, in style of zaha hadid, in style of nanospace michael menzelincev, in style of lee souder, in plastic, dark atmosphere, tilt shift, depth of field",
       ]
print(target_prompts)

# %% [markdown]
# ## Optimize Prompt

# %%
learned_prompt = optimize_prompt(model, preprocess, args, device, target_prompts=target_prompts)

# %% [markdown]
# ## Generate with Stable Diffusion Model

# %%
num_images = 4
guidance_scale = 9
num_inference_steps = 25
seed = 0

set_random_seed(seed)
images = pipe(
    target_prompts[0],
    num_images_per_prompt=num_images,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    height=image_length,
    width=image_length,
    ).images
print(f"original prompt: {target_prompts[0]}")
media.show_images(images)

set_random_seed(seed)
images = pipe(
    learned_prompt,
    num_images_per_prompt=num_images,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    height=image_length,
    width=image_length,
    ).images

print(f"learned prompt: {learned_prompt}")
media.show_images(images)

# %%



