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

args.prompt_len = 16
args.batch_size = 2

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
        "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/d457e81b-8fff-4551-bf7b-041a87f7a8fd/d9fe6yk-0e97713b-a432-4bba-bc84-c2c6e0a91da0.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2Q0NTdlODFiLThmZmYtNDU1MS1iZjdiLTA0MWE4N2Y3YThmZFwvZDlmZTZ5ay0wZTk3NzEzYi1hNDMyLTRiYmEtYmM4NC1jMmM2ZTBhOTFkYTAuanBnIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.S0JNhbp5dIHZnHn6EXmmJ_ns3grKv35fWC7b_nF_QNE",
        "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/d457e81b-8fff-4551-bf7b-041a87f7a8fd/d8o0liu-2e215aaf-b752-4fc5-bed3-39f46b97b1fb.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2Q0NTdlODFiLThmZmYtNDU1MS1iZjdiLTA0MWE4N2Y3YThmZFwvZDhvMGxpdS0yZTIxNWFhZi1iNzUyLTRmYzUtYmVkMy0zOWY0NmI5N2IxZmIuanBnIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.2zJumpTr6tc64IVpQH1T1aQ17myEtVYFMkTvcOtOUSI",
        "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/d457e81b-8fff-4551-bf7b-041a87f7a8fd/d9v4o2s-1d877e32-0952-4a97-af81-5103a63f9120.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2Q0NTdlODFiLThmZmYtNDU1MS1iZjdiLTA0MWE4N2Y3YThmZFwvZDl2NG8ycy0xZDg3N2UzMi0wOTUyLTRhOTctYWY4MS01MTAzYTYzZjkxMjAuanBnIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.3dl3eJs-rt_Z5ytXexem3o2kF1ECRiY-nF989Zel510",
        "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/d457e81b-8fff-4551-bf7b-041a87f7a8fd/datb7tb-407d2491-8673-4b8a-b48b-b4d74b76f82c.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2Q0NTdlODFiLThmZmYtNDU1MS1iZjdiLTA0MWE4N2Y3YThmZFwvZGF0Yjd0Yi00MDdkMjQ5MS04NjczLTRiOGEtYjQ4Yi1iNGQ3NGI3NmY4MmMuanBnIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.hl1dtToXHxrYufT7soCfaOWlN4TazSwMTaqesPl7aUw",
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
prompt = f"the streets of Paris in the style of {learned_prompt}"

# %%
num_images = 4
guidance_scale = 9
num_inference_steps = 25

images = pipe(
    prompt,
    num_images_per_prompt=num_images,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    height=image_length,
    width=image_length,
    ).images

print(f"prompt: {prompt}")
media.show_images(images)

# %%



