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
import copy

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

tokenizer = open_clip.tokenizer._tokenizer
original_token_embedding = copy.deepcopy(model.token_embedding)

# %% [markdown]
# ## Load Target Image

# %%
urls = [
        "https://www.nps.gov/caco/planyourvisit/images/WEF_7206181_2.jpg",
       ]

orig_images = list(filter(None,[download_image(url) for url in urls]))
media.show_images(orig_images)

# %% [markdown]
# ## Blocklist

# %%
### substrings you want to block
blocklist_substrings = ["shark", "jaw", "🦈"]

blocklist = []
blocklist_words = []

for curr_w in blocklist_substrings:
    blocklist += tokenizer.encode(curr_w)
    blocklist_words.append(curr_w)

for curr_w in list(tokenizer.encoder.keys()):
    for blocklist_substring in blocklist_substrings:
        if blocklist_substring in curr_w:
            blocklist.append(tokenizer.encoder[curr_w])
            blocklist_words.append(curr_w)
blocklist = list(set(blocklist))

token_embedding = copy.deepcopy(original_token_embedding)
if blocklist is not None:
    with torch.no_grad():
        token_embedding.weight[blocklist] = 0
        
model.token_embedding = token_embedding

print("blocked words")
print(blocklist_words)

# %% [markdown]
# ## Optimize Prompt

# %%
learned_prompt = optimize_prompt(model, preprocess, args, device, target_images=orig_images)

# %%



