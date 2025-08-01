# -*- coding: utf-8 -*-
"""IMG GEN.ipynb



Original file is located at
    https://colab.research.google.com/drive/1vck1mxPpqsoh_9kjdx4Td2h8MZta42iL
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install diffusers transformers accelerate invisible_watermark

"""Once the library is installed, you can load a model and generate an image. This can take some time depending on the model size and your hardware."""

import torch
from diffusers import StableDiffusionPipeline

# Load a pre-trained model. Using a smaller model here for demonstration.
# You can explore other models like "runwayml/stable-diffusion-v1-5" or others on Hugging Face.
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Define your text prompt
prompt = "a photo of an astronaut riding a horse on the moon"

# Generate the image
image = pipe(prompt).images[0]

# Display the generated image
display(image)

import torch
from diffusers import StableDiffusionPipeline

# Load a pre-trained model. Using a smaller model here for demonstration.
# You can explore other models like "runwayml/stable-diffusion-v1-5" or others on Hugging Face.
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Define your text prompt
prompt = "a photo of a dog"

# Generate the image
image = pipe(prompt).images[0]

# Display the generated image
display(image)

import torch
from diffusers import StableDiffusionPipeline

# Load a pre-trained model. Using a smaller model here for demonstration.
# You can explore other models like "runwayml/stable-diffusion-v1-5" or others on Hugging Face.
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Define your text prompt
prompt = "a photo of a horse"

# Generate the image
image = pipe(prompt).images[0]

# Display the generated image
display(image)

import torch
from diffusers import StableDiffusionPipeline

# Load a pre-trained model. Using a smaller model here for demonstration.
# You can explore other models like "runwayml/stable-diffusion-v1-5" or others on Hugging Face.
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Define your text prompt
prompt = "a photo of flower vase"

# Generate the image
image = pipe(prompt).images[0]

# Display the generated image
display(image)
