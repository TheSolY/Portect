from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
import torch
import os

num_images = 10
save_dir = '/tmp/sol/portect/gen_images_portect_org'

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
lora_path = "dreambooth-outputs/noa_org_model/pytorch_lora_weights.safetensors"
pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
pipe = pipe.to("cuda:3")
pipe.load_lora_weights(lora_path)


prompt = "ukj, portrait, (detailed eyes), (Intricate),(High Detail), (bokeh)."
images = pipe(prompt,
             num_inference_steps=100, guidance_scale=7.5, num_images_per_prompt=num_images)[0]

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

for i in range(num_images):
    images[i].save(f"{save_dir}/gen_img{i}.png")