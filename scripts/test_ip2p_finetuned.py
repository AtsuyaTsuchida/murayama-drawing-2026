#!/usr/bin/env python3
import os, torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

MODEL = "output/ip2p_murayama"
INPUT = "data/murayama_full/test_A/000050.png"
PROMPT = "add the next brush strokes in the artist's style"
OUT = "outputs_ip2p_finetuned"
SEEDS = [1, 2, 3, 4]
IMG_CFGS = [1.2, 1.8]
STEPS = 30

os.makedirs(OUT, exist_ok=True)
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

image = Image.open(INPUT).convert("RGB")
for icfg in IMG_CFGS:
    for seed in SEEDS:
        g = torch.Generator("cuda").manual_seed(seed)
        result = pipe(PROMPT, image=image, num_inference_steps=STEPS,
                      image_guidance_scale=icfg, guidance_scale=7.5,
                      generator=g).images[0]
        name = f"icfg{icfg}_seed{seed}.png"
        result.save(os.path.join(OUT, name))
        print("saved:", name)
print("done ->", OUT)
