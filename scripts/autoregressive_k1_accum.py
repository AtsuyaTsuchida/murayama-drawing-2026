#!/usr/bin/env python3
"""自己回帰生成(累積方式)。各反復の新規ストロークをキャンバスに積み上げ、
元画像の既存部分は常に保持する。"""
import os, numpy as np, torch
from PIL import Image, ImageFilter
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

MODEL  = "output/ip2p_murayama"
INPUT  = "data/murayama_full/test_A/000050.png"
PROMPT = "add the next brush strokes in the artist's style"
OUT    = "outputs_autoregressive_accum"
STEPS_N, N_ITER = 30, 30
IMG_CFG, GUID   = 1.2, 7.5
SEED_BASE = 100
LO, HI, BLUR = 18, 42, 1.5

os.makedirs(OUT, exist_ok=True)
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

base = Image.open(INPUT).convert("RGB")
base.save(os.path.join(OUT, "iter_000.png"))
canvas = np.asarray(base, dtype=np.float32)   # 積み上げ先
acc = np.zeros(canvas.shape[:2], dtype=np.float32)  # これまでに描いた領域

for i in range(1, N_ITER+1):
    img = Image.fromarray(canvas.astype(np.uint8))
    g = torch.Generator("cuda").manual_seed(SEED_BASE + i)
    out = pipe(PROMPT, image=img, num_inference_steps=STEPS_N,
               image_guidance_scale=IMG_CFG, guidance_scale=GUID,
               generator=g).images[0]
    if out.size != img.size:
        out = out.resize(img.size, Image.BICUBIC)
    cur = np.asarray(out, dtype=np.float32)

    d = np.abs(cur - canvas).mean(axis=2)
    w = np.clip((d - LO) / (HI - LO), 0, 1)
    w = np.asarray(Image.fromarray((w*255).astype(np.uint8))
                   .filter(ImageFilter.GaussianBlur(BLUR)), dtype=np.float32) / 255.0
    # 既に描いた場所は上書きしない(新規領域を優先)
    w = w * (1.0 - np.clip(acc, 0, 1))
    acc = np.clip(acc + w, 0, 1)

    canvas = np.clip(cur * w[...,None] + canvas * (1 - w[...,None]), 0, 255)
    Image.fromarray(canvas.astype(np.uint8)).save(os.path.join(OUT, f"iter_{i:03d}.png"))
    print(f"iter {i:03d} saved", flush=True)

print("\nsaved ->", OUT)
