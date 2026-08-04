#!/usr/bin/env python3
"""変化領域マスク内での SNR を測る。全体平均では埋もれる信号が、
局所的には十分な強度で存在するかを確認する。"""
import os, numpy as np, torch
from glob import glob
from PIL import Image
from diffusers import AutoencoderKL

RAW = "data/murayama_raw"
R = 256
THRESH = 25          # 前回レポートの差分データセットと同じ閾値
KS = [1, 2, 5, 10]
N_DIRS = 20          # 測る作品数
N_FRAMES = 15        # 各作品の先頭から何フレーム使うか

vae = AutoencoderKL.from_pretrained("output/ip2p_murayama_skip10", subfolder="vae",
                                    torch_dtype=torch.float16).to("cuda")

def rt(a):
    x = torch.from_numpy(a.copy()).float().div(127.5).sub(1).permute(2,0,1)[None].half().cuda()
    with torch.no_grad():
        y = vae.decode(vae.encode(x).latent_dist.mode()).sample
    return ((y[0].permute(1,2,0).float().cpu().numpy()+1)*127.5).clip(0,255)

def collect(d):
    fs=[]
    for e in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG"): fs.extend(glob(os.path.join(d,e)))
    return sorted(fs, key=lambda p: os.path.basename(p))

dirs = sorted([d for d in glob(os.path.join(RAW,"*")) if os.path.isdir(d)])[:N_DIRS]

print(f"{'k':>3s} {'変化率%':>8s} {'全体GT':>8s} {'全体床':>8s} {'全体SNR':>8s} "
      f"{'マスクGT':>9s} {'マスク床':>9s} {'マスクSNR':>9s}")
for k in KS:
    rates, g_all, f_all, g_msk, f_msk = [], [], [], [], []
    for d in dirs:
        fr = collect(d)
        if len(fr) < N_FRAMES: continue
        imgs = [np.asarray(Image.open(p).convert("RGB").resize((R,R)), dtype=np.float32)
                for p in fr[:N_FRAMES]]
        for i in range(len(imgs)-k):
            A, B = imgs[i], imgs[i+k]
            d_map = np.abs(B-A).mean(axis=2)
            m = d_map > THRESH                      # 変化領域マスク
            if m.sum() < 10: continue               # 変化がほぼ無いペアは除外
            vA = rt(A)
            e_map = np.abs(vA-A).mean(axis=2)       # VAE 誤差マップ
            rates.append(m.mean()*100)
            g_all.append(d_map.mean());  f_all.append(e_map.mean())
            g_msk.append(d_map[m].mean()); f_msk.append(e_map[m].mean())
    if not g_msk:
        print(f"{k:3d}  (有効ペアなし)"); continue
    ga, fa = np.mean(g_all), np.mean(f_all)
    gm, fm = np.mean(g_msk), np.mean(f_msk)
    print(f"{k:3d} {np.mean(rates):8.3f} {ga:8.2f} {fa:8.2f} {ga/fa:8.2f} "
          f"{gm:9.2f} {fm:9.2f} {gm/fm:9.2f}")
