#!/usr/bin/env python3
import argparse, json, os
import numpy as np
from PIL import Image

def make_diff(img_a, img_b, threshold):
    a = np.asarray(img_a, dtype=np.int16)
    b = np.asarray(img_b, dtype=np.int16)
    delta = np.abs(b - a).max(axis=2)
    diff = np.full_like(b, 255, dtype=np.uint8)
    mask = delta > threshold
    diff[mask] = b[mask].astype(np.uint8)
    return Image.fromarray(diff), float(mask.mean())

def process_split(root_in, root_out, split, threshold, stats):
    dir_a_in  = os.path.join(root_in,  split + "_A")
    dir_b_in  = os.path.join(root_in,  split + "_B")
    dir_a_out = os.path.join(root_out, split + "_A")
    dir_b_out = os.path.join(root_out, split + "_B")
    os.makedirs(dir_a_out, exist_ok=True)
    os.makedirs(dir_b_out, exist_ok=True)
    names = sorted(os.listdir(dir_a_in))
    for i, name in enumerate(names):
        img_a = Image.open(os.path.join(dir_a_in, name)).convert("RGB")
        img_b = Image.open(os.path.join(dir_b_in, name)).convert("RGB")
        diff, ratio = make_diff(img_a, img_b, threshold)
        img_a.save(os.path.join(dir_a_out, name))
        diff.save(os.path.join(dir_b_out, name))
        stats.append(ratio)
        if (i + 1) % 1000 == 0:
            print("  " + split + ": " + str(i+1) + "/" + str(len(names)))
    print(split + ": " + str(len(names)) + " pairs done")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root",  default="data/murayama_full")
    ap.add_argument("--out_root", default="data/murayama_diff")
    ap.add_argument("--threshold", type=int, default=25)
    ap.add_argument("--prompt", default="new brush strokes on white background")
    args = ap.parse_args()
    stats = []
    for split in ("train", "test"):
        process_split(args.in_root, args.out_root, split, args.threshold, stats)
        dir_a = os.path.join(args.out_root, split + "_A")
        prompts = {n: args.prompt for n in sorted(os.listdir(dir_a))}
        with open(os.path.join(args.out_root, split + "_prompts.json"), "w") as f:
            json.dump(prompts, f, indent=2)
    s = np.array(stats)
    print("change ratio: mean {:.3%} / median {:.3%} / max {:.3%}".format(s.mean(), np.median(s), s.max()))
    print("out: " + args.out_root)

if __name__ == "__main__":
    main()
