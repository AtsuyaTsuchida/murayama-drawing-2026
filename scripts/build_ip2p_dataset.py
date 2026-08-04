#!/usr/bin/env python3
"""murayama_full の A/B ペアを HuggingFace datasets 形式に変換して保存する。"""
import os
from datasets import Dataset, Features, Image as HFImage, Value

SRC = "data/murayama_full"
DST = "data/murayama_ip2p"
PROMPT = "add the next brush strokes in the artist's style"

def build(split):
    dir_a = os.path.join(SRC, f"{split}_A")
    dir_b = os.path.join(SRC, f"{split}_B")
    names = sorted(os.listdir(dir_a))
    data = {
        "input_image":  [os.path.join(dir_a, n) for n in names],
        "edited_image": [os.path.join(dir_b, n) for n in names],
        "edit_prompt":  [PROMPT] * len(names),
    }
    feats = Features({
        "input_image": HFImage(),
        "edited_image": HFImage(),
        "edit_prompt": Value("string"),
    })
    ds = Dataset.from_dict(data, features=feats)
    print(f"{split}: {len(ds)} examples")
    return ds

train = build("train")
train.save_to_disk(DST)
print("saved ->", DST)
