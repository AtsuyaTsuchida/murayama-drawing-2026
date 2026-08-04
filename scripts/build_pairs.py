#!/usr/bin/env python3
"""連番ストローク画像から pix2pix-turbo 用ペアデータセットを構築する。"""
import argparse, json, os, random
from glob import glob
from PIL import Image

def collect_frames(work_dir):
    files = []
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG"):
        files.extend(glob(os.path.join(work_dir, ext)))
    return sorted(files, key=lambda p: os.path.basename(p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--prompt", default="a black and white ink drawing")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    for split in ("train","test"):
        for ab in ("A","B"):
            os.makedirs(os.path.join(args.out_root, f"{split}_{ab}"), exist_ok=True)

    work_dirs = [d for d in glob(os.path.join(args.raw_root,"*")) if os.path.isdir(d)]
    def _key(d):
        b = os.path.basename(d)
        return (0,int(b)) if b.isdigit() else (1,b)
    work_dirs = sorted(work_dirs, key=_key)
    if not work_dirs:
        raise SystemExit(f"作品フォルダが見つかりません: {args.raw_root}")

    pairs = []
    for wd in work_dirs:
        frames = collect_frames(wd)
        if len(frames) < 2:
            print(f"  スキップ: {wd}"); continue
        for i in range(len(frames)-1):
            pairs.append((frames[i], frames[i+1]))
        print(f"  {os.path.basename(wd)}: {len(frames)} フレーム -> {len(frames)-1} ペア")
    print(f"総ペア数: {len(pairs)}")

    idx = list(range(len(pairs)))
    random.shuffle(idx)
    n_val = max(1, int(len(pairs)*args.val_ratio))
    val_set = set(idx[:n_val])

    train_prompts, test_prompts = {}, {}
    tr_count = te_count = 0
    R = args.resolution
    for k,(src_a,src_b) in enumerate(pairs):
        is_val = k in val_set
        split = "test" if is_val else "train"
        num = te_count if is_val else tr_count
        name = f"{num:06d}.png"
        Image.open(src_a).convert("RGB").resize((R,R),Image.LANCZOS).save(
            os.path.join(args.out_root,f"{split}_A",name))
        Image.open(src_b).convert("RGB").resize((R,R),Image.LANCZOS).save(
            os.path.join(args.out_root,f"{split}_B",name))
        if is_val:
            test_prompts[name]=args.prompt; te_count+=1
        else:
            train_prompts[name]=args.prompt; tr_count+=1

    with open(os.path.join(args.out_root,"train_prompts.json"),"w") as f:
        json.dump(train_prompts,f,indent=2)
    with open(os.path.join(args.out_root,"test_prompts.json"),"w") as f:
        json.dump(test_prompts,f,indent=2)
    print(f"完了: train {tr_count} / test {te_count}")
    print(f"出力先: {args.out_root}")

if __name__ == "__main__":
    main()
