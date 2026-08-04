cd ~/img2img-turbo
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup accelerate launch --gpu_ids 1 --mixed_precision no src/train_pix2pix_turbo.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="output/pix2pix_turbo/murayama_diff_256" \
    --dataset_folder="data/murayama_diff" \
    --resolution=256 \
    --train_batch_size=1 \
    --viz_freq 100 \
    --report_to "tensorboard" --tracker_project_name "murayama_stroke_diff" \
    > train_diff.log 2>&1 &
echo "差分予測の学習をバックグラウンドで開始しました。PID: $!"
