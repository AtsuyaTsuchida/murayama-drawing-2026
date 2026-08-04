cd ~/img2img-turbo
CUDA_VISIBLE_DEVICES=0 \
nohup accelerate launch --gpu_ids 0 --mixed_precision fp16 train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=timbrooks/instruct-pix2pix \
    --dataset_name=data/murayama_ip2p \
    --original_image_column=input_image \
    --edited_image_column=edited_image \
    --edit_prompt_column=edit_prompt \
    --resolution=256 \
    --train_batch_size=4 \
    --gradient_checkpointing \
    --conditioning_dropout_prob=0.05 \
    --max_train_steps=10000 \
    --checkpointing_steps=1000 --checkpoints_total_limit=2 --resume_from_checkpoint=latest \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --seed=42 \
    --output_dir=output/ip2p_murayama \
    > train_ip2p.log 2>&1 &
echo "InstructPix2Pix微調整を開始しました。PID: $!"
