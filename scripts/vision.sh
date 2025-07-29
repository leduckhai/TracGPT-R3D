export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_LAUNCH_BLOCKING=1
export WANDB_DIR=./output/wandb

# Redirect stdout and stderr to a log file
python main_vision.py \
    --version v0 \
    --train_val_dir /root/TracGPT-R3D/pseudo_3d/32_overlap_slices/0691cd9f-8dad-4005-811d-34fb610d4f88/train/data \
    --dataset trac_white \

    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --model_type tiny_llama \
    --lora_enable True \
    --vision_tower vit3d \
    --bf16 0 \
    --fp16 1 \
    --output_dir ./output/TinyLLama-finetune-0000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    > training.log 2>&1