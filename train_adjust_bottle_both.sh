export CUDA_VISIBLE_DEVICES=1
# Prefer local SmolVLA-MoE src when running lerobot-train
export PYTHONPATH="/data0/lumina/wenjun/SmolVLA-MoE/src:${PYTHONPATH}"
export HF_HOME=/data0/lumina/wenjun/.cache/huggingface
export XDG_CACHE_HOME=/data0/lumina/wenjun/.cache
#!/usr/bin/env bash

# Train SmolVLA on the local LeRobot dataset
# Notes:
# - Requires HF_LEROBOT_HOME set or explicit --dataset.root below
# - Adjust batch_size if you see CUDA OOM; start lower on small GPUs

lerobot-train \
  --policy.type=smolvla \
  --dataset.repo_id=adjust_bottle \
  --batch_size=32 \
  --steps=20000 \
  --output_dir=outputs/train/smolvla/adjust_bottle_both_test \
  --policy.vlm_model_name=/data0/lumina/wenjun/SmolVLA-MoE/SmolVLM2-500M\
  --job_name=smolvla_training_adjust_bottle \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --policy.group_arms="[1,2]" \
  --policy.arm_dims="[7,7]"\ 
