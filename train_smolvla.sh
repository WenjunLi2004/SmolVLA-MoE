export CUDA_VISIBLE_DEVICES=3

#!/usr/bin/env bash

# Train SmolVLA on the local LeRobot dataset
# Notes:
# - Requires HF_LEROBOT_HOME set or explicit --dataset.root below
# - Adjust batch_size if you see CUDA OOM; start lower on small GPUs

lerobot-train \
  --policy.type=smolvla \
  --dataset.repo_id=move_can_pot \
  --batch_size=1 \
  --steps=100 \
  --output_dir=outputs/train/smolvla/move_can_pot \
  --policy.vlm_model_name=/data0/lumina/wenjun/lerobot/SmolVLM2-500M\
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false