#!/usr/bin/env bash

python src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id=/data0/lumina/wenjun/.cache/huggingface/lerobot/handover_block\
    --push-to-hub=false