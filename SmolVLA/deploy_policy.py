from typing import Any, Dict
from pathlib import Path
import torch
import numpy as np
import cv2
import os
import sys

SMOLVLA_SRC = os.environ.get("SMOLVLA_SRC", "/data0/lumina/wenjun/SmolVLA-MoE/src")
if SMOLVLA_SRC and SMOLVLA_SRC not in sys.path:
    sys.path.insert(0, SMOLVLA_SRC)

from .smolvla_model import SmolVLA
from lerobot.policies.factory import make_pre_post_processors
from safetensors.torch import load_file
from lerobot.utils.constants import ACTION

def _prepare_img(img: np.ndarray) -> torch.Tensor:
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    if img.max() > 1.0:
        img = img / 255.0
    return torch.from_numpy(img)

def encode_obs(observation: Dict, model: SmolVLA) -> Dict:
    img_head = observation["observation"]["head_camera"]["rgb"]
    img_right = observation["observation"]["right_camera"]["rgb"]
    img_left = observation["observation"]["left_camera"]["rgb"]
    state = observation["joint_action"]["vector"]
    state_np = np.array(state, dtype=np.float32)
    state_tensor = torch.from_numpy(state_np)
    obs = {
        "observation.state": state_tensor,
        "observation.images.cam_high": _prepare_img(img_head),
        "observation.images.cam_right_wrist": _prepare_img(img_right),
        "observation.images.cam_left_wrist": _prepare_img(img_left),
        "task": getattr(model, "_instruction", None),
    }
    if hasattr(model, "_preprocessor") and model._preprocessor is not None:
        try:
            obs = model._preprocessor(obs)
        except Exception as e:
            print(f"[SmolVLA] Preprocessor failed, using raw obs: {e}")
            sys.exit(1)
    return obs

def get_model(usr_args: Dict) -> SmolVLA:
    device = usr_args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    policy_path = Path(usr_args["policy_path"]) if isinstance(usr_args.get("policy_path"), str) else usr_args["policy_path"]
    print(f"[SmolVLA] Loading model from: {policy_path}")
    model = SmolVLA.from_pretrained(str(policy_path))
    model.to(device)
    model.eval()
    model.reset()
    model.config.device = device
   
    try:
        preprocessor, postprocessor = make_pre_post_processors(model.config, pretrained_path=str(policy_path))
        model._preprocessor = preprocessor
        model._postprocessor = postprocessor
        print("[SmolVLA] Loaded pre/post processors from checkpoint")
    except Exception as e:
        print(f"[SmolVLA] Could not load processors from checkpoint: {e}")
        sys.exit(1)
    
    model._instruction = None
    return model

def eval(TASK_ENV: Any, model: SmolVLA, observation: Dict) -> None:
    if getattr(model, "_instruction", None) is None:
        model._instruction = TASK_ENV.get_instruction()
        print(f"[SmolVLA] Instruction: {model._instruction}")
    
    # 预测一次执行50steps——将50steps的action放到queue中
    if len(model._queues[ACTION]) == 0:
        obs = encode_obs(observation, model)
        with torch.no_grad():
            action_tensor = model.select_action(obs)
        try:
            if hasattr(model, "_postprocessor") and model._postprocessor is not None:
                action_tensor = model._postprocessor(action_tensor)
        except Exception as e:
            print(f"[SmolVLA] Postprocessor failed on first action: {e}")
            sys.exit(1)
    else:
        action_tensor = model._queues[ACTION].popleft()
        try:
            if hasattr(model, "_postprocessor") and model._postprocessor is not None:
                action_tensor = model._postprocessor(action_tensor)
        except Exception as e:
            print(f"[SmolVLA] Postprocessor failed on queued action: {e}")
            sys.exit(1)
        
    action = action_tensor.squeeze(0).detach().cpu().numpy()
    TASK_ENV.take_action(action, action_type="qpos")

def reset_model(model) -> None:
    if model:
        model._instruction = None
        model.reset()
