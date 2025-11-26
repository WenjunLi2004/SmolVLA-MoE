#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
from collections import deque
from typing import TypedDict

import os
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.pretrained import PreTrainedPolicy  # 预训练策略基类
from lerobot.policies.rtc.modeling_rtc import RTCProcessor  # 实时分块处理器
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # SmolVLA 配置
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel  # 引入带动作专家（Action Expert）的 VLM 封装
from lerobot.policies.utils import (
    populate_queues,  # 将批次数据入队以便按窗口提取
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype


class ActionSelectKwargs(TypedDict, total=False):  # 选择动作时可选的关键字参数（RTC/延迟等）
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None
    group_arms: list[int] | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
      # 为时间标量生成正弦-余弦位置编码（维度需为偶数）
    if dimension % 2 != 0:  # 维度必须为偶数
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:  # 时间输入需为一维（批大小）
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)  # 在不同设备上选择安全的浮点类型
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)  # 均匀分布的比例（长度为一半维度）
    period = min_period * (max_period / min_period) ** fraction  # 周期按指数从 min 到 max 平滑变化

    # Compute the outer product  # 计算时间与频率的外积
    scaling_factor = 1.0 / period * 2 * math.pi  # 频率缩放因子
    sin_input = scaling_factor[None, :] * time[:, None]  # 扩展并乘法得到各频率上的输入
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)  # 拼接正弦与余弦分量
    return pos_emb  # 返回位置编码


def make_att_2d_masks(pad_masks, att_masks):  # 将一维的填充与注意边界掩码转换为二维注意力掩码
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:  # 期望 att_masks 为二维（B×N）
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:  # 期望 pad_masks 为二维（B×N）
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)  # 累积和形成分段界限（prefix/causal 等）
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]  # 基于界限构造二维可访问关系
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]  # 同时考虑 padding 掩码
    att_2d_masks = att_2d_masks & pad_2d_masks  # 仅允许非 padding 的位置参与注意力
    return att_2d_masks  # 返回最终二维注意力掩码


def resize_with_pad(img, width, height, pad_value=-1):  # 比例缩放并左/上填充到目标尺寸
    # assume no-op when width height fits already  # 若尺寸已匹配则无操作
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")  # 需要 (B, C, H, W)

    cur_height, cur_width = img.shape[2:]  # 当前高宽

    ratio = max(cur_width / width, cur_height / height)  # 按长边对齐保持比例
    resized_height = int(cur_height / ratio)  # 缩放后高度
    resized_width = int(cur_width / ratio)  # 缩放后宽度
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )  # 双线性插值缩放

    pad_height = max(0, int(height - resized_height))  # 顶部填充高度
    pad_width = max(0, int(width - resized_width))  # 左侧填充宽度

    # pad on left and top of image  # 左侧与顶部填充
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img  # 返回填充结果


def pad_vector(vector, new_dim):  # 将向量最后一维填充到 new_dim，保持前置维度不变
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:  # 若维度已匹配则直接返回
        return vector
    shape = list(vector.shape)  # 复制形状
    current_dim = shape[-1]  # 原有最后一维长度
    shape[-1] = new_dim  # 替换为新维度
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)  # 按新维度创建零张量
    new_vector[..., :current_dim] = vector  # 拷贝原值到前 current_dim 段
    return new_vector  # 返回填充后的向量


def normalize(x, min_val, max_val):  # 线性归一化到 [0,1]
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):  # 线性反归一化到原始范围
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):  # 对输入裁剪到 [-1,1] 后取 arcsin
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):  # 将 Aloha 线性夹爪位姿反变换到角度空间并归一化
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)  # 反归一化到线性长度（米）

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)  # 由几何关系求角度
        return safe_arcsin(value)  # 安全 arcsin

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)  # 机械臂长度与舵盘半径常数

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)  # 归一化到角度经验区间


def aloha_gripper_from_angular(value):  # 将角度空间转换为 Aloha 所用线性夹爪空间
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)  # 反归一化到角度值

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)  # 归一化到线性行程范围


def aloha_gripper_from_angular_inv(value):  # 反转 gripper_from_angular 映射
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)  # 反归一化到线性值
    return normalize(value, min_val=0.4, max_val=1.5)  # 归一化到角度区间


class SmolVLAPolicy(PreTrainedPolicy):  # 策略包装类：封装 VLAFlowMatching 用于训练与推理
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)  # 调用父类初始化
        config.validate_features()  # 校验/补充视觉特征（空相机等）
        self.config = config  # 保存配置
        self.init_rtc_processor()  # 初始化 RTC 处理器（可选，用于流式/在线推理）
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)  # 主模型组合：VLM + Action Expert
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),  # 动作清空缓冲队列（长度为 n_action_steps）
        }

    def init_rtc_processor(self):  # 初始化 RTC 处理器
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None  # 默认无处理器

        # 如果提供了 RTC 配置则创建处理器；即便未启用也可用于调试记录去噪数据
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:  # 返回将参与优化的参数集合
        return self.parameters()  # 默认返回整模型参数；微调范围由各模块 requires_grad 控制

    def visualize_action_expert(self) -> str:  # 可视化专家层与 VLM 层的映射关系，便于检查层级对齐
        return self.model.vlm_with_expert.visualize_layer_mapping()  # 调用底层映射可视化接口

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.  # 待确认：该循环是否必要
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
          # 背景：推理时批次通常不包含 ACTION，队列仅维护 ACTION
        # In the case of offline inference, we have the action in the batch  # 离线推理可能包含 ACTION
        # that why without the k != ACTION check, it will raise an error because we are trying to stack  # 因此需跳过 ACTION，避免在空容器上 stack 报错
        # on an empty container.  # 解释：若容器为空直接 stack 会触发错误
        for k in batch:  # 遍历批次键
            if k in self._queues and k != ACTION:  # 仅当该键在队列中且不是 ACTION
                batch[k] = torch.stack(list(self._queues[k]), dim=1)  # 将队列内容按时间维堆叠回批次

        images, img_masks = self.prepare_images(batch)  # 图像预处理并生成掩码（归一化、缩放、padding）
        state = self.prepare_state(batch)  # 状态填充至统一维度（max_state_dim）
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]  # 语言 token 张量（形状 B×L）
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]  # 语言 padding 掩码（形状 B×L）

        actions = self.model.sample_actions(  # 调用主模型进行动作推理（迭代去噪得到 x_t）
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )

        # Unpad actions  # 去除填充维度对齐到原始动作维度
        original_action_dim = self.config.action_feature.shape[0]  # 原始动作特征维度（不含 padding）
        actions = actions[:, :, :original_action_dim]  # 截取前 original_action_dim 维

        if self.config.adapt_to_pi_aloha:  # 如启用 Aloha 空间适配，则对动作做转换（角度→线性等）
            actions = self._pi_aloha_encode_actions(actions)  # 应用编码以匹配 Aloha 运行时

        return actions  # 返回动作序列（B×n_action_steps×action_dim）

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:  # 批次前处理：按需做 Aloha 空间解码
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:  # 批量预测动作序列（支持 RTC）
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )  # 单步选择不支持 RTC，需使用 predict_action_chunk

        self.eval()  # 切换到评估模式
        batch = self._prepare_batch(batch)  # 批次预处理
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])  # 入队非 ACTION 键

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:  # 检查动作队列是否为空，决定是否触发一次推理
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:  # 判断是否启用 RTC 模式
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss"""  # 训练前向：构造流匹配损失
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)  # 前缀：图像嵌入（SigLIP）
        state = self.prepare_state(batch)  # 前缀：状态线性映射到文本隐藏维度
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)  # 后缀：动作填充到统一维度
        actions_is_pad = batch.get("actions_id_pad")  # 动作是否在序列外（episode 边界）
        loss_dict = {}  # 记录各阶段的损失以便调试
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time, group_arms=self.config.group_arms)  # 经 VLM+Expert 前向得到速度 v_t 并计算 MSE(u_t, v_t)
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad  # 取反得到有效范围
            losses = losses * in_episode_bound.unsqueeze(-1)  # 屏蔽 episode 外的损失
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]  # 去除对齐填充的维度，仅保留原始动作维度
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # For backward pass
        loss = losses.mean()  # 标量化损失用于反向传播
        # For backward pass
        loss_dict["loss"] = loss.item()  # 保存数值型损失
        return loss, loss_dict  # 返回标量损失及中间过程

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []  # 存放预处理后的图像张量（按批次）
        img_masks = []  # 存放图像的 padding 掩码（按批次）
        present_img_keys = [key for key in self.config.image_features if key in batch]  # 批次中存在的图像特征键
        missing_img_keys = [key for key in self.config.image_features if key not in batch]  # 批次中缺失的图像特征键

        if len(present_img_keys) == 0:  # 没有任何图像输入时直接报错（至少需要一个图像模态）
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:  # 逐个图像特征键进行预处理
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]  # 5D（视频）取最后一帧，否则直接取图像张量
            if self.config.resize_imgs_with_padding is not None:  # 若配置启用尺寸对齐
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)  # 保持比例缩放并填充到目标尺寸

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0  # 像素值从 [0,1] 线性映射到 SigLIP 期望的 [-1,1]

            bsize = img.shape[0]  # 批大小
            device = img.device  # 当前张量设备
            if f"{key}_padding_mask" in batch:  # 若批次内提供了该键的 padding 掩码
                mask = batch[f"{key}_padding_mask"].bool()  # 转为布尔掩码
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)  # 默认所有样本有效
            images.append(img)  # 追加处理后的图像张量
            img_masks.append(mask)  # 追加对应的掩码

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):  # 针对缺失的相机特征，按配置添加占位
            if num_empty_cameras >= self.config.empty_cameras:  # 限制占位数量不超过配置值
                break
            img = torch.ones_like(img) * -1  # 使用 -1（与规范化后图像一致的无效值）作为占位图像
            mask = torch.zeros_like(mask)  # 占位图像对应的掩码为全 padding
            images.append(img)  # 追加占位图像
            img_masks.append(mask)  # 追加占位掩码
        return images, img_masks  # 返回图像张量列表与其掩码列表

    def _pi_aloha_decode_state(self, state):  # 将 Aloha 运行时的状态解码到 SmolVLA 使用的角度空间
        # Flip the joints.  # 翻转指定关节的方向符号
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1  # 乘以 -1 进行符号翻转
        # Reverse the gripper transformation that is being applied by the Aloha runtime.  # 将夹爪的线性位姿反变换到角度空间
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])  # 调用转换函数
        return state  # 返回解码后的状态

    def _pi_aloha_encode_actions(self, actions):  # 将 SmolVLA 的角度空间动作编码到 Aloha 运行时的线性空间
        # Flip the joints.  # 翻转指定关节方向
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1  # 乘以 -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.  # 将角度空间的夹爪位姿转换到线性空间
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])  # 转换函数
        return actions  # 返回编码后的动作

    def _pi_aloha_encode_actions_inv(self, actions):  # 对编码过程的逆变换（用于训练时的目标动作）
        # Flip the joints again.  # 再次翻转关节方向
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.  # 使用逆变换函数还原夹爪位姿
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions  # 返回逆编码后的动作

    def prepare_state(self, batch):  # 提取并填充状态到统一维度
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]  # 若为时间序列取最后一帧
        state = pad_vector(state, self.config.max_state_dim)  # 填充到 max_state_dim
        return state  # 返回填充后的状态

    def prepare_action(self, batch):  # 填充动作到统一维度
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)  # 填充到 max_action_dim
        return actions  # 返回填充后的动作


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]  # 批大小与当前序列长度

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )  # 创建固定长度的填充张量（按 pad_value 填充）
    padded_tensor[:, :d] = tensor  # 将原始序列拷贝到前 d 段，保持后部为填充值

    return padded_tensor  # 返回对齐后的张量

# SmolVLA 流匹配主模型：组合 VLM 与动作专家，执行条件动作解码
class VLAFlowMatching(nn.Module):  
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):  # 初始化：实例化带专家的 VLM 与各投影层
        super().__init__()
        self.config = config
        
        # 动作专家（Action Expert）通过该封装与 VLM 交互
        self.vlm_with_expert = SmolVLMWithExpertModel(  
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )  # 将状态映射到 VLM 文本隐藏维度，用作前缀条件
        self.action_in_proj = nn.Linear(
            self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size
        )  # 将动作映射到专家隐藏维度，用作后缀输入
        self.action_out_proj = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim
        )  # 将专家输出映射回动作空间（预测速度 v_t）

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )  # 将动作+时间特征融合至专家维度（前 MLP）
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )  # 融合后的非线性变换（后 MLP）

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id  # 图像特殊令牌：占位/边界标记
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id  # 图像全局令牌：表示图像开始
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens  # 是否启用图像特殊令牌（前缀序列边界更清晰）
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor  # 绑定 RTC 处理器到主模型（用于流式去噪记录与约束）

    def _rtc_enabled(self):  # 判断是否启用 RTC 模式
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):  # 设置各投影层是否参与训练（按配置开关）
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj  # 控制 state_proj 是否训练；其余由上层封装控制

    def sample_noise(self, shape, device):  # 采样标准高斯噪声，用作流匹配输入或初始状态
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise  # 返回形状与动作一致的噪声张量

    def sample_time(self, bsize, device):  # 从 Beta(1.5,1.0) 分布采样时间步 t∈(0,1)
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001  # 轻微远离 0/1 避免数值边界问题
        return time  # 返回每个样本的时间标量

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """  # 前缀嵌入：图像经视觉塔，语言用嵌入层，状态线性映射；并构造 pad/att 掩码
        embs = []  # 前缀序列的嵌入列表
        pad_masks = []  # 前缀序列的 padding 掩码列表
        att_masks = []  # 前缀序列的注意力边界标记（后续转二维）
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):  # 逐图像模态嵌入
            if self.add_image_special_tokens:  # 在图像序列前插入特殊令牌，便于明确边界
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)  # 调用视觉塔 + connector 获取图像序列嵌入
            img_emb = img_emb  # 直接使用得到的嵌入

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]  # 隐藏维度
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)  # 按隐藏维度开方缩放，稳定注意力数值

            bsize, num_img_embs = img_emb.shape[:2]  # 批大小与图像令牌数
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)  # 将每图像掩码扩展到每个令牌

            embs.append(img_emb)  # 拼接图像嵌入到前缀序列
            pad_masks.append(img_mask)  # 图像序列的 padding 掩码

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:  # 在图像序列后插入结束令牌
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)  # 语言令牌嵌入
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]  # 隐藏维度
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)  # 同样按维度开方缩放

        embs.append(lang_emb)  # 拼接语言嵌入
        pad_masks.append(lang_masks)  # 拼接语言掩码

        num_lang_embs = lang_emb.shape[1]  # 语言令牌数
        att_masks += [0] * num_lang_embs  # 标记语言令牌的注意力边界

        state_emb = self.state_proj(state)  # 将状态映射到文本隐藏维度，作为前缀条件
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb  # 若为 (B,D) 则添加序列维
        embs.append(state_emb)  # 拼接状态嵌入
        bsize = state_emb.shape[0]  # 批大小
        device = state_emb.device  # 设备

        states_seq_len = state_emb.shape[1]  # 状态令牌数
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)  # 状态均有效
        pad_masks.append(state_mask)  # 拼接状态掩码

        # 设置注意力掩码：图像/语言不关注状态或动作（隔离前缀与后缀的注意范围）
        att_masks += [1] * (states_seq_len)  # 前缀边界：状态之后不允许被图像/语言关注
        embs = torch.cat(embs, dim=1)  # 拼接得到完整前缀嵌入序列
        pad_masks = torch.cat(pad_masks, dim=1)  # 拼接前缀 padding 掩码
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)  # 转为张量
        att_masks = att_masks[None, :]  # 添加批维

        seq_len = pad_masks.shape[1]  # 前缀总长度
        if seq_len < self.prefix_length:  # 对齐到固定前缀长度，便于批次内位置索引一致
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)  # 嵌入补零
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)  # 掩码补零
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)  # 注意掩码补零

        att_masks = att_masks.expand(bsize, -1)  # 扩展到批维

        return embs, pad_masks, att_masks  # 返回前缀嵌入与掩码

    def embed_suffix(self, noisy_actions, timestep):  # 后缀嵌入：将带噪动作与时间编码融合，提供给动作专家
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []  # 后缀序列嵌入列表
        pad_masks = []  # 后缀序列 padding 掩码列表
        att_masks = []  # 后缀序列注意力边界标记

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)  # 动作投影到专家隐藏维度
        device = action_emb.device  # 设备
        bsize = action_emb.shape[0]  # 批大小
        dtype = action_emb.dtype  # 数据类型
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(  # 时间正弦-余弦编码，维度与专家隐藏维度一致
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)  # 与动作嵌入 dtype 对齐

        time_emb = time_emb[:, None, :].expand_as(action_emb)  # 扩展到动作序列形状
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)  # 拼接动作与时间嵌入

        action_time_emb = self.action_time_mlp_in(action_time_emb)  # 动作+时间融合的前 MLP
        action_time_emb = F.silu(action_time_emb)  # swish == silu 非线性
        action_time_emb = self.action_time_mlp_out(action_time_emb)  # 融合后的后 MLP

        # Add to input tokens
        embs.append(action_time_emb)  # 添加到后缀序列

        bsize, action_time_dim = action_time_emb.shape[:2]  # 批大小与后缀长度
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)  # 全部有效
        pad_masks.append(action_time_mask)  # 添加掩码

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size  # 后缀为动作序列，设置边界以形成分段注意（配合 make_att_2d_masks）
        embs = torch.cat(embs, dim=1)  # 拼接后缀嵌入
        pad_masks = torch.cat(pad_masks, dim=1)  # 拼接后缀掩码
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)  # 转为张量
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))  # 扩展到批维
        return embs, pad_masks, att_masks  # 返回后缀嵌入与掩码

    def get_group_mask(self, group_arms: list[int] | None, bsize: int, device: torch.device) -> torch.Tensor:
        if group_arms is not None and self.config.arm_dims is not None and len(self.config.arm_dims) > 0:
            mask = torch.zeros(bsize, self.config.chunk_size, self.config.max_action_dim, dtype=torch.float32, device=device)
            start = 0
            for i, dim in enumerate(self.config.arm_dims):
                end = start + dim
                arm_id = i + 1
                if arm_id in group_arms:
                    if end > start:
                        idxs = torch.arange(start, end, dtype=torch.long, device=device)
                        mask[:, :, idxs] = 1.0
                start = end
            return mask
        return torch.ones(bsize, self.config.chunk_size, self.config.max_action_dim, dtype=torch.float32, device=device)

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None, group_arms: list[int] | None = None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""  # 流匹配训练：预测 v_t 拟合 u_t
        if noise is None:  # 训练时若未提供噪声，则按动作形状采样
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:  # 训练时若未提供时间，则按 Beta 分布采样 t∈(0,1)
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        group_arms = group_arms if group_arms is not None else self.config.group_arms
        group_mask = self.get_group_mask(group_arms, actions.shape[0], actions.device)
        # TEST
        if os.getenv("SMOLVLA_DEBUG_MASK", "0") == "1" and not getattr(self, "_mask_debug_train_printed", False):
            try:
                enabled = torch.nonzero(group_mask[0, 0], as_tuple=False).squeeze(-1).tolist()
                start = 0
                segs = []
                for i, dim in enumerate(self.config.arm_dims or []):
                    end = start + dim
                    seg_enabled = [d for d in enabled if start <= d < end]
                    segs.append((i + 1, start, end, seg_enabled))
                    start = end
                print(f"[SmolVLA][train] group_arms={group_arms} arm_dims={self.config.arm_dims} enabled_dims@t0={enabled} segments={segs}")
                if os.getenv("SMOLVLA_DEBUG_MASK_PDB", "0") == "1":
                    import pdb
                    pdb.set_trace()
            except Exception:
                pass
            setattr(self, "_mask_debug_train_printed", True)
        x_t = x_t * group_mask
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )  # 生成前缀嵌入与掩码
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)  # 生成后缀嵌入与掩码

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)  # 拼接 pad 掩码
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)  # 拼接 att 掩码

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)  # 生成二维注意力掩码（含前后缀分界与 padding）
        position_ids = torch.cumsum(pad_masks, dim=1) - 1  # 计算位置索引（仅对非 padding 递增）
        (_, suffix_out), _ = self.vlm_with_expert.forward(  # 调用 VLM+Expert 前向：前缀生成 KV，后缀由专家条件解码
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]  # 仅保留动作序列对应的后缀输出
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t * group_mask
        losses = F.mse_loss(u_t, v_t, reduction="none")
        losses = losses * group_mask
        return losses

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""  # 推理：迭代去噪得到动作序列
        bsize = state.shape[0]  # 批大小
        device = state.device  # 当前设备

        group_arms = kwargs.get("group_arms", self.config.group_arms)
        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)  # 形状：B×chunk×dim
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(  # 仅前缀嵌入用于构建 KV 缓存
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)  # 前缀二维注意力掩码
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1  # 前缀位置索引
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        dt = -1.0 / self.config.num_steps  # 负时间步长（从 1 到 0 回退）
        dt = torch.tensor(dt, dtype=torch.float32, device=device)  # 张量化以便计算

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)  # 从 t=1 开始向 0 迭代

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)  # 扩展到批维

            # Define a closure function to properly capture expanded_time
            # This avoids the lambda expression (E731) and loop variable binding (B023) issues
            def denoise_step_partial_call(input_x_t, current_timestep=expanded_time):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                    group_arms=group_arms,
                )

            if self._rtc_enabled():  # 流式 RTC 模式
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(  # RTC 模式：在流式执行约束下进行单步去噪
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:  # 非 RTC 模式
                v_t = denoise_step_partial_call(x_t)

            # Euler step
            group_mask = self.get_group_mask(group_arms, bsize, device)
            x_t += dt * (v_t * group_mask)

            # Record x_t and v_t after Euler step (other params are recorded in rtc_processor.denoise_step)
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)  # 记录轨迹用于调试与可视化

            time += dt  # 时间向 0 收敛

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        group_arms: list[int] | None = None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""  # 单步去噪：仅后缀前向，复用前缀 KV
        bsize = prefix_pad_masks.shape[0]
        device = x_t.device
        group_mask = self.get_group_mask(group_arms, bsize, device)
        x_t = x_t * group_mask
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]  # 后缀序列长度
        batch_size = prefix_pad_masks.shape[0]  # 批大小
        prefix_len = prefix_pad_masks.shape[1]  # 前缀序列长度
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)  # 扩展前缀 pad 掩码以拼接形成完整二维掩码

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)  # 后缀二维注意力掩码

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)  # 拼接得到完整二维注意力掩码
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]  # 位置偏移：后缀位置从前缀有效长度开始计数
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1  # 后缀位置索引（不计 padding）

        outputs_embeds, _ = self.vlm_with_expert.forward(  # 仅后缀输入 + 前缀 KV：专家跨注意从 KV 中检索条件
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]  # 取专家对应的输出分支
        suffix_out = suffix_out[:, -self.config.chunk_size :]  # 仅保留动作序列的后缀部分
        suffix_out = suffix_out.to(dtype=torch.float32)  # 上采到 float32 以匹配投影头精度
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t * group_mask
        # TEST
        if os.getenv("SMOLVLA_DEBUG_MASK", "0") == "1" and not getattr(self, "_mask_debug_infer_printed", False):
            try:
                enabled = torch.nonzero(group_mask[0, 0], as_tuple=False).squeeze(-1).tolist()
                start = 0
                segs = []
                for i, dim in enumerate(self.config.arm_dims or []):
                    end = start + dim
                    seg_enabled = [d for d in enabled if start <= d < end]
                    segs.append((i + 1, start, end, seg_enabled))
                    start = end
                print(f"[SmolVLA][infer] group_arms={group_arms} arm_dims={self.config.arm_dims} enabled_dims@t0={enabled} segments={segs}")
                if os.getenv("SMOLVLA_DEBUG_MASK_PDB", "0") == "1":
                    import pdb
                    pdb.set_trace()
            except Exception:
                pass
            setattr(self, "_mask_debug_infer_printed", True)
        return v_t
