# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig # 引入基础策略配置类
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature # 引入类型与特征定义
from lerobot.optim.optimizers import AdamWConfig # 引入 AdamW 优化器配置
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
) # 引入余弦退火 + 热身 调度器配置
from lerobot.policies.rtc.configuration_rtc import RTCConfig # 引入 RTC（实时分块）配置类型
from lerobot.utils.constants import OBS_IMAGES # 引入观测图像常量键名


@PreTrainedConfig.register_subclass("smolvla") # 注册本配置为 "smolvla" 子类，便于选择
@dataclass # 使用数据类简化字段定义与默认值
class SmolVLAConfig(PreTrainedConfig): # SmolVLA 策略配置主体
    # Input / output structure. # 输入/输出结构设置
    n_obs_steps: int = 1 # 每次前向使用的历史观测步数（通常为 1）
    chunk_size: int = 50 # 单次模型调用可生成的动作序列长度上限
    n_action_steps: int = 50 # 实际需要解码的动作步数（<= chunk_size）

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY, # 图像不归一（由视觉塔处理）
            "STATE": NormalizationMode.MEAN_STD, # 状态按均值方差归一化
            "ACTION": NormalizationMode.MEAN_STD, # 动作按均值方差归一化
        }
    ) # 各模态归一化策略映射

    # Shorter state and action vectors will be padded # 维度不足时填充到统一长度
    max_state_dim: int = 32 # 状态向量的最大维度（用于线性投影）
    max_action_dim: int = 32 # 动作向量的最大维度（用于线性投影）

    arm_dims: list[int] | None = None
    group_arms: list[int] | None = None

    # Image preprocessing # 图像预处理参数
    resize_imgs_with_padding: tuple[int, int] = (512, 512) # 图像短边填充后统一到 512×512

    # Add empty images. Used by smolvla_aloha_sim which adds the empty # 为仿真添加空相机占位
    # left and right wrist cameras in addition to the top camera. # 除顶置相机外添加左右腕空相机
    empty_cameras: int = 0 # 添加空相机数量（占位输入，统一序列结构）

    # Converts the joint and gripper values from the standard Aloha space to # 关节与夹爪空间转换
    # the space used by the pi internal runtime which was used to train the base model. # 兼容 PI 运行时使用的训练空间
    adapt_to_pi_aloha: bool = False # 是否适配到 PI Aloha 空间（真实机型可能需要）

    # Converts joint dimensions to deltas with respect to the current state before passing to the model. # 关节动作改为相对增量
    # Gripper dimensions will remain in absolute values. # 夹爪保持绝对值
    use_delta_joint_actions_aloha: bool = False # 是否使用 Aloha 关节增量动作（LeRobot 尚未支持）

    # Tokenizer # 分词器相关设置
    tokenizer_max_length: int = 48 # 文本最大长度（限制语言令牌数量）

    # Decoding # 解码相关设置
    num_steps: int = 10 # 采样/推理迭代步数（用于流匹配/扩散过程）

    # Attention utils # 注意力相关优化
    use_cache: bool = True # 是否使用 KV 缓存（加速推理并支持 RTC）

    # Finetuning settings # 微调相关开关
    freeze_vision_encoder: bool = True # 冻结视觉编码器参数，避免过拟合
    train_expert_only: bool = True # 仅训练动作专家（Action Expert），冻结主 VLM
    train_state_proj: bool = True # 是否训练状态投影层（映射到文本隐藏维度）

    # Training presets # 训练超参数预设
    optimizer_lr: float = 1e-4 # 学习率（AdamW）
    optimizer_betas: tuple[float, float] = (0.9, 0.95) # AdamW 动量项 betas
    optimizer_eps: float = 1e-8 # 数值稳定项 epsilon
    optimizer_weight_decay: float = 1e-10 # 权重衰减（轻微正则化）
    optimizer_grad_clip_norm: float = 10 # 梯度截断范数阈值

    scheduler_warmup_steps: int = 1_000 # 学习率热身步数
    scheduler_decay_steps: int = 30_000 # 余弦退火总步数
    scheduler_decay_lr: float = 2.5e-6 # 退火后最低学习率

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone. # 主干 VLM 模型名称
    load_vlm_weights: bool = True  # Set to True in case of training the expert from scratch. True when init from pretrained SmolVLA weights # 是否加载主干权重

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features. # 是否在图像特征前后添加特殊令牌

    attention_mode: str = "cross_attn" # 注意力模式："cross_attn" 或 "self_attn"
    # cross_attn：VLM 产出 KV 缓存，动作专家作为查询对 KV 做跨注意力以条件化动作解码（参见 smolvlm_with_expert.py:275-386）

    prefix_length: int = -1 # 前缀长度（在输入序列前添加的特殊令牌数；-1 表示由处理器决定）

    pad_language_to: str = "longest"  # "max_length" # 文本填充策略：与批次中最长对齐或固定最大长度

    # 动作专家层数：<=0 表示与 VLM 层数一致；>0 表示减少
    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    # 使用的 VLM 层数（裁剪前 N 层）
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers) 
    # 每隔 N 层使用自注意力以稳定特征对齐
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers 
    # 专家隐藏维度相对 VLM 的缩放系数
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM) 
    
    # Action Expert 详解：
    # - 深度（num_expert_layers）：可与 VLM 同步或按整数倍间隔挂载（smolvlm_with_expert.py:101-107, 393-402）。层数减少可降算力但需保证层次对齐。
    # - 宽度（expert_width_multiplier）：专家隐藏维度按比例缩小（smolvlm_with_expert.py:95-99, 133-134），在效率与表达之间权衡（通常 0.5~0.75）。
    # - 交错自注意（self_attn_every_n_layers）：每隔 N 层采用自注意而非跨注意，维持专家自身表征稳定性并减少对 KV 的过度依赖。
    # - 注意力模式（attention_mode）：cross_attn 下专家使用 VLM 的 KV 作为条件输入进行解码；self_attn 下专家独立注意（见 forward 的分支逻辑）。

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding # 正弦-余弦位置编码最小周期
    max_period: float = 4.0 # 正弦-余弦位置编码最大周期

    # Real-Time Chunking (RTC) configuration # 实时分块配置
    rtc_config: RTCConfig | None = None # RTC 配置对象或 None（启用后进行流式窗口推理）

    def __post_init__(self): # 初始化后校验
        super().__post_init__()

        """Input validation (not exhaustive).""" # 输入校验（非穷尽）
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            ) # 动作步数不得超过分块上限
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            ) # Aloha 关节增量动作尚未在 LeRobot 中实现

    def validate_features(self) -> None: # 特征校验与补充
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera # 为每个空相机添加视觉占位特征

    def get_optimizer_preset(self) -> AdamWConfig: # 构造优化器预设
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self): # 构造调度器预设
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property # 只读属性：观测增量索引
    def observation_delta_indices(self) -> list:
        return [0] # 仅当前步（0）参与增量计算

    @property # 只读属性：动作增量索引
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size)) # 覆盖整个分块区间 [0, chunk_size)

    @property # 只读属性：奖励增量索引
    def reward_delta_indices(self) -> None:
        return None # 不计算奖励增量（None）
