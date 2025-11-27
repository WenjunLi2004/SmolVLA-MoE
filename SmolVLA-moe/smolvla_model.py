import os
import sys
from collections import deque
from torch import Tensor

SMOLVLA_SRC = os.environ.get("SMOLVLA_SRC", "/data0/lumina/wenjun/SmolVLA-MoE/src")
if SMOLVLA_SRC and SMOLVLA_SRC not in sys.path:
    sys.path.insert(0, SMOLVLA_SRC)

from lerobot.utils.constants import ACTION
from lerobot.policies.smolvla_moe.modeling_smolvla import SmolVLAMoEPolicy
from lerobot.policies.smolvla_moe.configuration_smolvla import SmolVLAMoEConfig

class SmolVLAMoE(SmolVLAMoEPolicy):
    def __init__(
        self,
        config: SmolVLAMoEConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}