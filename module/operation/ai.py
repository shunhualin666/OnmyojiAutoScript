"""AI 策略（AIPolicy）：由外部模型决策 / 仅占位。

接入方式：继承 Policy 并实现 aim/path 从外部 AI（LLM/RL）获取决策；
或直接替换 OperationLayer.policy。当前为占位实现。
"""
from __future__ import annotations

from .policy import Policy


class AIPolicy(Policy):
    """AI 操作策略（占位，未接入外部模型）。"""

    name = 'ai'

    def aim(self, region) -> tuple[int, int]:
        raise NotImplementedError('AIPolicy 尚未实现，请接入外部 AI 后再使用')

    def path(self, region_a, region_b) -> list[tuple[int, int]]:
        raise NotImplementedError('AIPolicy 尚未实现，请接入外部 AI 后再使用')
