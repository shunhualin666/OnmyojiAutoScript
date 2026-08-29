"""拟人化策略（HumanPolicy）：区域内高斯落点 + 拟人轨迹。

默认不启用；通过 make_policy / 配置 operation.policy=human 启用。
复用拟人化模块的经验：落点在区域内截断、轨迹首尾固定、防卡死按逻辑动作计数。
"""
from __future__ import annotations

import numpy as np

from . import region as R
from .policy import Policy


class HumanPolicy(Policy):
    """拟人化操作策略。

    - 落点：区域内中心截断高斯（中心多、边缘稀疏，模拟真人点按钮）
    - 轨迹：最小 jerk 分段（首尾固定），多段由 OperationLayer 执行
    """

    name = 'human'

    def __init__(self, seed: int | None = None, spread: float = 1.0,
                 n_segments: int = 6):
        self.rng = np.random.default_rng(seed)
        self.spread = float(spread)
        self.n_segments = int(n_segments)

    def aim(self, region) -> tuple[int, int]:
        # 中心截断高斯，落点保证在区域内
        return R.region_normal(region, sigma_ratio=1 / 6, rng=self.rng)

    def path(self, region_a, region_b) -> list[tuple[int, int]]:
        a = R.region_center(region_a)
        b = R.region_center(region_b)
        # 最小 jerk 标准化曲线（首尾固定）
        t = np.linspace(0.0, 1.0, self.n_segments + 1)
        s = 10.0 * t ** 3 - 15.0 * t ** 4 + 6.0 * t ** 5
        path = [(int(round(a[0] + (b[0] - a[0]) * si)),
                 int(round(a[1] + (b[1] - a[1]) * si))) for si in s]
        path[0] = a
        path[-1] = b
        return path
