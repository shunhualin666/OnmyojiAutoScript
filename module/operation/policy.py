"""操作策略：决定 区域 → 具体落点/轨迹/时长。

策略是操作中间层的核心扩展点：默认 / 拟人化 / AI 都是 Policy 的一种实现。
"""
from __future__ import annotations

from . import region as R


class Policy:
    """操作策略基类。

    子类实现 aim/path 即可定义"在区域内如何落点、区域间如何滑动"。
    """

    name: str = 'base'

    def aim(self, region) -> tuple[int, int]:
        """区域内落点。"""
        raise NotImplementedError

    def path(self, region_a, region_b) -> list[tuple[int, int]]:
        """区域 A -> 区域 B 的轨迹点（首尾固定为两端点）。"""
        raise NotImplementedError

    def duration(self, distance: float) -> float | None:
        """一次动作的时长（秒）；返回 None 使用设备默认。"""
        return None

    # ------------------------------------------------------------------
    # 节奏钩子（默认零开销；拟人化策略覆盖）
    # ------------------------------------------------------------------
    def after_click(self) -> float:
        """点击后停顿（秒）。"""
        return 0.0

    def after_swipe(self) -> float:
        """滑动后停顿（秒）。"""
        return 0.0

    def move_delay(self) -> float:
        """分段滑动段间延迟（秒）。"""
        return 0.0

    def interval(self, base: float = 0.0) -> float:
        """拟人化操作间隔（秒）。"""
        return float(base)

    def record(self, dt: float = 0.0) -> None:
        """记录一次操作耗时（推进疲劳/学习）。"""
        pass


class DefaultPolicy(Policy):
    """默认策略：与现状行为完全一致。

    - 落点：区域内均匀随机（等价 RuleClick.coord()）
    - 轨迹：起点 -> 终点 直线（一次滑动）
    """

    name = 'default'

    def aim(self, region) -> tuple[int, int]:
        return R.region_random(region)

    def path(self, region_a, region_b) -> list[tuple[int, int]]:
        return [R.region_center(region_a), R.region_center(region_b)]
