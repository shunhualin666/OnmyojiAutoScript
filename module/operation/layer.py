"""操作中间层：区域指令 -> 策略 -> 物理执行。

任务代码只描述"在哪块区域做什么"（click(region) / swipe(region_a, region_b)），
不直接操纵坐标。具体落点、轨迹、物理参数由策略（Policy）决定，
并在此统一记录操作日志供 AI 学习。
"""
from __future__ import annotations

from .policy import Policy, DefaultPolicy


class OperationLayer:
    """统一操作入口。

    Args:
        device: 设备对象（device.click/swipe/long_click）。
        policy: 操作策略；默认 DefaultPolicy（零回归）。
        recorder: 操作记录器（可选，供 AI 学习）。
    """

    def __init__(self, device, policy: Policy | None = None, recorder=None):
        self.device = device
        self.policy = policy if policy is not None else DefaultPolicy()
        self.recorder = recorder

    # ------------------------------------------------------------------
    # 点击
    # ------------------------------------------------------------------
    def click(self, region, name: str = 'Click', long_click: bool = False,
              duration=None) -> None:
        """区域内点击/长按。

        Args:
            region: (x, y, w, h) 点击区域。
            name: 控件名（用于日志/防卡死）。
            long_click: 是否长按。
            duration: 长按时长（秒）；None 用设备默认。
        """
        x, y = self.policy.aim(region)
        if long_click:
            self.device.long_click(x, y, duration=duration, control_name=name)
        else:
            self.device.click(x, y, control_name=name)
        self._record('click', region, (x, y), name)

    def long_click(self, region, duration=None, name: str = 'LongClick') -> None:
        """区域内长按。"""
        self.click(region, name=name, long_click=True, duration=duration)

    # ------------------------------------------------------------------
    # 滑动
    # ------------------------------------------------------------------
    def swipe(self, region_a, region_b, name: str = 'SWIPE') -> None:
        """从区域 A 滑到区域 B。

        轨迹由策略 path() 生成；多段轨迹时仅第一段计入防卡死，
        避免一次逻辑滑动被拆成多次 device 调用导致误判。
        """
        path = self.policy.path(region_a, region_b)
        if len(path) <= 2:
            p1, p2 = path[0], path[-1]
            self.device.swipe(p1=p1, p2=p2, control_name=name)
        else:
            for idx in range(len(path) - 1):
                self.device.swipe(
                    p1=path[idx], p2=path[idx + 1], control_name=name,
                    control_check=(idx == 0),
                )
        self._record('swipe', (region_a, region_b), path, name)

    # ------------------------------------------------------------------
    # 记录
    # ------------------------------------------------------------------
    def _record(self, kind: str, region, decision, name: str) -> None:
        if self.recorder is not None:
            self.recorder.record(kind=kind, region=region,
                                 decision=decision, name=name)

    # ------------------------------------------------------------------
    # 策略切换
    # ------------------------------------------------------------------
    def set_policy(self, policy: Policy) -> None:
        """运行时切换策略（配置驱动 / 任务覆写）。"""
        self.policy = policy
