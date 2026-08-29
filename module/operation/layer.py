"""操作中间层：区域指令 -> 策略 -> 物理执行。

任务代码只描述"在哪块区域做什么"（click(region) / swipe(region_a, region_b)），
不直接操纵坐标。具体落点、轨迹、物理参数由策略（Policy）决定，
并在此统一记录操作日志供 AI 学习。
"""
from __future__ import annotations

from time import perf_counter, sleep

from . import region as R
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
        # 最终保险：任何策略产出的落点都不越出屏幕
        x, y = self._clamp(x, y)
        phys = self.policy.physical()
        start = perf_counter()
        if phys:
            # 注入拟人化物理参数（压力/停留/微动）
            if long_click:
                self.device.long_click(x, y, duration=duration, control_name=name,
                                       pressure=phys.get('pressure'))
            else:
                self.device.click(x, y, control_name=name,
                                  pressure=phys.get('pressure'),
                                  dwell=phys.get('dwell_ms'),
                                  micro_move=phys.get('micro_move'))
        else:
            if long_click:
                self.device.long_click(x, y, duration=duration, control_name=name)
            else:
                self.device.click(x, y, control_name=name)
        dt = perf_counter() - start
        self._record('click', region, (x, y), name)
        # 策略节奏：推进疲劳 + 操作后停顿（拟人化决策延时/按压停留）
        self.policy.record(dt=dt)
        self._pause(self.policy.after_click())

    def long_click(self, region, duration=None, name: str = 'LongClick') -> None:
        """区域内长按。"""
        self.click(region, name=name, long_click=True, duration=duration)

    # ------------------------------------------------------------------
    # 滑动
    # ------------------------------------------------------------------
    def swipe(self, region_a, region_b, name: str = 'SWIPE',
              duration=None, method: str | None = None) -> None:
        """从区域 A 滑到区域 B。

        轨迹由策略 path() 生成；多段轨迹时仅第一段计入防卡死，
        避免一次逻辑滑动被拆成多次 device 调用导致误判。

        Args:
            duration: 指定滑动时长（秒）；仅 method='adb' 时使用。
            method: 'adb' 强制 adb 慢速精确滑动（忽略策略轨迹）；
                None 用策略轨迹 + device.swipe。
        """
        if method == 'adb':
            # adb 慢速精确滑动（拖动列表/摇杆等场景），起终点取区域中心
            p1 = self._clamp(*R.region_center(region_a))
            p2 = self._clamp(*R.region_center(region_b))
            start = perf_counter()
            self.device.swipe_adb(p1=p1, p2=p2, duration=duration or (0.1, 0.2))
            dt = perf_counter() - start
            self._record('swipe', (region_a, region_b), [p1, p2], name)
            self.policy.record(dt=dt)
            self._pause(self.policy.after_swipe())
            return
        self._apply_screen()
        path = self.policy.path(region_a, region_b)
        # 最终保险：轨迹所有点都不越出屏幕
        path = [self._clamp(px, py) for px, py in path]
        start = perf_counter()
        if len(path) <= 2:
            p1, p2 = path[0], path[-1]
            phys = self.policy.physical()
            if phys:
                self.device.swipe(p1=p1, p2=p2, control_name=name,
                                  pressure=phys.get('pressure'),
                                  move_delay=phys.get('move_delay_ms'))
            else:
                self.device.swipe(p1=p1, p2=p2, control_name=name)
        else:
            delay = max(0.0, self.policy.move_delay())
            phys = self.policy.physical()
            for idx in range(len(path) - 1):
                if phys:
                    self.device.swipe(
                        p1=path[idx], p2=path[idx + 1], control_name=name,
                        control_check=(idx == 0),
                        pressure=phys.get('pressure'),
                        move_delay=phys.get('move_delay_ms'),
                    )
                else:
                    self.device.swipe(
                        p1=path[idx], p2=path[idx + 1], control_name=name,
                        control_check=(idx == 0),
                    )
                # 拟人化段间延迟（模拟连续滑动的移动时间）
                if idx < len(path) - 2 and delay > 0:
                    sleep(delay)
        dt = perf_counter() - start
        self._record('swipe', (region_a, region_b), path, name)
        # 策略节奏：推进疲劳 + 滑动后决策延时
        self.policy.record(dt=dt)
        self._pause(self.policy.after_swipe())

    # ------------------------------------------------------------------
    # 记录 / 节奏
    # ------------------------------------------------------------------
    def _record(self, kind: str, region, decision, name: str) -> None:
        if self.recorder is not None:
            self.recorder.record(kind=kind, region=region,
                                 decision=decision, name=name)

    @staticmethod
    def _pause(seconds: float) -> None:
        """按策略要求的停顿等待（秒）；<=0 不等待。"""
        if seconds and seconds > 0:
            sleep(seconds)

    # ------------------------------------------------------------------
    # 屏幕边界（不越界保障）
    # ------------------------------------------------------------------
    def _screen(self) -> tuple[int, int] | None:
        """从设备当前截图取屏幕尺寸 (宽, 高)；无则 None。"""
        try:
            img = getattr(self.device, 'image', None)
            if img is not None:
                h, w = img.shape[:2]
                return (int(w), int(h))
        except Exception:
            pass
        return None

    def _apply_screen(self) -> None:
        """把当前屏幕尺寸同步给策略（拟人化据此钳制坐标不越界）。"""
        setter = getattr(self.policy, 'set_screen', None)
        if setter is not None:
            screen = self._screen()
            if screen is not None:
                setter(*screen)

    def _clamp(self, px, py) -> tuple[int, int]:
        """把点钳制到屏幕边界内（最终保险）。"""
        screen = self._screen()
        if screen is None:
            return int(px), int(py)
        w, h = screen
        return (int(min(max(px, 0), w - 1)), int(min(max(py, 0), h - 1)))

    # ------------------------------------------------------------------
    # 策略切换
    # ------------------------------------------------------------------
    def set_policy(self, policy: Policy) -> None:
        """运行时切换策略（配置驱动 / 任务覆写）。"""
        self.policy = policy
