"""拟人化策略（HumanPolicy）：完整拟人化操作模型。

基于拟人化输入模块的模型经验（见《拟人化模块实现与经验总结.md》），
通过操作中间层的可插拔策略接口接入：
- 落点：两成分运动（初始弹道 + 慢速修正）+ 1/f 低频漂移 + 手抖，**截断在区域内**
- 轨迹：最小 jerk 五次多项式 + 垂直弯曲 + 抖动，**首尾固定**，**过滤 <10px 短段**
- 节奏：Hick 决策延时、对数正态间隔、AR(1) 负自相关、两态突发性
- 疲劳：随会话增长落点更散/间隔更慢/轨迹更抖，休息回弹
- 个体签名：seed 派生（同账号固定、跨账号独立）
- 物理：按压压力/停留/微动/移动延迟（供设备层或记录）

历史踩坑已规避：
- 分段滑动防卡死：OperationLayer.swipe 仅第一段计防卡死
- 短段 <10px 被 distance_check 丢弃：path 内过滤短段
- 落点越界：aim 以区域为界截断
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import region as R
from .policy import Policy


class HumanPolicy(Policy):
    """拟人化操作策略。

    Args:
        seed: 个体签名种子；同一账号固定、不同账号独立。
        n_segments: 最小 jerk 轨迹分段数。
        enabled: 总开关；关闭时退化为默认行为。
    """

    name = 'human'

    def __init__(self, seed: int | None = None, n_segments: int = 6,
                 enabled: bool = True,
                 screen: tuple[int, int] | None = None) -> None:
        self.enabled = bool(enabled)
        self.rng = np.random.default_rng(seed)
        self.n_segments = int(n_segments)
        # 屏幕边界 (宽, 高)；非空时所有拟人化坐标被钳制在屏幕内（不越界）
        self.screen = screen

        # ---- 落点散布（点击不总在中心）----
        self.sigma_jitter = self.rng.uniform(0.5, 2.5)      # 手抖 px（Harris & Wolpert 1998）
        self.w_e = self.rng.uniform(3.0, 8.0)               # 有效宽度 px（Fitts 1964）
        self.alpha_1f = self.rng.uniform(0.0, 1.0)          # 1/f 低频漂移强度
        self._drift_x = 0.0
        self._drift_y = 0.0

        # ---- 滑动轨迹 ----
        self.curve_bias = self.rng.uniform(0.05, 0.30)      # 垂直弯曲比例×长度
        self.track_jitter = self.rng.uniform(0.5, 2.0)      # 轨迹抖动 px

        # ---- 操作间隔（对数正态，中位 300~450ms）----
        self.lat_mu = self.rng.uniform(float(np.log(300)), float(np.log(450)))
        self.lat_sigma = self.rng.uniform(0.20, 0.40)

        # ---- 菲茨定律移动时间 ----
        self.fitts_a = self.rng.uniform(0.05, 0.20)         # s
        self.fitts_b = self.rng.uniform(0.10, 0.20)         # s/bit

        # ---- 按下时长（截断对数正态）----
        self.hold_mu = self.rng.uniform(float(np.log(55)), float(np.log(110)))
        self.hold_sigma = self.rng.uniform(0.15, 0.35)

        # ---- Hick 选择反应时间 ----
        self.rt_c = self.rng.uniform(0.08, 0.15)            # 基线 s
        self.rt_d = self.rng.uniform(0.03, 0.07)            # 每 bit 增量 s

        # ---- 节奏：AR(1) 负自相关 ----
        self.ar_alpha = -self.rng.uniform(0.1, 0.5)
        self.ar_noise = self.rng.uniform(0.15, 0.35)
        self._last_interval: float | None = None

        # ---- 突发性：两态马尔可夫 ----
        self.burst_p_aa = self.rng.uniform(0.70, 0.95)
        self.burst_p_ii = self.rng.uniform(0.80, 0.97)
        self.burst_active_dt = self.rng.uniform(0.10, 0.25)
        self.burst_idle_dt = self.rng.uniform(1.0, 3.0)
        self._burst_state = 'active'

        # ---- 疲劳 ----
        self.fatigue_k = self.rng.uniform(0.10, 0.30)
        self.fatigue_Tf = self.rng.uniform(1800.0, 3600.0)
        self.t_session = 0.0
        self.t_rest = 0.0

        # ---- 触摸物理 ----
        self.press_ratio = self.rng.uniform(0.55, 0.95)
        self.dwell_peak = self.rng.uniform(60.0, 110.0)
        self.dwell_lo = self.rng.uniform(45.0, self.dwell_peak)   # lo <= peak
        self.dwell_hi = self.rng.uniform(self.dwell_peak, 180.0)  # peak <= hi
        self.micro_move = self.rng.uniform(1.0, 3.0)
        self.move_delay_lo = self.rng.uniform(6.0, 12.0)
        self.move_delay_hi = self.rng.uniform(14.0, 24.0)

    # ------------------------------------------------------------------
    # 个体签名
    # ------------------------------------------------------------------
    @staticmethod
    def seed_from_name(name: str) -> int:
        """由账号/配置名派生稳定种子（同一账号签名固定）。"""
        if not name:
            return 0
        digest = hashlib.sha256(str(name).encode('utf-8')).hexdigest()
        return int(digest[:8], 16)

    def set_screen(self, width: int, height: int) -> None:
        """设置屏幕边界；此后所有拟人化坐标都不越界。"""
        self.screen = (int(width), int(height))

    def _clip_screen(self, px: float, py: float) -> tuple[int, int]:
        """把点钳制到屏幕边界内（不越界）。"""
        if self.screen is None:
            return int(round(px)), int(round(py))
        w, h = self.screen
        return (int(min(max(px, 0), w - 1)), int(min(max(py, 0), h - 1)))

    # ------------------------------------------------------------------
    # 疲劳 / 漂移
    # ------------------------------------------------------------------
    def fatigue_factor(self) -> float:
        """疲劳速度因子：<1 越疲劳越慢。"""
        eff = self.t_session + 0.5 * self.t_rest
        fatigue = 1.0 - self.fatigue_k * (1.0 - np.exp(-eff / self.fatigue_Tf))
        return float(max(0.5, min(1.0, fatigue)))

    def spread_factor(self) -> float:
        """落点散布因子：>=1，疲劳时更大。"""
        return float(1.0 / self.fatigue_factor())

    def drift(self) -> tuple[float, float]:
        """低频 1/f 手部漂移（AR(1) 平滑随机游走）。"""
        s = self.sigma_jitter * self.alpha_1f * 0.4
        self._drift_x = 0.92 * self._drift_x + self.rng.normal(0.0, s)
        self._drift_y = 0.92 * self._drift_y + self.rng.normal(0.0, s)
        return self._drift_x, self._drift_y

    # ------------------------------------------------------------------
    # 落点（两成分运动 + 漂移 + 区域截断）
    # ------------------------------------------------------------------
    def aim(self, region) -> tuple[int, int]:
        """区域内拟人落点：两成分对准 + 手抖 + 低频漂移，截断在区域内。"""
        cx, cy = R.region_center(region)
        x, y, w, h = region
        # 落点散布与目标尺寸相关（菲茨定律：目标越大越随意），
        # σ ≈ 宽度/6（中心多、边缘少的正态散布）；w_e/4.133 作为最小散布下限
        sigma_x = max(self.w_e / 4.133, w / 6.0)
        sigma_y = max(self.w_e / 4.133, h / 6.0)
        s = max(sigma_x, sigma_y) * self.spread_factor()
        px = cx + self.rng.normal(0.0, s)
        py = cy + self.rng.normal(0.0, s)
        # 修正亚运动：25% 概率产生修正对准，更贴近中心但不过分集中
        if self.rng.random() < 0.25:
            px = cx + self.rng.normal(0.0, s * 0.5)
            py = cy + self.rng.normal(0.0, s * 0.5)
        # 手抖
        px += self.rng.normal(0.0, self.sigma_jitter * self.spread_factor())
        py += self.rng.normal(0.0, self.sigma_jitter * self.spread_factor())
        # 低频漂移
        dx, dy = self.drift()
        # 落点一定在区域内，再钳制到屏幕内（不越界）
        px, py = R.clip_to_region(region, px + dx, py + dy)
        return self._clip_screen(px, py)

    # ------------------------------------------------------------------
    # 轨迹（最小 jerk + 弯曲 + 抖动，首尾固定，过滤短段）
    # ------------------------------------------------------------------
    def path(self, region_a, region_b) -> list[tuple[int, int]]:
        """区域 A -> 区域 B 的拟人轨迹点。"""
        # 首尾 = 区域内合法点，并保证在屏幕内（不越界）
        a = self._clip_screen(*R.region_center(region_a))
        b = self._clip_screen(*R.region_center(region_b))
        t = np.linspace(0.0, 1.0, self.n_segments + 1)
        # 最小 jerk 五次多项式
        s = 10.0 * t ** 3 - 15.0 * t ** 4 + 6.0 * t ** 5
        path = [(int(round(a[0] + (b[0] - a[0]) * si)),
                 int(round(a[1] + (b[1] - a[1]) * si))) for si in s]
        # 垂直方向随机弯曲（中段最大）
        delta = np.array([b[0] - a[0], b[1] - a[1]], dtype=float)
        normal = np.array([-delta[1], delta[0]], dtype=float)
        nl = float(np.linalg.norm(normal)) + 1e-9
        normal = normal / nl
        dist = float(np.linalg.norm(delta))
        envelope = np.sin(np.pi * t)[:, None]
        bend = self.rng.normal(0.0, self.curve_bias * dist) * normal[None, :] * envelope
        # 轨迹抖动（疲劳时更抖）
        jitter = self.track_jitter * self.spread_factor()
        path = [(int(round(px + bx + jx)), int(round(py + by + jy)))
                for (px, py), (bx, by), (jx, jy) in
                zip(path, bend, self.rng.normal(0.0, jitter, (len(path), 2)))]
        # 中间点全部钳制到屏幕内（不越界），首尾严格固定
        path = [self._clip_screen(px, py) for px, py in path]
        path[0] = a
        path[-1] = b
        # 过滤 <10px 短段（避免被 device distance_check 当作点击丢弃），末段强制到终点
        points = [path[0]]
        for pt in path[1:-1]:
            if np.hypot(pt[0] - points[-1][0], pt[1] - points[-1][1]) >= 10:
                points.append(pt)
        if np.hypot(path[-1][0] - points[-1][0], path[-1][1] - points[-1][1]) < 10:
            if len(points) > 1:
                points[-1] = path[-1]
            else:
                points.append(path[-1])
        else:
            points.append(path[-1])
        return points

    # ------------------------------------------------------------------
    # 菲茨移动时长 / Hick / 按下时长
    # ------------------------------------------------------------------
    def move_time(self, distance: float, width: float | None = None) -> float:
        """菲茨定律移动时间（秒）。"""
        width = width if width else self.w_e
        id_ = np.log2(distance / width + 1.0)
        mt = self.fitts_a + self.fitts_b * id_
        mt *= self.rng.normal(1.0, 0.08) / self.fatigue_factor()
        return float(max(0.03, mt))

    def hick_rt(self, n_options: int = 1) -> float:
        """Hick 选择反应时间（秒），用作操作后决策延时。"""
        rt = self.rt_c + self.rt_d * np.log2(n_options + 1)
        rt *= self.rng.normal(1.0, 0.10)
        return float(max(0.10, rt))

    def hold_time(self, kind: str = 'tap') -> float:
        """按下时长（秒），截断对数正态。"""
        mu = self.hold_mu if kind == 'tap' else self.hold_mu + 0.6
        h = float(np.exp(self.rng.normal(mu, self.hold_sigma))) / 1000.0
        lo, hi = (0.01, 0.35) if kind == 'tap' else (0.4, 2.0)
        return float(np.clip(h, lo, hi))

    # ------------------------------------------------------------------
    # 操作节奏钩子（OperationLayer 调用）
    # ------------------------------------------------------------------
    def after_click(self) -> float:
        """点击后停顿（秒）：决策延时（按压停留已由设备注入承担，不重复）。"""
        if not self.enabled:
            return 0.0
        return self.hick_rt()

    def after_swipe(self) -> float:
        """滑动后停顿（秒）：决策延时。"""
        if not self.enabled:
            return 0.0
        return self.hick_rt()

    def move_delay(self) -> float:
        """分段滑动段间延迟（秒）。"""
        if not self.enabled:
            return 0.0
        return self.move_delay_ms() / 1000.0

    def interval(self, base: float = 0.0) -> float:
        """拟人化操作间隔（秒），对数正态 + AR(1) 负自相关 + 疲劳变慢。"""
        if not self.enabled:
            return float(base)
        jitter = float(np.exp(self.rng.normal(self.lat_mu, self.lat_sigma))) / 1000.0
        jitter = float(np.clip(jitter, 0.03, 0.5))
        if self._last_interval is not None:
            mean = float(np.exp(self.lat_mu + 0.5 * self.lat_sigma ** 2)) / 1000.0
            jitter = mean + self.ar_alpha * (self._last_interval - mean) + jitter * self.ar_noise
        jitter = float(np.clip(jitter, 0.03, 0.5))
        self._last_interval = jitter
        if base <= 0:
            return jitter / self.fatigue_factor()
        return float(np.clip((base * 0.6 + jitter * 0.4) / self.fatigue_factor(), 0.03, 0.6))

    def burst_step(self) -> tuple[float, str]:
        """推进两态突发性，返回 (当前态基间隔, 状态名)。"""
        r = self.rng.random()
        if self._burst_state == 'active':
            dt = self.burst_active_dt
            if r > self.burst_p_aa:
                self._burst_state = 'idle'
        else:
            dt = self.burst_idle_dt
            if r > self.burst_p_ii:
                self._burst_state = 'active'
        return float(dt), self._burst_state

    # ------------------------------------------------------------------
    # 物理参数（供设备层注入 / 记录）
    # ------------------------------------------------------------------
    def pressure(self, top: int = 100) -> int:
        """拟人按压压力。"""
        top = int(top or 100)
        lo = max(20, int(top * self.press_ratio))
        return int(self.rng.integers(lo, top + 1))

    def dwell_ms(self) -> int:
        """按压停留时长（ms，triangular，lo<=peak<=hi）。"""
        return int(round(float(self.rng.triangular(self.dwell_lo, self.dwell_peak, self.dwell_hi))))

    def micro_move_xy(self) -> tuple[int, int]:
        """点击微动偏移（px）。"""
        return (int(round(self.rng.normal(0.0, self.micro_move))),
                int(round(self.rng.normal(0.0, self.micro_move))))

    def move_delay_ms(self) -> int:
        """滑动移动步间延迟（ms）。"""
        return int(round(float(self.rng.uniform(self.move_delay_lo, self.move_delay_hi))))

    def physical(self) -> dict | None:
        """本次操作的物理参数快照。

        策略启用时返回全字段有效值（供设备注入），保证设备层兜底不可达；
        禁用时返回 None，退化为默认行为（此时才由设备层兜底）。
        """
        if not self.enabled:
            return None
        return {
            'pressure': self.pressure(),
            'dwell_ms': self.dwell_ms(),
            'micro_move': self.micro_move_xy(),
            'move_delay_ms': self.move_delay_ms(),
        }

    # ------------------------------------------------------------------
    # 会话推进
    # ------------------------------------------------------------------
    def record(self, dt: float = 0.0) -> None:
        """记录一次操作耗时，推进疲劳。"""
        self.t_session += max(0.0, dt)

    def rest(self, seconds: float) -> None:
        """记录休息时长，触发疲劳回弹。"""
        self.t_rest += max(0.0, seconds)
