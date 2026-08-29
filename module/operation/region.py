"""区域（ROI）工具函数。

统一使用 (x, y, w, h) 表示一块区域，兼容资产系统的 roi_front 格式。
"""
from __future__ import annotations

import numpy as np

# 区域 (x, y, w, h) / 点 (x, y)
Region = tuple[int, int, int, int]
Point = tuple[int, int]


def region_center(region) -> Point:
    """区域中心。"""
    x, y, w, h = region
    return int(x + w / 2), int(y + h / 2)


def region_random(region, rng=None) -> Point:
    """区域内均匀随机落点（与 RuleClick.coord 行为一致）。"""
    x, y, w, h = region
    rng = rng if rng is not None else np.random
    return int(rng.randint(x, x + w)), int(rng.randint(y, y + h))


def region_normal(region, sigma_ratio: float = 1 / 6, rng=None) -> Point:
    """区域内中心截断高斯落点（拟人化用，中心多、边缘稀疏）。"""
    x, y, w, h = region
    rng = rng if rng is not None else np.random
    cx, cy = x + w / 2.0, y + h / 2.0
    sx, sy = max(1.0, w * sigma_ratio), max(1.0, h * sigma_ratio)
    px = int(round(float(np.clip(rng.normal(cx, sx), x, x + w))))
    py = int(round(float(np.clip(rng.normal(cy, sy), y, y + h))))
    return px, py


def region_contains(region, px, py) -> bool:
    """点是否在区域内。"""
    x, y, w, h = region
    return x <= px <= x + w and y <= py <= y + h


def clip_to_region(region, px, py) -> Point:
    """把点截断到区域内。"""
    x, y, w, h = region
    return int(min(max(px, x), x + w)), int(min(max(py, y), y + h))


def point_region(x, y, radius: int = 3) -> Region:
    """把单个坐标扩成一个小区域（供动态/计算坐标调用，避免直接传点）。"""
    return (x - radius, y - radius, 2 * radius, 2 * radius)


def rule_region(rule) -> Region:
    """从规则对象取点击区域（优先 roi_front，其次 roi）。"""
    roi = getattr(rule, 'roi_front', None)
    if roi and len(roi) >= 4:
        return int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    roi = getattr(rule, 'roi', None)
    if roi and len(roi) >= 4:
        return int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    raise ValueError(f'规则 {rule} 没有可用的 roi 区域')
