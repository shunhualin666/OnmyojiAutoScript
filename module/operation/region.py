"""区域（ROI）工具函数。

统一使用 (x, y, w, h) 表示一块区域，兼容资产系统的 roi_front 格式。
"""
from __future__ import annotations

import numpy as np

# 区域 (x, y, w, h) / 点 (x, y)
Region = tuple[int, int, int, int]
Point = tuple[int, int]


def region_center(region) -> Point:
    """区域中心，钳制到区域内（保证退化/越界区域也返回合法点）。"""
    x, y, w, h = region
    cx, cy = x + w / 2.0, y + h / 2.0
    # ROI 语义 [x, x+w-1]：上限为 x+w-1；w/h<=0（退化区域）返回区域原点
    hi_x, hi_y = x + max(0, w - 1), y + max(0, h - 1)
    return int(min(max(cx, x), hi_x)), int(min(max(cy, y), hi_y))


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
    hi_x, hi_y = x + max(0, w - 1), y + max(0, h - 1)
    px = int(round(float(np.clip(rng.normal(cx, sx), x, hi_x))))
    py = int(round(float(np.clip(rng.normal(cy, sy), y, hi_y))))
    return px, py


def region_contains(region, px, py) -> bool:
    """点是否在区域内（ROI 语义 [x, x+w-1]）。"""
    x, y, w, h = region
    hi_x, hi_y = x + max(0, w - 1), y + max(0, h - 1)
    return x <= px <= hi_x and y <= py <= hi_y


def clip_to_region(region, px, py) -> Point:
    """把点截断到区域内（ROI 语义 [x, x+w-1]）。"""
    x, y, w, h = region
    hi_x, hi_y = x + max(0, w - 1), y + max(0, h - 1)
    return int(min(max(px, x), hi_x)), int(min(max(py, y), hi_y))


def point_region(x, y, w: int = 8, h: int = 8) -> Region:
    """以中心点 (x, y) 和宽高 (w, h) 构造区域（中心点语义）。

    返回 (x - w//2, y - h//2, w, h)，区域中心精确落在 (x, y)。
    """
    return int(x - w // 2), int(y - h // 2), int(w), int(h)


def rule_region(rule) -> Region:
    """从规则对象取点击区域。

    优先级：area（OCR 检测后的动态文字区域）> roi_front > roi。

    注意：RuleOcr 的 self.area 在 ocr() 检测后会更新为实际文字框，
    点击必须落在其上（而非静态搜索区域 roi），否则会点偏导致跳转失败。
    """
    # OCR 动态区域：ocr() 后 self.area 更新为实际文字框；仅 RuleOcr 有 area 属性
    area = getattr(rule, 'area', None)
    if area and len(area) >= 4:
        return int(area[0]), int(area[1]), int(area[2]), int(area[3])
    roi = getattr(rule, 'roi_front', None)
    if roi and len(roi) >= 4:
        return int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    roi = getattr(rule, 'roi', None)
    if roi and len(roi) >= 4:
        return int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    raise ValueError(f'规则 {rule} 没有可用的 roi 区域')
