"""操作中间层（Operation Layer）。

把"坐标操作"抽象为"区域指令"，由可插拔策略统一承载
默认 / 拟人化 / AI 三种行为，并为 AI 学习提供数据闭环。
"""
from .region import (
    Region, Point,
    region_center, region_random, region_normal,
    region_contains, clip_to_region, point_region, point_region_around, rule_region,
)
from .policy import Policy, DefaultPolicy
from .human import HumanPolicy
from .ai import AIPolicy
from .recorder import Recorder
from .layer import OperationLayer

__all__ = [
    'Region', 'Point',
    'region_center', 'region_random', 'region_normal',
    'region_contains', 'clip_to_region', 'point_region', 'point_region_around', 'rule_region',
    'Policy', 'DefaultPolicy', 'HumanPolicy', 'AIPolicy',
    'Recorder', 'OperationLayer',
]
