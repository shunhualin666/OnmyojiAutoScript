# This Python file uses the following encoding: utf-8
# @author runhey
# github https://github.com/runhey
import numpy as np
import random

from math import dist

from module.base.decorator import cached_property
from module.atom.cBezier import BezierTrajectory
from module.logger import logger


class RuleSwipe:

    def __init__(self, roi_front: tuple, roi_back: tuple, mode: str, name: str =None) -> None:
        """
        初始化
        :param roi_front:
        :param roi_back:
        :param mode:
        """
        self.roi_front = roi_front
        self.roi_back = roi_back
        self.mode = mode
        if name:
            self.name = name
        else:
            self.name = 'swipe'

        self.interval: int = 8  # 每次移动的间隔时间

    @cached_property
    def is_default_mode(self) -> bool:
        """
        是否是默认模式
        :return:
        """
        return self.mode == 'default'

    @cached_property
    def is_vector_mode(self) -> bool:
        """
        是否是向量模式
        :return:
        """
        return self.mode == 'vector'

    def coord(self) -> tuple:
        """
        获取坐标, 从roi_front随机获取坐标 和从roi_back随机获取的坐标
        :return: 两个坐标的tuple
        """
        x, y, w, h = self.roi_front
        x = np.random.randint(x, x + w)
        y = np.random.randint(y, y + h)
        x2, y2, w2, h2 = self.roi_back
        x2 = np.random.randint(x2, x2 + w2)
        y2 = np.random.randint(y2, y2 + h2)
        return x, y, x2, y2