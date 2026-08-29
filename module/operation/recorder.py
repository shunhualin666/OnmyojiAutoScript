"""操作记录器：为 AI 学习采集 (状态-动作-结果) 数据。

每条记录是 jsonl 的一行，包含时间、动作类型、区域、策略决策结果等。
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class Recorder:
    """把操作记录追加写入 log/operation/operation.jsonl。"""

    def __init__(self, path=None, enabled: bool = True):
        self.enabled = enabled
        self.path = Path(path) if path else Path.cwd() / 'log' / 'operation'
        self.path.mkdir(parents=True, exist_ok=True)

    def record(self, kind: str, region, decision, name: str,
               state=None, result=None) -> None:
        """记录一次操作。

        Args:
            kind: 'click' / 'swipe' / 'long_click'。
            region: 区域（或区域对）。
            decision: 策略产生的具体动作（落点/轨迹）。
            name: 控件名。
            state: 操作前状态（可选，供 AI 观测）。
            result: 操作后结果（可选）。
        """
        if not self.enabled:
            return
        entry = {
            'ts': round(time.time(), 3),
            'kind': kind,
            'region': region,
            'decision': decision,
            'name': name,
            'state': state,
            'result': result,
        }
        with open(self.path / 'operation.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
