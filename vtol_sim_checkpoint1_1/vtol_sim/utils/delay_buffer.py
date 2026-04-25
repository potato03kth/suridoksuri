"""
시간 지연 버퍼
==============

특정 시간만큼 신호를 지연시키기 위한 FIFO 버퍼.

사용 예시:
    buf = DelayBuffer(delay=0.05, dt=0.01, init_value=zero_state)
    delayed = buf.update(current_value, t)
"""
from __future__ import annotations
from collections import deque
from typing import Any


class DelayBuffer:
    """
    delay 시간만큼 신호를 지연.

    구현: deque에 최근 N개를 저장. N = ceil(delay / dt).
    매 update 호출마다 새 값을 push하고 가장 오래된 값을 pop.
    """

    def __init__(self, delay: float, dt: float, init_value: Any = None):
        if delay < 0:
            raise ValueError("delay must be non-negative")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.delay = float(delay)
        self.dt = float(dt)
        # delay=0이면 버퍼 크기 1 (즉시 통과)
        self.size = max(1, int(round(delay / dt)))
        self._buf: deque = deque(maxlen=self.size)
        # 초기 값으로 채움 (cold start 시 None 반환 방지)
        for _ in range(self.size):
            self._buf.append(init_value)

    def update(self, value: Any) -> Any:
        """새 값 푸시. 지연된 값 반환."""
        delayed = self._buf[0]  # 가장 오래된 값
        self._buf.append(value)  # maxlen이므로 가장 오래된 값 자동 제거
        return delayed

    def reset(self, init_value: Any = None) -> None:
        self._buf.clear()
        for _ in range(self.size):
            self._buf.append(init_value)

    def __repr__(self) -> str:
        return f"DelayBuffer(delay={self.delay}, dt={self.dt}, size={self.size})"


class VariableDelayBuffer:
    """
    가변 지연 버퍼 — 매 push마다 timestamp를 함께 저장하여,
    pop 시 요청한 시각에 해당하는 값을 반환.

    실연산시간 측정 모드에서 사용 (지연이 매번 다름).
    """

    def __init__(self, max_history: int = 10000, init_value: Any = None):
        self._history: list[tuple[float, Any]] = []
        self._max_history = max_history
        self._init_value = init_value

    def push(self, t: float, value: Any) -> None:
        self._history.append((t, value))
        if len(self._history) > self._max_history:
            # 오래된 것 잘라내기
            self._history = self._history[-self._max_history // 2:]

    def get_at(self, t_query: float) -> Any:
        """t_query 시점에 해당하는 값을 반환 (그 시점 이전의 가장 최근 값)."""
        if not self._history:
            return self._init_value
        # 단순 선형 검색 (역방향)
        for t_pushed, value in reversed(self._history):
            if t_pushed <= t_query:
                return value
        return self._init_value

    def reset(self) -> None:
        self._history.clear()
