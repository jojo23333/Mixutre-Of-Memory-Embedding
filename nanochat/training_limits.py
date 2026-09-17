"""
Helpers for training-loop stop conditions.
"""

import time


def elapsed_wall_time_seconds(start_time: float, now: float | None = None) -> float:
    if now is None:
        now = time.time()
    return max(0.0, now - start_time)


def should_stop_for_wall_time(
    start_time: float,
    max_wall_time_seconds: float,
    now: float | None = None,
) -> bool:
    if max_wall_time_seconds <= 0:
        return False
    return elapsed_wall_time_seconds(start_time, now=now) >= max_wall_time_seconds
