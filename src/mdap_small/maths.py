import math


def k_min(target_success: float, steps: int, p_step: float, m: int = 1) -> int:
    if not (0.5 < p_step < 1.0):
        raise ValueError("p_step must be in (0.5, 1.0)")
    if not (0.0 < target_success < 1.0):
        raise ValueError("target_success must be in (0, 1)")
    ratio = (1.0 - p_step) / p_step
    num = math.log((target_success ** (-m / steps)) - 1.0)
    den = math.log(ratio)
    return max(1, math.ceil(num / den))
