import math
from typing import Iterable, Tuple


def _to_tuple(q: Iterable[float]) -> Tuple[float, float, float, float]:
    vals = list(q)
    if len(vals) != 4:
        raise ValueError("Quaternion must have 4 elements [w, x, y, z]")
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


def quat_normalize(q: Iterable[float]) -> Tuple[float, float, float, float]:
    w, x, y, z = _to_tuple(q)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-9:
        return 1.0, 0.0, 0.0, 0.0
    return w / norm, x / norm, y / norm, z / norm


def quat_multiply(q1: Iterable[float], q2: Iterable[float]) -> Tuple[float, float, float, float]:
    a1, b1, c1, d1 = _to_tuple(q1)
    a2, b2, c2, d2 = _to_tuple(q2)
    return (
        a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
        a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
        a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
        a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
    )


def quat_slerp(q1: Iterable[float], q2: Iterable[float], alpha: float) -> Tuple[float, float, float, float]:
    """Slerp with clamping and fall back to lerp for small angles."""
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    dot = sum(a * b for a, b in zip(q1, q2))
    if dot < 0.0:
        q2 = tuple(-x for x in q2)
        dot = -dot
    dot = max(min(dot, 1.0), -1.0)
    if dot > 0.9995:
        lerped = tuple(a + alpha * (b - a) for a, b in zip(q1, q2))
        return quat_normalize(lerped)
    theta_0 = math.acos(dot)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (
        s0 * q1[0] + s1 * q2[0],
        s0 * q1[1] + s1 * q2[1],
        s0 * q1[2] + s1 * q2[2],
        s0 * q1[3] + s1 * q2[3],
    )
