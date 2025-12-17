"""
Pose/grip filtering and gating for ee_targets v1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .math_utils import quat_normalize, quat_slerp


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


@dataclass
class ArmPoseCommand:
    id: str
    ee_frame: str
    p: Vec3
    q: Quat
    grip: float
    mode: str
    precision: bool
    clutch: bool
    reset_edge: bool
    frame: str


class PoseCommandFilter:
    def __init__(
        self,
        timeout_s: float = 0.5,
        alpha_p: float = 0.2,
        alpha_q: float = 0.2,
        alpha_grip: float = 0.3,
        grip_deadband: float = 0.02,
        current_pose_fn: Optional[Callable[[str], Tuple[Vec3, Quat]]] = None,
    ):
        self.timeout_s = timeout_s
        self.alpha_p = alpha_p
        self.alpha_q = alpha_q
        self.alpha_grip = alpha_grip
        self.grip_deadband = grip_deadband
        self.current_pose_fn = current_pose_fn
        self._last: Dict[str, ArmPoseCommand] = {}
        self._last_time = time.time()

    def _snap_to_current(self, arm_id: str) -> Optional[Tuple[Vec3, Quat]]:
        if self.current_pose_fn is None:
            return None
        try:
            return self.current_pose_fn(arm_id)
        except Exception:
            return None

    def update(self, state, reset_edge: bool) -> List[ArmPoseCommand]:
        now = time.time()
        self._last_time = now

        if state is None:
            return list(self._last.values())

        clutch_on = bool(state.clutch)
        precision_on = bool(state.precision)
        frame = state.frame

        out: List[ArmPoseCommand] = []
        for arm in state.arms:
            prev = self._last.get(arm.id)
            if reset_edge:
                snap = self._snap_to_current(arm.id)
                if snap:
                    p_snap, q_snap = snap
                    prev = ArmPoseCommand(
                        id=arm.id,
                        ee_frame=arm.ee_frame,
                        p=p_snap,
                        q=quat_normalize(q_snap),
                        grip=arm.grip,
                        mode=arm.mode,
                        precision=precision_on,
                        clutch=clutch_on,
                        reset_edge=True,
                        frame=frame,
                    )

            if prev is None:
                prev = ArmPoseCommand(
                    id=arm.id,
                    ee_frame=arm.ee_frame,
                    p=arm.p,
                    q=quat_normalize(arm.q),
                    grip=arm.grip,
                    mode=arm.mode,
                    precision=precision_on,
                    clutch=clutch_on,
                    reset_edge=reset_edge,
                    frame=frame,
                )

            p_filtered = tuple(prev.p[i] * (1 - self.alpha_p) + arm.p[i] * self.alpha_p for i in range(3))
            q_filtered = quat_slerp(prev.q, quat_normalize(arm.q), self.alpha_q)

            grip_target = arm.grip
            if abs(grip_target - prev.grip) < self.grip_deadband:
                grip_target = prev.grip
            else:
                grip_target = prev.grip * (1 - self.alpha_grip) + grip_target * self.alpha_grip

            cmd = ArmPoseCommand(
                id=arm.id,
                ee_frame=arm.ee_frame,
                p=p_filtered,
                q=q_filtered,
                grip=grip_target,
                mode=arm.mode,
                precision=precision_on,
                clutch=clutch_on,
                reset_edge=reset_edge,
                frame=frame,
            )
            if clutch_on:
                self._last[arm.id] = cmd
            out.append(cmd if clutch_on else prev)

        return out
