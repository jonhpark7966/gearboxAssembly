"""
UDP ee_targets v1 receiver and validation for gearboxAssembly agents.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


REQUIRED_HEADER_KEYS = {"v", "type", "seq", "t", "frame", "clutch", "precision", "reset", "arms"}
ARM_REQUIRED_KEYS = {"id", "ee_frame", "p", "q", "grip", "mode"}


@dataclass
class ArmTarget:
    id: str
    ee_frame: str
    p: Tuple[float, float, float]
    q: Tuple[float, float, float, float]
    grip: float
    mode: str


@dataclass
class EETargetsState:
    seq: int
    t: float
    frame: str
    clutch: int
    precision: int
    reset: int
    arms: List[ArmTarget]
    raw: Dict


def _validate_arm(arm: Dict) -> Optional[ArmTarget]:
    if not ARM_REQUIRED_KEYS.issubset(arm.keys()):
        return None
    try:
        pid = str(arm["id"])
        ee_frame = str(arm["ee_frame"])
        p = tuple(float(x) for x in arm["p"])  # type: ignore
        q = tuple(float(x) for x in arm["q"])  # type: ignore
        grip = float(arm["grip"])
        mode = str(arm["mode"])
        if len(p) != 3 or len(q) != 4:
            return None
        if pid not in ("L", "R"):
            return None
        if mode not in ("free", "insert", "rotate"):
            return None
        grip = max(0.0, min(1.0, grip))
        return ArmTarget(id=pid, ee_frame=ee_frame, p=p, q=q, grip=grip, mode=mode)
    except (TypeError, ValueError):
        return None


def parse_packet(data: bytes) -> Optional[EETargetsState]:
    try:
        msg = json.loads(data.decode("utf-8"))
    except Exception:
        return None
    if not REQUIRED_HEADER_KEYS.issubset(msg.keys()):
        return None
    if msg.get("v") != 1 or msg.get("type") != "ee_targets":
        return None
    try:
        seq = int(msg["seq"])
        ts = float(msg["t"])
        clutch = int(msg["clutch"])
        precision = int(msg["precision"])
        reset = int(msg["reset"])
        frame = str(msg["frame"])
    except (TypeError, ValueError):
        return None
    arms_raw = msg.get("arms") or []
    parsed_arms: List[ArmTarget] = []
    for arm in arms_raw:
        parsed = _validate_arm(arm)
        if parsed:
            parsed_arms.append(parsed)
    if not parsed_arms:
        return None
    return EETargetsState(
        seq=seq,
        t=ts,
        frame=frame,
        clutch=clutch,
        precision=precision,
        reset=reset,
        arms=parsed_arms,
        raw=msg,
    )


class EETargetsReceiver:
    """UDP listener that stores the latest valid ee_targets packet."""

    def __init__(self, bind_ip: str = "0.0.0.0", port: int = 5005, timeout_s: float = 0.5):
        self.bind_ip = bind_ip
        self.port = port
        self.timeout_s = timeout_s
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._alive = False
        self._latest: Optional[EETargetsState] = None
        self._latest_time: float = 0.0
        self._last_reset_flag: int = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._alive:
            return
        self._alive = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._alive = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._sock:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _run(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.2)
        self._sock.bind((self.bind_ip, self.port))
        while self._alive:
            try:
                data, _ = self._sock.recvfrom(2048)
                state = parse_packet(data)
                if not state:
                    continue
                with self._lock:
                    if self._latest is None or state.seq > self._latest.seq:
                        self._latest = state
                        self._latest_time = time.time()
                        self._last_reset_flag = state.reset
            except socket.timeout:
                continue
            except OSError:
                break

    def inject(self, payload: Dict) -> None:
        state = parse_packet(json.dumps(payload).encode("utf-8"))
        if not state:
            return
        with self._lock:
            if self._latest is None or state.seq > self._latest.seq:
                self._latest = state
                self._latest_time = time.time()
                self._last_reset_flag = state.reset

    def get_latest(self) -> Tuple[Optional[EETargetsState], float, bool]:
        """Returns (state, age_seconds, reset_edge)."""
        with self._lock:
            state = self._latest
            ts = self._latest_time
            prev_reset = self._last_reset_flag
            self._last_reset_flag = state.reset if state else 0
        if not state:
            return None, float("inf"), False
        age = time.time() - ts
        reset_edge = prev_reset == 0 and state.reset == 1
        if age > self.timeout_s:
            return None, age, reset_edge
        return state, age, reset_edge
