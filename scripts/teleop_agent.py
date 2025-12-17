#!/usr/bin/env python3
"""
Minimal teleop harness that mirrors random_agent.py:
- Brings up an Isaac Lab env from a task name.
- Steps the sim with zero actions (teleop control will be added later).
- Listens for UDP ee_targets packets and logs them so we can prove packets land.
"""

from __future__ import annotations

import argparse
import builtins
import json
import socket
import time
from contextlib import contextmanager
from typing import Optional, Tuple

from isaaclab.app import AppLauncher

# CLI args (same shape as other *_agent scripts)
parser = argparse.ArgumentParser(description="Teleop UDP smoke test (no control yet).")
parser.add_argument(
    "--task",
    type=str,
    default="Template-Galaxea-Lab-External-Direct-v0",
    help="Task name (see scripts/list_envs.py).",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--listen_ip", type=str, default="0.0.0.0", help="UDP listen IP.")
parser.add_argument("--port", type=int, default=5005, help="UDP listen port.")
parser.add_argument("--log_payload", action="store_true", help="Print full UDP payloads.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations (matches random_agent).",
)
# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
# Parse args
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Galaxea_Lab_External.tasks  # noqa: F401


def _stub_rule_policy(target_env) -> None:
    """Replace the env's rule_policy.get_action with a no-op so no scripted moves run."""
    try:
        rule_policy = target_env.rule_policy

        def _noop_get_action():
            return None, None

        rule_policy.get_action = _noop_get_action  # type: ignore[attr-defined]
        # Also clear any cached step action/joint ids.
        target_env.env_step_action = None
        target_env.env_step_joint_ids = None
        print("[INFO] rule_policy.get_action stubbed (no scripted motion).")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not stub rule_policy (may not exist): {exc}")


@contextmanager
def suppress_print():
    """Temporarily silence print statements (to hide noisy env logs)."""

    original_print = builtins.print
    try:
        builtins.print = lambda *args, **kwargs: None  # type: ignore[assignment]
        yield
    finally:
        builtins.print = original_print


class UdpPacketLogger:
    """Non-blocking UDP receiver that prints packet summaries."""

    def __init__(self, listen_ip: str, port: int, log_payload: bool = False):
        self.log_payload = log_payload
        self.packet_count = 0
        self.last_seq: Optional[int] = None
        self.last_from: Optional[Tuple[str, int]] = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((listen_ip, port))
        self.sock.setblocking(False)
        print(f"[UDP] Listening on {listen_ip}:{port}")

    def poll(self) -> None:
        """Drain all pending UDP packets and log a summary for each."""
        while True:
            try:
                data, addr = self.sock.recvfrom(65535)
            except BlockingIOError:
                return
            raw = data.decode("utf-8", errors="replace").strip()
            self.packet_count += 1
            self.last_from = addr
            try:
                packet = json.loads(raw)
                seq = packet.get("seq")
                precision = packet.get("precision")
                frame = packet.get("frame")
                ts = packet.get("t")
                arms = packet.get("arms") or []
                arm_ids = ",".join(str(a.get("id", "?")) for a in arms)
                arm_summaries = "; ".join(
                    f"{a.get('id','?')}:p={a.get('p')} q={a.get('q')} grip={a.get('grip')}"
                    for a in arms
                )
                self.last_seq = seq
                print(
                    f"[UDP] seq={seq} frame={frame} t={ts} prec={precision} arms=[{arm_ids}] from {addr}"
                )
                if arm_summaries:
                    print(f"[UDP:arms] {arm_summaries}")
            except Exception as exc:  # noqa: BLE001
                print(f"[UDP] Received {len(data)} bytes from {addr} but failed to parse JSON: {exc}")
            else:
                if self.log_payload:
                    print(f"[UDP:payload] {raw}")

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


def main() -> None:
    """Bring up env, spin sim with zero actions, and log UDP ee_targets packets."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Patch env.reset/_reset_idx so rule_policy stays stubbed even after resets.
    try:
        original_reset_idx = env.unwrapped._reset_idx

        def patched_reset_idx(env_ids=None):  # type: ignore[override]
            result = original_reset_idx(env_ids)
            _stub_rule_policy(env.unwrapped)
            return result

        env.unwrapped._reset_idx = patched_reset_idx  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not wrap _reset_idx: {exc}")
    # Stub once now (before first reset call inside env.reset()).
    _stub_rule_policy(env.unwrapped)

    print(f"[INFO] Gym observation space: {env.observation_space}")
    print(f"[INFO] Gym action space: {env.action_space}")
    print(f"[INFO] Starting UDP listener on {args_cli.listen_ip}:{args_cli.port}")
    receiver = UdpPacketLogger(args_cli.listen_ip, args_cli.port, args_cli.log_payload)

    # Precompute a zero action to keep the sim ticking without control.
    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    with suppress_print():
        env.reset()
    last_heartbeat = time.time()
    heartbeat_s = 5.0
    waiting_for_first_packet = True

    try:
        while simulation_app.is_running():
            receiver.poll()
            now = time.time()
            if waiting_for_first_packet and receiver.packet_count == 0:
                # Hold the sim until the first packet arrives so we can prove ingress.
                if now - last_heartbeat > heartbeat_s:
                    print("[HEARTBEAT] waiting for UDP... udp_packets=0")
                    last_heartbeat = now
                time.sleep(0.01)
                continue

            waiting_for_first_packet = False
            with torch.inference_mode():
                with suppress_print():
                    env.step(zero_action)

            if now - last_heartbeat > heartbeat_s:
                print(
                    f"[HEARTBEAT] sim ok, udp_packets={receiver.packet_count}, "
                    f"last_seq={receiver.last_seq}, last_from={receiver.last_from}"
                )
                last_heartbeat = now
    finally:
        receiver.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
