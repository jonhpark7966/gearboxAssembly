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
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

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

# Make the top-level repo importable so we can use teleop utils without installation.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Galaxea_Lab_External.tasks  # noqa: F401

try:
    from teleop.ee_targets import EETargetsReceiver
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        f"[ERROR] Could not import teleop utilities. Ensure repo root ({REPO_ROOT}) is on PYTHONPATH. {exc}"
    ) from exc


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


class TeleopRulePolicy:
    """Minimal rule_policy replacement that only drives grippers from UDP."""

    def __init__(self, env, receiver: EETargetsReceiver, max_open: float = 0.04):
        self.env = env
        self.receiver = receiver
        self.device = env.device
        self.max_open = max_open
        self.count = 0
        self.total_time_steps = 1_000_000_000  # keep episode alive
        # Cache joint ids for grippers
        self.left_ids = [int(i) for i in env._left_gripper_dof_idx]  # type: ignore[attr-defined]
        self.right_ids = [int(i) for i in env._right_gripper_dof_idx]  # type: ignore[attr-defined]
        self.last_seq: Optional[int] = None

    def _grip_to_joint(self, grip: float) -> float:
        grip_clamped = max(0.0, min(1.0, float(grip)))
        return grip_clamped * self.max_open

    def get_action(self):
        state, _ = self.receiver.get_latest()
        if not state:
            return None, None

        joint_ids = []
        values = []
        for arm in state.arms:
            if arm.id == "L":
                joint_ids.extend(self.left_ids)
            elif arm.id == "R":
                joint_ids.extend(self.right_ids)
            else:
                continue
            values.append(self._grip_to_joint(arm.grip))

        if not joint_ids:
            return None, None

        self.last_seq = state.seq
        action = torch.tensor([values], device=self.device)
        return action, joint_ids

    def reset_counters(self):
        self.count = 0
        self.last_seq = None


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

    # Start UDP receiver (threaded) and build a teleop rule policy that consumes it.
    receiver = EETargetsReceiver(bind_ip=args_cli.listen_ip, port=args_cli.port)
    receiver.start()
    teleop_policy = TeleopRulePolicy(env.unwrapped, receiver)

    def install_teleop_policy():
        env.unwrapped.rule_policy = teleop_policy
        env.unwrapped.env_step_action = None
        env.unwrapped.env_step_joint_ids = None
        teleop_policy.reset_counters()
        print("[INFO] teleop rule_policy installed (gripper only).")

    try:
        bound_reset = env.unwrapped._reset_idx
        original_reset_idx_func = getattr(bound_reset, "__func__", None)

        def original_reset_call(env_ids=None):
            if original_reset_idx_func:
                return original_reset_idx_func(env.unwrapped, env_ids)
            return bound_reset(env_ids)

        def patched_reset_idx(env_ids=None):  # type: ignore[override]
            result = original_reset_call(env_ids)
            install_teleop_policy()
            return result

        env.unwrapped._reset_idx = patched_reset_idx  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not wrap _reset_idx: {exc}")

    install_teleop_policy()

    # Precompute a zero action to keep the sim ticking without control.
    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    with suppress_print():
        env.reset()
    last_heartbeat = time.time()
    heartbeat_s = 5.0
    waiting_for_first_packet = True
    last_logged_seq: Optional[int] = None

    try:
        while simulation_app.is_running():
            state, age = receiver.get_latest()
            if state and state.seq != last_logged_seq:
                grip_summary = ", ".join(f"{arm.id}=grip{arm.grip:.2f}" for arm in state.arms)
                print(f"[UDP] seq={state.seq} age={age:.3f}s frame={state.frame} {grip_summary}")
                last_logged_seq = state.seq
            now = time.time()
            if waiting_for_first_packet and not state:
                # Hold the sim until the first packet arrives so we can prove ingress.
                if now - last_heartbeat > heartbeat_s:
                    print("[HEARTBEAT] waiting for UDP... udp_packets=0")
                    last_heartbeat = now
                time.sleep(0.01)
                continue
            elif waiting_for_first_packet and state:
                print("[INFO] First UDP packet received; starting sim stepping.")
                waiting_for_first_packet = False

            with torch.inference_mode():
                with suppress_print():
                    env.step(zero_action)

            if now - last_heartbeat > heartbeat_s:
                print(
                    f"[HEARTBEAT] sim ok, last_seq={teleop_policy.last_seq}, "
                    f"age={age if state else float('inf'):.2f}s"
                )
                last_heartbeat = now
    finally:
        receiver.stop()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
