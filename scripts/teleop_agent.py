#!/usr/bin/env python3
"""
Teleop harness that mirrors random_agent.py but consumes UDP ee_targets v1 and
drives the Galaxea arms via Differential IK. Gripper control is applied, and
XYZ+orientation are tracked per arm.
"""

from __future__ import annotations

import argparse
import builtins
import json
import socket
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

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
parser.add_argument("--state_port", type=int, default=5006, help="UDP state reply port (ee_state_request -> ee_state).")
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
import isaacsim.core.utils.torch as torch_utils

# Make the top-level repo importable so we can use teleop utils without installation.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Galaxea_Lab_External.tasks  # noqa: F401

try:
    from teleop_utils.ee_targets import EETargetsReceiver
    from teleop_utils.filters import PoseCommandFilter
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        f"[ERROR] Could not import teleop utilities. Ensure repo root ({REPO_ROOT}) is on PYTHONPATH. {exc}"
    ) from exc

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms


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


class StateResponder:
    """Listens for ee_state_request and replies with current poses/grips."""

    def __init__(self, policy: TeleopRulePolicy, bind_ip: str, port: int):
        self.policy = policy
        self.bind_ip = bind_ip
        self.port = port
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._alive = False

    def start(self):
        if self._alive:
            return
        self._alive = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[INFO] ee_state responder listening on {self.bind_ip}:{self.port}")

    def stop(self):
        self._alive = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._sock:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _run(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.2)
        self._sock.bind((self.bind_ip, self.port))
        while self._alive:
            try:
                data, addr = self._sock.recvfrom(2048)
                try:
                    msg = json.loads(data.decode("utf-8"))
                except Exception:
                    continue
                if not isinstance(msg, dict) or msg.get("type") != "ee_state_request":
                    continue
                payload = self.policy.build_state_payload()
                self._sock.sendto(json.dumps(payload).encode("utf-8"), addr)
            except socket.timeout:
                continue
            except OSError:
                break


class ArmIKBridge:
    """Per-arm helper that transforms ee_targets into joint targets via DiffIK."""

    def __init__(self, env, arm_prefix: str):
        self.env = env
        self.scene = env.scene
        self.robot = self.scene["robot"]
        self.device = env.device
        self.arm_prefix = arm_prefix

        self.entity_cfg = SceneEntityCfg(
            "robot", joint_names=[f"{arm_prefix}_arm_joint.*"], body_names=[f"{arm_prefix}_arm_link6"]
        )
        self.gripper_cfg = SceneEntityCfg("robot", joint_names=[f"{arm_prefix}_gripper_axis.*"])
        self.entity_cfg.resolve(self.scene)
        self.gripper_cfg.resolve(self.scene)

        if not self.entity_cfg.body_ids:
            raise RuntimeError(f"Could not resolve body for {arm_prefix} arm.")
        self.ee_body_id = self.entity_cfg.body_ids[0]
        self.ee_jacobi_idx = self.ee_body_id - 1 if self.robot.is_fixed_base else self.ee_body_id

        diff_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.ik = DifferentialIKController(diff_cfg, num_envs=self.scene.num_envs, device=self.device)

        self._warned_frames: Set[str] = set()

    def _pose_to_world(
        self, pos: Tuple[float, float, float], quat: Tuple[float, float, float, float], frame: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform a pose expressed in `frame` into world coordinates."""
        p_t = torch.tensor(pos, device=self.device).unsqueeze(0)
        q_t = torch.tensor(quat, device=self.device).unsqueeze(0)
        frame_l = (frame or "world").lower()

        if frame_l in ("world",):
            return p_t, q_t

        if frame_l == "torso":
            root_state = self.robot.data.root_state_w[:, 0:7]
            base_pos = root_state[:, 0:3]
            base_quat = root_state[:, 3:7]
            q_w, p_w = torch_utils.tf_combine(base_quat, base_pos, q_t, p_t)
            return p_w, q_w

        if frame_l not in self._warned_frames:
            print(f"[WARN] Unsupported frame '{frame}'; interpreting pose as world.")
            self._warned_frames.add(frame_l)
        return p_t, q_t

    def current_pose(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        pose_w = self.robot.data.body_state_w[:, self.ee_body_id, 0:7]
        return tuple(pose_w[0, 0:3].tolist()), tuple(pose_w[0, 3:7].tolist())

    def compute_joint_targets(self, cmd) -> torch.Tensor:
        target_p, target_q = self._pose_to_world(cmd.p, cmd.q, cmd.frame)
        ik_commands = torch.cat([target_p, target_q], dim=-1)
        self.ik.set_command(ik_commands)

        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.entity_cfg.joint_ids]
        ee_pose_w = self.robot.data.body_state_w[:, self.ee_body_id, 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        joint_pos = self.robot.data.joint_pos[:, self.entity_cfg.joint_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        joint_pos_des = self.ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        return joint_pos_des


class TeleopRulePolicy:
    """rule_policy replacement that consumes ee_targets and applies IK + gripper targets."""

    def __init__(self, env, receiver: EETargetsReceiver, max_open: float = 0.04):
        self.env = env
        self.scene = env.scene
        self.robot = self.scene["robot"]
        self.receiver = receiver
        self.device = env.device
        self.max_open = max_open
        self.count = 0
        self.total_time_steps = 1_000_000_000  # keep episode alive

        # IK helpers per arm
        self._arms: Dict[str, ArmIKBridge] = {
            "L": ArmIKBridge(env, "left"),
            "R": ArmIKBridge(env, "right"),
        }
        self.left_grip_ids = [int(i) for i in env._left_gripper_dof_idx]  # type: ignore[attr-defined]
        self.right_grip_ids = [int(i) for i in env._right_gripper_dof_idx]  # type: ignore[attr-defined]
        self.joint_order = (
            self._arms["L"].entity_cfg.joint_ids
            + self._arms["R"].entity_cfg.joint_ids
            + self.left_grip_ids
            + self.right_grip_ids
        )

        # Filtered command smoother
        self.filter = PoseCommandFilter(
            timeout_s=0.5,
            alpha_p=0.2,
            alpha_q=0.2,
            alpha_grip=0.3,
            grip_deadband=0.02,
            current_pose_fn=self._current_pose_fn,
        )
        self.last_seq: Optional[int] = None

    def _grip_to_joint(self, grip: float) -> float:
        grip_clamped = max(0.0, min(1.0, float(grip)))
        return grip_clamped * self.max_open

    def get_action(self):
        state, _ = self.receiver.get_latest()
        commands = self.filter.update(state)
        if not commands:
            return None, None

        num_envs = self.scene.num_envs
        # Default to current joint positions to hold steady.
        left_des = self.robot.data.joint_pos[:, self._arms["L"].entity_cfg.joint_ids]
        right_des = self.robot.data.joint_pos[:, self._arms["R"].entity_cfg.joint_ids]
        left_grip_des = self.robot.data.joint_pos[:, self.left_grip_ids]
        right_grip_des = self.robot.data.joint_pos[:, self.right_grip_ids]

        for cmd in commands:
            arm = self._arms.get(cmd.id)
            if arm is None:
                continue
            try:
                joint_targets = arm.compute_joint_targets(cmd)
                if cmd.id == "L":
                    left_des = joint_targets
                else:
                    right_des = joint_targets
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] IK failed for arm {cmd.id}: {exc}")

            grip_val = self._grip_to_joint(cmd.grip)
            if cmd.id == "L":
                left_grip_des = torch.full((num_envs, len(self.left_grip_ids)), grip_val, device=self.device)
            else:
                right_grip_des = torch.full((num_envs, len(self.right_grip_ids)), grip_val, device=self.device)

        action = torch.cat([left_des, right_des, left_grip_des, right_grip_des], dim=1)
        if state:
            self.last_seq = state.seq
        return action, self.joint_order

    def reset_counters(self):
        self.count = 0
        self.last_seq = None
        try:
            self.filter._last.clear()  # reset cached commands between episodes
        except Exception:
            pass

    def _current_pose_fn(self, arm_id: str):
        arm = self._arms.get(arm_id)
        if arm:
            return arm.current_pose()
        return None

    def build_state_payload(self) -> Dict:
        """Return current EE pose and gripper state for both arms in world frame."""
        arms = []
        for arm_id, arm in self._arms.items():
            try:
                p, q = arm.current_pose()
                if arm_id == "L":
                    grip_ids = self.left_grip_ids
                else:
                    grip_ids = self.right_grip_ids
                grip_val = 0.0
                if grip_ids:
                    grip_val = float(self.robot.data.joint_pos[0, grip_ids].mean().item())
            except Exception:
                continue
            arms.append(
                {
                    "id": arm_id,
                    "ee_frame": f"{arm.arm_prefix}_gripper_tcp",
                    "p": p,
                    "q": q,
                    "grip": grip_val / self.max_open if self.max_open > 0 else 0.0,
                }
            )
        return {
            "type": "ee_state",
            "frame": "world",
            "t": time.time(),
            "arms": arms,
        }


def main() -> None:
    """Bring up env, spin sim with zero actions, and log UDP ee_targets packets."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] Gym observation space: {env.observation_space}")
    print(f"[INFO] Gym action space: {env.action_space}")

    # Start UDP receiver (threaded) and build a teleop rule policy that consumes it.
    receiver = EETargetsReceiver(bind_ip=args_cli.listen_ip, port=args_cli.port)
    receiver.start()
    teleop_policy = TeleopRulePolicy(env.unwrapped, receiver)
    state_responder = StateResponder(teleop_policy, args_cli.listen_ip, args_cli.state_port)
    state_responder.start()

    def install_teleop_policy():
        env.unwrapped.rule_policy = teleop_policy
        env.unwrapped.env_step_action = None
        env.unwrapped.env_step_joint_ids = None
        teleop_policy.reset_counters()
        print("[INFO] teleop rule_policy installed (arms + grippers via IK).")

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
                arm_summaries = []
                for arm in state.arms:
                    arm_summaries.append(
                        f"{arm.id}: p=({arm.p[0]:.3f},{arm.p[1]:.3f},{arm.p[2]:.3f}) "
                        f"q=({arm.q[0]:.2f},{arm.q[1]:.2f},{arm.q[2]:.2f},{arm.q[3]:.2f}) "
                        f"grip={arm.grip:.2f}"
                    )
                prec = int(getattr(state, "precision", 0))
                print(
                    f"[UDP] seq={state.seq} age={age:.3f}s frame={state.frame} precision={prec} | "
                    + " | ".join(arm_summaries)
                )
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
        state_responder.stop()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
