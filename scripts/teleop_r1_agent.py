# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause
"""
OpenXR Teleoperation for Galaxea R1 Robot

Script to run teleoperation with Apple Vision Pro (CloudXR) for the Galaxea R1 robot
in the gearbox assembly environment.

Usage:
    ./isaaclab.sh -p submodules/gearboxAssembly/scripts/teleop_r1_agent.py \
        --task Template-Galaxea-Lab-Agent-Direct-v0 \
        --teleop_device handtracking \
        --view_mode headcam \
        --record \
        --record_dir ./data/teleop_demos
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from datetime import datetime

from isaaclab.app import AppLauncher

# Parse arguments before IsaacLab imports
parser = argparse.ArgumentParser(description="OpenXR Teleoperation for Galaxea R1")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument(
    "--task",
    type=str,
    default="Template-Galaxea-Lab-Agent-Direct-v0",
    help="Task name",
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="handtracking",
    help="Teleoperation device (handtracking, keyboard)",
)
parser.add_argument(
    "--view_mode",
    type=str,
    default="headcam",
    choices=["headcam", "world"],
    help="XR view mode",
)
parser.add_argument(
    "--anchor_mode",
    type=str,
    default="world",
    choices=["headcam", "world"],
    help="XR anchor mode",
)
parser.add_argument(
    "--gripper_open",
    type=float,
    default=0.04,
    help="Gripper open position (meters)",
)
parser.add_argument(
    "--gripper_close",
    type=float,
    default=0.0,
    help="Gripper close position (meters)",
)
parser.add_argument(
    "--pinch_close_dist",
    type=float,
    default=0.02,
    help="Pinch distance (meters) for gripper fully closed (default: 0.02)",
)
parser.add_argument(
    "--pinch_open_dist",
    type=float,
    default=0.06,
    help="Pinch distance (meters) for gripper fully open (default: 0.06)",
)
parser.add_argument(
    "--use_raw_grip",
    action="store_true",
    help="Use raw finger distance for gripper instead of retargeter (smoother, no hysteresis)",
)
parser.add_argument(
    "--record",
    action="store_true",
    help="Enable data recording",
)
parser.add_argument(
    "--record_dir",
    type=str,
    default="./data/teleop_demos",
    help="Directory to save recordings",
)
parser.add_argument(
    "--visualize_targets",
    action="store_true",
    help="Visualize EE target markers",
)
parser.add_argument(
    "--hand_markers",
    action="store_true",
    help="Show hand markers",
)
parser.add_argument(
    "--no_randomize_objects",
    action="store_false",
    dest="randomize_objects",
    default=True,
    help="Disable object randomization on reset (default: randomize)",
)
parser.add_argument(
    "--debug_log_dir",
    type=str,
    default="./logs/teleop_debug",
    help="Directory to save debug logs",
)
parser.add_argument(
    "--debug_console",
    action="store_true",
    help="Print verbose debug info to console every frame",
)
parser.add_argument(
    "--debug_frames",
    type=int,
    default=200,
    help="Number of frames to print debug output (default: 200)",
)

# Add AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Unified XR mode detection
IS_XR_MODE = "handtracking" in args_cli.teleop_device.lower()

# Configure AppLauncher with XR if using hand tracking
app_launcher_args = vars(args_cli)
if IS_XR_MODE:
    app_launcher_args["xr"] = True

# Launch application
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

# Rest of imports after AppLauncher
import logging
import torch
import gymnasium as gym
import numpy as np

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr import OpenXRDevice, remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, SPHERE_MARKER_CFG
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_mul, quat_inv, quat_apply
# Import the task module to register the environment
import Galaxea_Lab_External.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Import data recorder
from teleop_data_recorder import TeleopDataRecorder

# Visualization:
# - --visualize_targets: frame markers for EE targets
# - --hand_markers: spheres for wrist/thumb/index (OpenXR only)

logger = logging.getLogger(__name__)


# ============================================================================
# Debug Logger Setup
# ============================================================================
class TeleopDebugLogger:
    """File-based debug logger for teleoperation data."""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"teleop_debug_{timestamp}.log")
        self.frame_count = 0
        self._file = open(self.log_file, "w")
        self._write_header()
        print(f"[Debug] Logging to: {self.log_file}")

    def _write_header(self):
        self._file.write("# Teleop Debug Log\n")
        self._file.write(f"# Started: {datetime.now().isoformat()}\n")
        self._file.write("# Format: frame, left_pos(3), left_quat(4), left_grip, right_pos(3), right_quat(4), right_grip\n")
        self._file.write("# Grip values: GripperRetargeter outputs -1(close) / +1(open)\n")
        self._file.write("#" + "=" * 100 + "\n")
        self._file.flush()

    def log_raw_action(self, raw_action):
        """Log raw action from teleop device."""
        if raw_action is None:
            return
        arr = raw_action.cpu().numpy() if hasattr(raw_action, "cpu") else np.array(raw_action)
        self._file.write(f"[{self.frame_count}] RAW: shape={arr.shape}, data={arr.flatten()[:20]}...\n")

    def log_parsed(self, left_pos, left_quat, left_grip, right_pos, right_quat, right_grip):
        """Log parsed teleop values."""
        def to_list(t):
            if hasattr(t, "cpu"):
                return t.cpu().numpy().flatten().tolist()
            return list(t)

        self._file.write(
            f"[{self.frame_count}] PARSED: "
            f"L_pos={to_list(left_pos)[:3]}, L_quat={to_list(left_quat)[:4]}, L_grip={to_list(left_grip)}, "
            f"R_pos={to_list(right_pos)[:3]}, R_quat={to_list(right_quat)[:4]}, R_grip={to_list(right_grip)}\n"
        )

    def log_targets(self, left_pos_target, left_quat_target, right_pos_target, right_quat_target):
        """Log target positions after reanchoring."""
        def to_list(t):
            if hasattr(t, "cpu"):
                return t.cpu().numpy().flatten().tolist()
            return list(t)

        self._file.write(
            f"[{self.frame_count}] TARGETS: "
            f"L_pos={to_list(left_pos_target)[:3]}, L_quat={to_list(left_quat_target)[:4]}, "
            f"R_pos={to_list(right_pos_target)[:3]}, R_quat={to_list(right_quat_target)[:4]}\n"
        )

    def log_gripper(self, left_gripper_joint, right_gripper_joint, left_grip_raw, right_grip_raw,
                    left_gripper_curr=None, right_gripper_curr=None):
        """Log gripper mapping with optional current position."""
        curr_str = ""
        if left_gripper_curr is not None and right_gripper_curr is not None:
            curr_str = f", L_curr={left_gripper_curr:.4f}, R_curr={right_gripper_curr:.4f}"
        self._file.write(
            f"[{self.frame_count}] GRIPPER: "
            f"L_raw={left_grip_raw:.3f} -> L_joint={left_gripper_joint:.4f}, "
            f"R_raw={right_grip_raw:.3f} -> R_joint={right_gripper_joint:.4f}{curr_str}\n"
        )

    def log_reanchor(self, left_pos_offset, left_quat_offset, right_pos_offset, right_quat_offset):
        """Log reanchoring offsets."""
        def to_list(t):
            if hasattr(t, "cpu"):
                return t.cpu().numpy().flatten().tolist()
            return list(t)

        self._file.write(
            f"[{self.frame_count}] REANCHOR: "
            f"L_pos_off={to_list(left_pos_offset)[:3]}, L_quat_off={to_list(left_quat_offset)[:4]}, "
            f"R_pos_off={to_list(right_pos_offset)[:3]}, R_quat_off={to_list(right_quat_offset)[:4]}\n"
        )
        self._file.flush()

    def log_event(self, event: str):
        """Log an event."""
        self._file.write(f"[{self.frame_count}] EVENT: {event}\n")
        self._file.flush()

    def next_frame(self):
        """Increment frame counter."""
        self.frame_count += 1
        # Flush every 100 frames
        if self.frame_count % 100 == 0:
            self._file.flush()

    def close(self):
        """Close the log file."""
        self._file.write(f"# Ended: {datetime.now().isoformat()}\n")
        self._file.write(f"# Total frames: {self.frame_count}\n")
        self._file.close()
        print(f"[Debug] Log saved: {self.log_file}")


def setup_ik_controller(env, arm_name: str, device: str):
    """Setup differential IK controller for an arm.

    Args:
        env: The environment.
        arm_name: "left" or "right".
        device: torch device.

    Returns:
        Tuple of (controller, arm_entity_cfg, body_ids, joint_ids).
    """
    # Create IK controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg,
        num_envs=env.num_envs,
        device=device,
    )

    # Configure arm entity
    arm_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=[f"{arm_name}_arm_joint.*"],
        body_names=[f"{arm_name}_arm_link6"],
    )

    # Resolve entity configuration
    robot = env.unwrapped.robot
    arm_entity_cfg.resolve(env.unwrapped.scene)

    joint_ids = arm_entity_cfg.joint_ids
    body_ids = arm_entity_cfg.body_ids

    return diff_ik_controller, arm_entity_cfg, body_ids, joint_ids


def compute_ik(
    env,
    controller: DifferentialIKController,
    arm_entity_cfg: SceneEntityCfg,
    body_ids: list,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> torch.Tensor:
    """Compute joint targets using IK.

    Args:
        env: The environment.
        controller: DifferentialIK controller.
        arm_entity_cfg: Arm entity configuration.
        body_ids: Body IDs for EE.
        target_pos: Target position (num_envs, 3).
        target_quat: Target quaternion wxyz (num_envs, 4).

    Returns:
        Joint position targets (num_envs, num_joints).
    """
    robot = env.unwrapped.robot

    # Get Jacobian
    if robot.is_fixed_base:
        ee_jacobi_idx = body_ids[0] - 1
    else:
        ee_jacobi_idx = body_ids[0]

    jacobian = robot.root_physx_view.get_jacobians()[
        :, ee_jacobi_idx, :, arm_entity_cfg.joint_ids
    ]

    # Get current EE pose
    ee_pose_w = robot.data.body_state_w[:, body_ids[0], 0:7]
    root_pose_w = robot.data.root_state_w[:, 0:7]

    # Transform to robot base frame
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        ee_pose_w[:, 0:3],
        ee_pose_w[:, 3:7],
    )

    # Transform target to robot base frame (DiffIK expects commands in base frame)
    target_pos_b, target_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        target_pos,
        target_quat,
    )

    # Set IK command
    ik_command = torch.cat([target_pos_b, target_quat_b], dim=-1)
    controller.set_command(ik_command)

    # Get current joint positions
    joint_pos = robot.data.joint_pos[:, arm_entity_cfg.joint_ids]

    # Compute IK
    joint_pos_des = controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    return joint_pos_des


def map_grip_to_joint(grip: float, open_pos: float, close_pos: float) -> float:
    """Map grip value from GripperRetargeter (-1/+1) to gripper joint position.

    GripperRetargeter output:
        - +1.0 = open (fingers apart)
        - -1.0 = close (fingers together / pinch)

    Gripper joint range (Galaxea R1):
        - 0.04 = open
        - 0.0 = close

    Args:
        grip: Grip value from GripperRetargeter: -1 (close) to +1 (open).
        open_pos: Gripper open joint position (default 0.04).
        close_pos: Gripper closed joint position (default 0.0).

    Returns:
        Gripper joint position.
    """
    # Convert from [-1, +1] to [0, 1] where 0=close, 1=open
    # grip=-1 -> normalized=0 (close)
    # grip=+1 -> normalized=1 (open)
    normalized = (grip + 1.0) / 2.0

    # Map to joint position: normalized=0 -> close_pos, normalized=1 -> open_pos
    return close_pos + normalized * (open_pos - close_pos)


def compute_grip_from_finger_distance(
    thumb_pos: np.ndarray,
    index_pos: np.ndarray,
    close_dist: float,
    open_dist: float,
) -> float:
    """Compute normalized grip value from raw finger distance.

    This bypasses the GripperRetargeter's hysteresis-based approach and provides
    smooth, continuous gripper control based on actual finger distance.

    Args:
        thumb_pos: 3D position of thumb tip (meters).
        index_pos: 3D position of index tip (meters).
        close_dist: Finger distance (meters) for fully closed gripper.
        open_dist: Finger distance (meters) for fully open gripper.

    Returns:
        Grip value in [-1, +1] where -1=close, +1=open.
    """
    distance = np.linalg.norm(thumb_pos - index_pos)

    # Map distance to [-1, +1] range
    # distance <= close_dist -> -1.0 (close)
    # distance >= open_dist -> +1.0 (open)
    # linear interpolation in between
    if distance <= close_dist:
        return -1.0
    elif distance >= open_dist:
        return 1.0
    else:
        # Linear interpolation: close_dist -> -1.0, open_dist -> +1.0
        normalized = (distance - close_dist) / (open_dist - close_dist)
        return -1.0 + 2.0 * normalized


def main() -> None:
    """Main teleoperation loop."""

    # Parse environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task

    # Optional: disable object randomization to debug stability / teleop mapping
    if hasattr(env_cfg, "randomize_objects"):
        env_cfg.randomize_objects = args_cli.randomize_objects

    # Disable timeout for teleoperation
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    # Configure for XR - don't remove camera configs since we need them for recording
    if IS_XR_MODE:
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # Create environment
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    device = env.device

    # Initialize debug logger
    debug_logger = TeleopDebugLogger(args_cli.debug_log_dir)
    debug_logger.log_event(f"Environment created: {args_cli.task}, device={device}, XR_MODE={IS_XR_MODE}")

    # Setup IK controllers for both arms
    left_ik, left_arm_cfg, left_body_ids, left_joint_ids = setup_ik_controller(env, "left", device)
    right_ik, right_arm_cfg, right_body_ids, right_joint_ids = setup_ik_controller(env, "right", device)

    # Get gripper joint indices
    robot = env.robot
    left_gripper_idx, _ = robot.find_joints("left_gripper_axis1")
    right_gripper_idx, _ = robot.find_joints("right_gripper_axis1")

    # Teleoperation state
    teleoperation_active = False if IS_XR_MODE else True
    should_reset = False
    is_recording = False

    # Reanchoring offsets (position + orientation)
    left_pos_offset = torch.zeros(3, device=device)
    right_pos_offset = torch.zeros(3, device=device)
    # Quaternion offsets (wxyz format, identity = [1, 0, 0, 0])
    left_quat_offset = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    right_quat_offset = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    need_reanchor = True

    # Hand marker visualization is now handled by Se3AbsRetargeter with enable_visualization=True
    # The retargeter shows coordinate frame markers at the target end-effector positions
    if args_cli.hand_markers:
        print("[Teleop] Hand markers enabled via Se3AbsRetargeter's built-in visualization")
        print("[Teleop] Coordinate frame markers will appear at target EE positions")

    # Verbose frame counter
    verbose_frame_count = 0

    # Data recorder
    recorder = TeleopDataRecorder(
        save_dir=args_cli.record_dir,
        record_rgb=True,
        record_depth=False,
    ) if args_cli.record else None

    # First action diagnostic flag
    first_action_printed = False

    # Callback handlers
    def start_teleoperation() -> None:
        nonlocal teleoperation_active, is_recording, need_reanchor, first_action_printed
        teleoperation_active = True
        need_reanchor = True  # Reanchor on next frame
        first_action_printed = False  # Reset to print first action after start
        if recorder and args_cli.record:
            recorder.start_recording()
            is_recording = True
        debug_logger.log_event("START gesture detected - teleop activated")
        print("[Teleop] Activated - START gesture detected")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active, is_recording
        teleoperation_active = False
        if recorder and is_recording:
            recorder.stop_recording(save=True)
            is_recording = False
        debug_logger.log_event("STOP gesture detected - teleop deactivated")
        print("[Teleop] Deactivated - STOP gesture detected")

    def reset_environment() -> None:
        nonlocal should_reset, need_reanchor
        should_reset = True
        need_reanchor = True
        debug_logger.log_event("RESET triggered")
        print("[Teleop] Reset triggered")

    # Teleoperation callbacks
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_environment,
        "R": reset_environment,  # Keyboard shortcut
    }

    # Create teleoperation device
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            # Print diagnostic info about teleop device configuration
            device_cfg = env_cfg.teleop_devices.devices[args_cli.teleop_device]
            print(f"\n[Teleop] === Device Configuration ===")
            print(f"[Teleop] Device: {args_cli.teleop_device}")
            if hasattr(device_cfg, "retargeters"):
                print(f"[Teleop] Retargeters ({len(device_cfg.retargeters)}):")
                for i, ret_cfg in enumerate(device_cfg.retargeters):
                    ret_type = type(ret_cfg).__name__
                    bound_hand = getattr(ret_cfg, "bound_hand", "N/A")
                    print(f"[Teleop]   [{i}] {ret_type} -> {bound_hand}")
                print(f"[Teleop] Expected output: 16D = [L_pos(3), L_quat(4), L_grip(1), R_pos(3), R_quat(4), R_grip(1)]")
            print(f"[Teleop] ==============================\n")

            teleop_interface = create_teleop_device(
                args_cli.teleop_device,
                env_cfg.teleop_devices.devices,
                teleoperation_callbacks,
            )
        else:
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05, rot_sensitivity=0.05)
                )
                for key, callback in teleoperation_callbacks.items():
                    try:
                        teleop_interface.add_callback(key, callback)
                    except (ValueError, TypeError):
                        pass
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Configure 'teleop_devices' in environment config for OpenXR.")
                env.close()
                simulation_app.close()
                return
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"[Teleop] Using device: {teleop_interface}")
    print(f"[Teleop] Gripper range: {args_cli.gripper_close} (close) - {args_cli.gripper_open} (open)")
    if args_cli.use_raw_grip:
        print(f"[Teleop] Gripper mode: RAW (smooth, pinch distance: {args_cli.pinch_close_dist}m close / {args_cli.pinch_open_dist}m open)")
    else:
        print("[Teleop] Gripper mode: RETARGETER (hysteresis, 0.03m close / 0.05m open)")
        print("[Teleop] Tip: Use --use_raw_grip for smoother gripper control")

    # Reset environment
    obs, _ = env.reset()
    teleop_interface.reset()

    print("[Teleop] Environment ready. Use START gesture to begin teleoperation.")
    if IS_XR_MODE:
        print("[Teleop] XR mode: Pinch thumb and index finger together, then release to START")
        print(f"[Teleop] Debug logs: {debug_logger.log_file}")

    # Optional visualization markers
    left_target_marker = None
    right_target_marker = None
    left_hand_marker = None
    right_hand_marker = None

    if args_cli.visualize_targets:
        left_cfg = FRAME_MARKER_CFG.copy()
        left_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        left_cfg.prim_path = "/Visuals/ee_goal_left"
        left_target_marker = VisualizationMarkers(left_cfg)

        right_cfg = FRAME_MARKER_CFG.copy()
        right_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        right_cfg.prim_path = "/Visuals/ee_goal_right"
        right_target_marker = VisualizationMarkers(right_cfg)

    if args_cli.hand_markers:
        if isinstance(teleop_interface, OpenXRDevice):
            left_cfg = SPHERE_MARKER_CFG.copy()
            left_cfg.prim_path = "/Visuals/hand_markers_left"
            left_cfg.markers["sphere"].radius = 0.015
            left_hand_marker = VisualizationMarkers(left_cfg)

            right_cfg = SPHERE_MARKER_CFG.copy()
            right_cfg.prim_path = "/Visuals/hand_markers_right"
            right_cfg.markers["sphere"].radius = 0.015
            right_hand_marker = VisualizationMarkers(right_cfg)
        else:
            print("[Teleop] --hand_markers requires OpenXR handtracking; ignoring for this device.")

    # Main simulation loop
    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                # Get teleop command
                # For OpenXR with bimanual retargeters, output is:
                # [left_pos(3), left_quat(4), left_grip(1), right_pos(3), right_quat(4), right_grip(1)]
                raw_action = teleop_interface.advance()

                if teleoperation_active and raw_action is not None:
                    # Log raw action for debugging
                    debug_logger.log_raw_action(raw_action)

                    # Print first action diagnostics
                    if not first_action_printed:
                        first_action_printed = True
                        print(f"\n[Teleop] === First Action Diagnostic ===")
                        print(f"[Teleop] raw_action.shape = {raw_action.shape}")
                        print(f"[Teleop] raw_action.dtype = {raw_action.dtype}")
                        print(f"[Teleop] raw_action values:")
                        arr = raw_action.cpu().numpy().flatten()
                        print(f"[Teleop]   [0:3]   L_pos  = {arr[0:3]}")
                        print(f"[Teleop]   [3:7]   L_quat = {arr[3:7]} (wxyz)")
                        print(f"[Teleop]   [7:8]   L_grip = {arr[7:8]} (-1=close, +1=open)")
                        print(f"[Teleop]   [8:11]  R_pos  = {arr[8:11]}")
                        print(f"[Teleop]   [11:15] R_quat = {arr[11:15]} (wxyz)")
                        print(f"[Teleop]   [15:16] R_grip = {arr[15:16]} (-1=close, +1=open)")
                        print(f"[Teleop] =====================================\n")

                    # Parse the teleop output
                    # The Se3AbsRetargeter outputs: [pos(3), quat(4)]
                    # The GripperRetargeter outputs: [grip(1)]
                    # With 4 retargeters (2 per hand), total is 16D

                    if raw_action.shape[-1] >= 16:
                        # Full bimanual output
                        left_pos = raw_action[..., 0:3]
                        left_quat = raw_action[..., 3:7]
                        left_grip = raw_action[..., 7:8]
                        right_pos = raw_action[..., 8:11]
                        right_quat = raw_action[..., 11:15]
                        right_grip = raw_action[..., 15:16]
                    else:
                        # Fallback: use current robot pose
                        debug_logger.log_event(f"Fallback mode: raw_action.shape={raw_action.shape}")
                        robot = env.robot
                        left_ee_pose = robot.data.body_state_w[:, left_body_ids[0], 0:7]
                        right_ee_pose = robot.data.body_state_w[:, right_body_ids[0], 0:7]
                        left_pos = left_ee_pose[:, 0:3]
                        left_quat = left_ee_pose[:, 3:7]
                        right_pos = right_ee_pose[:, 0:3]
                        right_quat = right_ee_pose[:, 3:7]
                        # Default grip = +1 (open) for GripperRetargeter convention
                        left_grip = torch.ones(env.num_envs, 1, device=device)
                        right_grip = torch.ones(env.num_envs, 1, device=device)

                    # Ensure batch dimension
                    if left_pos.dim() == 1:
                        left_pos = left_pos.unsqueeze(0)
                        left_quat = left_quat.unsqueeze(0)
                        left_grip = left_grip.unsqueeze(0)
                        right_pos = right_pos.unsqueeze(0)
                        right_quat = right_quat.unsqueeze(0)
                        right_grip = right_grip.unsqueeze(0)

                    # Log parsed values
                    debug_logger.log_parsed(left_pos, left_quat, left_grip, right_pos, right_quat, right_grip)

                    # Verbose console logging (first N frames)
                    if args_cli.debug_console and verbose_frame_count < args_cli.debug_frames:
                        lp = left_pos[0].cpu().numpy()
                        lq = left_quat[0].cpu().numpy()
                        lg = left_grip[0].cpu().numpy()
                        rp = right_pos[0].cpu().numpy()
                        rq = right_quat[0].cpu().numpy()
                        rg = right_grip[0].cpu().numpy()
                        print(f"[{verbose_frame_count:04d}] L_pos=[{lp[0]:.3f},{lp[1]:.3f},{lp[2]:.3f}] "
                              f"L_grip={lg[0]:+.2f} R_pos=[{rp[0]:.3f},{rp[1]:.3f},{rp[2]:.3f}] "
                              f"R_grip={rg[0]:+.2f}")
                        verbose_frame_count += 1
                        if verbose_frame_count == args_cli.debug_frames:
                            print(f"[Teleop] Debug console logging stopped after {args_cli.debug_frames} frames")

                    # Reanchor: calculate position and orientation offset to match current robot pose
                    if need_reanchor:
                        robot = env.robot
                        left_ee_pose = robot.data.body_state_w[:, left_body_ids[0], 0:7]
                        right_ee_pose = robot.data.body_state_w[:, right_body_ids[0], 0:7]

                        # Position offsets
                        left_pos_offset = left_ee_pose[:, 0:3] - left_pos
                        right_pos_offset = right_ee_pose[:, 0:3] - right_pos

                        # Orientation offsets: quat_offset = ee_quat * inv(xr_quat)
                        # This allows us to apply: target_quat = quat_offset * xr_quat
                        left_quat_offset = quat_mul(left_ee_pose[:, 3:7], quat_inv(left_quat))
                        right_quat_offset = quat_mul(right_ee_pose[:, 3:7], quat_inv(right_quat))

                        need_reanchor = False

                        # Log reanchoring
                        debug_logger.log_reanchor(
                            left_pos_offset, left_quat_offset,
                            right_pos_offset, right_quat_offset
                        )
                        print(f"[Teleop] Reanchored - Left pos offset: {left_pos_offset[0].cpu().numpy()}")
                        print(f"[Teleop] Reanchored - Right pos offset: {right_pos_offset[0].cpu().numpy()}")

                    # Apply position offsets
                    left_pos_target = left_pos + left_pos_offset
                    right_pos_target = right_pos + right_pos_offset

                    # Apply orientation offsets: target_quat = offset_quat * xr_quat
                    left_quat_target = quat_mul(left_quat_offset.unsqueeze(0) if left_quat_offset.dim() == 1 else left_quat_offset, left_quat)
                    right_quat_target = quat_mul(right_quat_offset.unsqueeze(0) if right_quat_offset.dim() == 1 else right_quat_offset, right_quat)

                    # Log targets
                    debug_logger.log_targets(left_pos_target, left_quat_target, right_pos_target, right_quat_target)

                    # Visualize target end-effector frames (optional)
                    if left_target_marker is not None:
                        left_target_marker.visualize(translations=left_pos_target, orientations=left_quat_target)
                    if right_target_marker is not None:
                        right_target_marker.visualize(translations=right_pos_target, orientations=right_quat_target)

                    # Visualize wrist/thumb/index markers (optional, requires OpenXR raw data)
                    if left_hand_marker is not None or right_hand_marker is not None:
                        xr_data = teleop_interface._get_raw_data() if isinstance(teleop_interface, OpenXRDevice) else None
                        if xr_data is not None:
                            # Teleop scripts assume num_envs=1; visualize env 0 only.
                            left_off = left_pos_offset[0] if left_pos_offset.dim() == 2 else left_pos_offset
                            right_off = right_pos_offset[0] if right_pos_offset.dim() == 2 else right_pos_offset

                            def _vis_hand(marker, hand_dict, offset_vec):
                                if marker is None or hand_dict is None:
                                    return
                                wrist = hand_dict.get("wrist")
                                thumb = hand_dict.get("thumb_tip")
                                index = hand_dict.get("index_tip")
                                if wrist is None or thumb is None or index is None:
                                    return
                                pts = torch.tensor(
                                    [wrist[:3], thumb[:3], index[:3]],
                                    dtype=torch.float32,
                                    device=device,
                                )
                                pts = pts + offset_vec.unsqueeze(0)
                                marker.visualize(translations=pts)

                            _vis_hand(left_hand_marker, xr_data.get(DeviceBase.TrackingTarget.HAND_LEFT), left_off)
                            _vis_hand(right_hand_marker, xr_data.get(DeviceBase.TrackingTarget.HAND_RIGHT), right_off)

                    # Compute IK for both arms (now with corrected orientation)
                    left_joint_targets = compute_ik(
                        env, left_ik, left_arm_cfg, left_body_ids,
                        left_pos_target, left_quat_target
                    )
                    right_joint_targets = compute_ik(
                        env, right_ik, right_arm_cfg, right_body_ids,
                        right_pos_target, right_quat_target
                    )

                    # Map grip to gripper joint position
                    # Option 1: Use retargeter output (with hysteresis)
                    # Option 2: Use raw finger distance (smooth, no hysteresis)
                    left_finger_dist = None
                    right_finger_dist = None
                    if args_cli.use_raw_grip and isinstance(teleop_interface, OpenXRDevice):
                        # Get raw XR data for finger positions
                        xr_raw = teleop_interface._get_raw_data()
                        if xr_raw is not None:
                            left_hand_data = xr_raw.get(DeviceBase.TrackingTarget.HAND_LEFT)
                            right_hand_data = xr_raw.get(DeviceBase.TrackingTarget.HAND_RIGHT)

                            if left_hand_data is not None:
                                left_thumb = left_hand_data["thumb_tip"][:3]
                                left_index = left_hand_data["index_tip"][:3]
                                left_finger_dist = np.linalg.norm(left_thumb - left_index)
                                left_grip_raw = compute_grip_from_finger_distance(
                                    left_thumb,
                                    left_index,
                                    args_cli.pinch_close_dist,
                                    args_cli.pinch_open_dist,
                                )
                                left_grip = torch.tensor([[left_grip_raw]], dtype=torch.float32, device=device)

                            if right_hand_data is not None:
                                right_thumb = right_hand_data["thumb_tip"][:3]
                                right_index = right_hand_data["index_tip"][:3]
                                right_finger_dist = np.linalg.norm(right_thumb - right_index)
                                right_grip_raw = compute_grip_from_finger_distance(
                                    right_thumb,
                                    right_index,
                                    args_cli.pinch_close_dist,
                                    args_cli.pinch_open_dist,
                                )
                                right_grip = torch.tensor([[right_grip_raw]], dtype=torch.float32, device=device)

                    # GripperRetargeter: -1 (close) / +1 (open)
                    left_gripper_target = map_grip_to_joint(
                        left_grip.squeeze(-1),
                        args_cli.gripper_open,
                        args_cli.gripper_close
                    )
                    right_gripper_target = map_grip_to_joint(
                        right_grip.squeeze(-1),
                        args_cli.gripper_open,
                        args_cli.gripper_close
                    )

                    # Log gripper mapping (with current position for debugging)
                    left_grip_scalar = left_grip.squeeze().item() if left_grip.numel() == 1 else left_grip[0, 0].item()
                    right_grip_scalar = right_grip.squeeze().item() if right_grip.numel() == 1 else right_grip[0, 0].item()
                    left_gripper_scalar = left_gripper_target.item() if isinstance(left_gripper_target, torch.Tensor) and left_gripper_target.numel() == 1 else float(left_gripper_target) if isinstance(left_gripper_target, (int, float)) else left_gripper_target[0].item()
                    right_gripper_scalar = right_gripper_target.item() if isinstance(right_gripper_target, torch.Tensor) and right_gripper_target.numel() == 1 else float(right_gripper_target) if isinstance(right_gripper_target, (int, float)) else right_gripper_target[0].item()

                    # Get current gripper position from robot (for logging and debugging)
                    curr_left_grip = robot.data.joint_pos[:, left_gripper_idx[0]].item()
                    curr_right_grip = robot.data.joint_pos[:, right_gripper_idx[0]].item()

                    # Log with target and current values
                    debug_logger.log_gripper(
                        left_gripper_scalar, right_gripper_scalar,
                        left_grip_scalar, right_grip_scalar,
                        curr_left_grip, curr_right_grip
                    )

                    # Debug: Print gripper values every 30 frames (2Hz at 60Hz loop)
                    if args_cli.debug_console and verbose_frame_count % 30 == 0:
                        dist_str = ""
                        if left_finger_dist is not None:
                            dist_str = f" | L_dist={left_finger_dist*100:.1f}cm R_dist={right_finger_dist*100:.1f}cm"
                        print(f"[GRIP] L: raw={left_grip_scalar:+.2f} -> target={left_gripper_scalar:.4f} | curr={curr_left_grip:.4f} | "
                              f"R: raw={right_grip_scalar:+.2f} -> target={right_gripper_scalar:.4f} | curr={curr_right_grip:.4f}{dist_str}")

                    # Ensure proper shape for grippers
                    if isinstance(left_gripper_target, float):
                        left_gripper_target = torch.tensor([[left_gripper_target]], device=device)
                    elif isinstance(left_gripper_target, torch.Tensor):
                        if left_gripper_target.dim() == 0:
                            left_gripper_target = left_gripper_target.unsqueeze(0).unsqueeze(0)
                        elif left_gripper_target.dim() == 1:
                            left_gripper_target = left_gripper_target.unsqueeze(-1)

                    if isinstance(right_gripper_target, float):
                        right_gripper_target = torch.tensor([[right_gripper_target]], device=device)
                    elif isinstance(right_gripper_target, torch.Tensor):
                        if right_gripper_target.dim() == 0:
                            right_gripper_target = right_gripper_target.unsqueeze(0).unsqueeze(0)
                        elif right_gripper_target.dim() == 1:
                            right_gripper_target = right_gripper_target.unsqueeze(-1)

                    # Construct action tensor (14D) matching the environment's joint index order:
                    # [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)]
                    actions = torch.cat(
                        [
                            left_joint_targets,
                            right_joint_targets,
                            left_gripper_target,
                            right_gripper_target,
                        ],
                        dim=-1,
                    )

                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(actions)

                    # Handle terminated or truncated episodes
                    if terminated.any() or truncated.any():
                        debug_logger.log_event(f"Episode ended: terminated={terminated.any().item()}, truncated={truncated.any().item()}")
                        # Auto-reset is handled by env, but we need to reanchor
                        need_reanchor = True

                    # Record data if enabled
                    if recorder and is_recording:
                        env_obs = env.obs if hasattr(env, "obs") else obs.get("policy", obs)
                        recorder.record_frame(env_obs, actions, env_idx=0)

                    # Increment debug frame counter
                    debug_logger.next_frame()

                else:
                    # Just render when not active
                    env.sim.render()

                # Handle reset
                if should_reset:
                    obs, _ = env.reset()
                    teleop_interface.reset()
                    should_reset = False
                    need_reanchor = True
                    debug_logger.log_event("Environment reset complete")
                    print("[Teleop] Environment reset complete")

        except Exception as e:
            logger.error(f"Error during simulation: {e}")
            debug_logger.log_event(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            break

    # Cleanup
    if recorder and is_recording:
        recorder.stop_recording(save=True)
    debug_logger.close()
    env.close()
    print("[Teleop] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()
