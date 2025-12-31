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
from collections.abc import Callable

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
    action="store_true",
    help="Disable object randomization on reset",
)

# Add AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Configure AppLauncher with XR if using hand tracking
app_launcher_args = vars(args_cli)
if "handtracking" in args_cli.teleop_device.lower():
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
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

# Import the task module to register the environment
import Galaxea_Lab_External.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Import data recorder
from teleop_data_recorder import TeleopDataRecorder

logger = logging.getLogger(__name__)


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

    # Set IK command
    ik_command = torch.cat([target_pos, target_quat], dim=-1)
    controller.set_command(ik_command)

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

    # Get current joint positions
    joint_pos = robot.data.joint_pos[:, arm_entity_cfg.joint_ids]

    # Compute IK
    joint_pos_des = controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    return joint_pos_des


def map_grip_to_joint(grip: float, open_pos: float, close_pos: float) -> float:
    """Map grip value (0-1) to gripper joint position.

    Args:
        grip: Grip value from 0 (open) to 1 (closed).
        open_pos: Gripper open joint position.
        close_pos: Gripper closed joint position.

    Returns:
        Gripper joint position.
    """
    return open_pos + grip * (close_pos - open_pos)


def main() -> None:
    """Main teleoperation loop."""

    # Parse environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task

    # Disable timeout for teleoperation
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    # Configure for XR - don't remove camera configs since we need them for recording
    if args_cli.xr:
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # Create environment
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    device = env.device

    # Setup IK controllers for both arms
    left_ik, left_arm_cfg, left_body_ids, left_joint_ids = setup_ik_controller(env, "left", device)
    right_ik, right_arm_cfg, right_body_ids, right_joint_ids = setup_ik_controller(env, "right", device)

    # Get gripper joint indices
    robot = env.robot
    left_gripper_idx, _ = robot.find_joints("left_gripper_axis1")
    right_gripper_idx, _ = robot.find_joints("right_gripper_axis1")

    # Teleoperation state
    teleoperation_active = False if args_cli.xr else True
    should_reset = False
    is_recording = False

    # Reanchoring offsets
    left_pos_offset = torch.zeros(3, device=device)
    right_pos_offset = torch.zeros(3, device=device)
    need_reanchor = True

    # Data recorder
    recorder = TeleopDataRecorder(
        save_dir=args_cli.record_dir,
        record_rgb=True,
        record_depth=False,
    ) if args_cli.record else None

    # Callback handlers
    def start_teleoperation() -> None:
        nonlocal teleoperation_active, is_recording, need_reanchor
        teleoperation_active = True
        need_reanchor = True  # Reanchor on next frame
        if recorder and args_cli.record:
            recorder.start_recording()
            is_recording = True
        print("[Teleop] Activated - START gesture detected")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active, is_recording
        teleoperation_active = False
        if recorder and is_recording:
            recorder.stop_recording(save=True)
            is_recording = False
        print("[Teleop] Deactivated - STOP gesture detected")

    def reset_environment() -> None:
        nonlocal should_reset, need_reanchor
        should_reset = True
        need_reanchor = True
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

    # Reset environment
    obs, _ = env.reset()
    teleop_interface.reset()

    print("[Teleop] Environment ready. Use START gesture to begin teleoperation.")
    if args_cli.xr:
        print("[Teleop] XR mode: Pinch thumb and index finger together, then release to START")

    # Main simulation loop
    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                # Get teleop command
                # For OpenXR with bimanual retargeters, output is:
                # [left_pos(3), left_quat(4), left_grip(1), right_pos(3), right_quat(4), right_grip(1)]
                raw_action = teleop_interface.advance()

                if teleoperation_active and raw_action is not None:
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
                        robot = env.robot
                        left_ee_pose = robot.data.body_state_w[:, left_body_ids[0], 0:7]
                        right_ee_pose = robot.data.body_state_w[:, right_body_ids[0], 0:7]
                        left_pos = left_ee_pose[:, 0:3]
                        left_quat = left_ee_pose[:, 3:7]
                        right_pos = right_ee_pose[:, 0:3]
                        right_quat = right_ee_pose[:, 3:7]
                        left_grip = torch.zeros(env.num_envs, 1, device=device)
                        right_grip = torch.zeros(env.num_envs, 1, device=device)

                    # Ensure batch dimension
                    if left_pos.dim() == 1:
                        left_pos = left_pos.unsqueeze(0)
                        left_quat = left_quat.unsqueeze(0)
                        left_grip = left_grip.unsqueeze(0)
                        right_pos = right_pos.unsqueeze(0)
                        right_quat = right_quat.unsqueeze(0)
                        right_grip = right_grip.unsqueeze(0)

                    # Reanchor: calculate offset to match current robot pose
                    if need_reanchor:
                        robot = env.robot
                        left_ee_pose = robot.data.body_state_w[:, left_body_ids[0], 0:7]
                        right_ee_pose = robot.data.body_state_w[:, right_body_ids[0], 0:7]
                        left_pos_offset = left_ee_pose[:, 0:3] - left_pos
                        right_pos_offset = right_ee_pose[:, 0:3] - right_pos
                        need_reanchor = False
                        print(f"[Teleop] Reanchored - Left offset: {left_pos_offset[0].cpu().numpy()}")
                        print(f"[Teleop] Reanchored - Right offset: {right_pos_offset[0].cpu().numpy()}")

                    # Apply position offsets
                    left_pos_target = left_pos + left_pos_offset
                    right_pos_target = right_pos + right_pos_offset

                    # Compute IK for both arms
                    left_joint_targets = compute_ik(
                        env, left_ik, left_arm_cfg, left_body_ids,
                        left_pos_target, left_quat
                    )
                    right_joint_targets = compute_ik(
                        env, right_ik, right_arm_cfg, right_body_ids,
                        right_pos_target, right_quat
                    )

                    # Map grip to gripper joint position
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

                    # Ensure proper shape for grippers
                    if isinstance(left_gripper_target, float):
                        left_gripper_target = torch.tensor([[left_gripper_target]], device=device)
                    elif left_gripper_target.dim() == 0:
                        left_gripper_target = left_gripper_target.unsqueeze(0).unsqueeze(0)
                    elif left_gripper_target.dim() == 1:
                        left_gripper_target = left_gripper_target.unsqueeze(-1)

                    if isinstance(right_gripper_target, float):
                        right_gripper_target = torch.tensor([[right_gripper_target]], device=device)
                    elif right_gripper_target.dim() == 0:
                        right_gripper_target = right_gripper_target.unsqueeze(0).unsqueeze(0)
                    elif right_gripper_target.dim() == 1:
                        right_gripper_target = right_gripper_target.unsqueeze(-1)

                    # Construct action tensor (14D):
                    # [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
                    actions = torch.cat([
                        left_joint_targets,
                        left_gripper_target,
                        right_joint_targets,
                        right_gripper_target,
                    ], dim=-1)

                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(actions)

                    # Record data if enabled
                    if recorder and is_recording:
                        env_obs = env.obs if hasattr(env, "obs") else obs.get("policy", obs)
                        recorder.record_frame(env_obs, actions, env_idx=0)

                else:
                    # Just render when not active
                    env.sim.render()

                # Handle reset
                if should_reset:
                    obs, _ = env.reset()
                    teleop_interface.reset()
                    should_reset = False
                    need_reanchor = True
                    print("[Teleop] Environment reset complete")

        except Exception as e:
            logger.error(f"Error during simulation: {e}")
            import traceback
            traceback.print_exc()
            break

    # Cleanup
    if recorder and is_recording:
        recorder.stop_recording(save=True)
    env.close()
    print("[Teleop] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()
