# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause
"""
VLA (Vision-Language-Action) Agent for Galaxea R1 Robot

ACT (Action Chunking Transformer) 모델을 사용하여 이미지와 현재 관절 상태로부터
action을 예측하는 에이전트입니다.

Usage:
    ./isaaclab.sh -p scripts/VLA_agent.py \
        --task Template-Galaxea-Lab-Agent-Direct-v0 \
        --checkpoint /path/to/policy_best.ckpt \
        --num_envs 1

Model Input:
    - qpos: 14D joint positions [left_arm(6), right_arm(6), left_grip(1), right_grip(1)]
    - images: (3, C, H, W) - head, left_hand, right_hand cameras

Model Output:
    - action: 14D joint position targets

Data Flow:
    1. env.step() → observation (RGB + joint positions)
    2. observation → ACT policy.predict()
    3. ACT output → env.step(action)
    4. 환경 내부: action → _apply_action() → set_joint_position_target()
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# ============================================================================
# CLI Arguments
# ============================================================================
parser = argparse.ArgumentParser(description="VLA agent for Galaxea R1")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--num_envs", type=int, default=1,
    help="Number of environments to simulate."
)
parser.add_argument(
    "--task", type=str, default="Template-Galaxea-Lab-Agent-Direct-v0",
    help="Name of the task."
)
parser.add_argument(
    "--checkpoint", type=str, default=None, required=True,
    help="Path to the VLA model checkpoint."
)
parser.add_argument(
    "--temporal_agg", action="store_true", default=True,
    help="Use temporal aggregation for smoother actions"
)
parser.add_argument(
    "--debug", action="store_true",
    help="Enable debug output"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

print(f"[VLAAgent] args_cli: {args_cli}")

# ============================================================================
# Launch Application
# ============================================================================
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# Post-Launch Imports
# ============================================================================
import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import Galaxea_Lab_External.tasks  # Register environment

from Galaxea_Lab_External.VLA.ACT.policy_wrapper import (
    ACTPolicyWrapper,
    DiffusionPolicyWrapper,
    BCPolicyWrapper,
    DataReplayPolicyWrapper
)

# Import base agent utilities
from base_agent import (
    TOTAL_ACTION_DIM,
    LEFT_GRIPPER_IDX,
    RIGHT_GRIPPER_IDX,
    parse_observation,
    print_action_debug,
    print_observation_debug,
    print_joint_index_debug,
)


# ============================================================================
# Main Function
# ============================================================================
def main():
    """VLA (ACT) Policy Agent 메인 루프

    Data Flow:
        1. 환경에서 observation 수집 (RGB images + joint positions)
        2. Observation을 policy 입력 형식으로 변환
           - qpos: (1, 14) float32
           - images: (1, 3, C, H, W) float32 normalized
        3. policy.predict(qpos, images) → action (1, 14)
        4. env.step(action) 호출
        5. 환경 내부: action → _apply_action() → set_joint_position_target()
    """
    # ========================================
    # Step 1: Environment Configuration
    # ========================================
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    # ========================================
    # Step 2: Load ACT Policy
    # ========================================
    print(f"[VLAAgent] Loading checkpoint: {args_cli.checkpoint}")
    policy = ACTPolicyWrapper(
        args_cli.checkpoint,
        temporal_agg=args_cli.temporal_agg
    )
    print("[VLAAgent] Policy loaded successfully")

    # ========================================
    # Step 3: Create Environment
    # ========================================
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_unwrapped = env.unwrapped

    print(f"[VLAAgent] Environment type: {type(env)}")
    print(f"[VLAAgent] Observation space: {env.observation_space}")
    print(f"[VLAAgent] Action space: {env.action_space}")

    # Debug: Joint index 확인
    if args_cli.debug:
        print_joint_index_debug(env_unwrapped)

    # ========================================
    # Step 4: Reset Environment
    # ========================================
    obs, _ = env.reset()
    print("[VLAAgent] Environment reset complete")

    # ========================================
    # Step 5: Main Simulation Loop
    # ========================================
    step_count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # ----------------------------------------
            # Step 5a: Parse Observation
            # ----------------------------------------
            obs_parsed = parse_observation(obs)

            # ----------------------------------------
            # Step 5b: Prepare Policy Inputs
            # ----------------------------------------
            # qpos: (num_envs, 14) - current joint positions
            qpos = obs_parsed.get_qpos()

            # images: (num_envs, 3, C, H, W) - 3 camera views
            # Note: Policy expects (B, N_cams, C, H, W) float32
            images = obs_parsed.get_images_for_vla()

            if args_cli.debug and step_count < 3:
                print(f"\n[VLAAgent] Step {step_count}")
                print(f"  qpos shape: {qpos.shape}")
                print(f"  images shape: {images.shape}")
                print(f"  qpos values: {qpos[0].cpu().numpy()}")

            # ----------------------------------------
            # Step 5c: Policy Inference
            # ----------------------------------------
            # ACT policy predicts 14D action
            action = policy.predict(qpos, images)

            if args_cli.debug and step_count < 3:
                print_action_debug(action, prefix="  Predicted ")

            # ----------------------------------------
            # Step 5d: Apply Action to Environment
            # ----------------------------------------
            # 이 호출이 핵심!
            # env.step() → env._pre_physics_step() → env._apply_action()
            # → robot.set_joint_position_target(action, _joint_idx)
            obs, reward, terminated, truncated, info = env.step(action)

            # ----------------------------------------
            # Step 5e: Handle Episode End
            # ----------------------------------------
            if terminated.any() or truncated.any():
                print(f"[VLAAgent] Episode ended at step {step_count}")
                obs, _ = env.reset()
                step_count = 0
                continue

            step_count += 1

            # Periodic status
            if step_count % 100 == 0:
                print(f"[VLAAgent] Step {step_count}, Reward: {reward}")

    # ========================================
    # Step 6: Cleanup
    # ========================================
    env.close()
    print("[VLAAgent] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()
