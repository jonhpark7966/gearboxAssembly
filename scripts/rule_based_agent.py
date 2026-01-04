# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause
"""
Rule-Based Random Agent for Galaxea R1 Robot

간단한 랜덤 액션 생성 에이전트입니다.
환경 테스트 및 기본 동작 확인용으로 사용됩니다.

Usage:
    ./isaaclab.sh -p scripts/rule_based_agent.py \
        --task Template-Galaxea-Lab-Agent-Direct-v0 \
        --num_envs 1

Action Format (14D):
    [0:6]   left_arm_joint[1-6]     라디안
    [6:12]  right_arm_joint[1-6]    라디안
    [12]    left_gripper_axis1      0.0 ~ 0.04 meters
    [13]    right_gripper_axis1     0.0 ~ 0.04 meters

Data Flow:
    1. Random action 생성 (14D)
    2. env.step(action) 호출
    3. 환경 내부에서 _apply_action() 실행
    4. robot.set_joint_position_target(action, _joint_idx)
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

# ============================================================================
# CLI Arguments
# ============================================================================
parser = argparse.ArgumentParser(description="Random agent for Galaxea R1")
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
    "--debug", action="store_true",
    help="Enable debug output"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

print(f"[RuleAgent] args_cli: {args_cli}")

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

# Import base agent utilities
from base_agent import (
    TOTAL_ACTION_DIM,
    LEFT_GRIPPER_IDX,
    RIGHT_GRIPPER_IDX,
    GRIPPER_OPEN,
    GRIPPER_CLOSE,
    ActionComponents,
    parse_observation,
    print_action_debug,
    print_joint_index_debug,
)


# ============================================================================
# Main Function
# ============================================================================
def main():
    """Random action agent 메인 루프

    Data Flow:
        1. 환경 생성 (gym.make)
        2. 초기 reset
        3. 매 스텝마다:
           a. 랜덤 action 생성
           b. env.step(action) 호출
           c. 환경 내부: action → _apply_action() → set_joint_position_target()
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
    # Step 2: Create Environment
    # ========================================
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_unwrapped = env.unwrapped

    print(f"[RuleAgent] Environment type: {type(env)}")
    print(f"[RuleAgent] Observation space: {env.observation_space}")
    print(f"[RuleAgent] Action space: {env.action_space}")
    print(f"[RuleAgent] Action space shape: {env.action_space.shape}")

    # Debug: Joint index 확인
    if args_cli.debug:
        print_joint_index_debug(env_unwrapped)

    # ========================================
    # Step 3: Reset Environment
    # ========================================
    obs, _ = env.reset()
    print("[RuleAgent] Environment reset complete")

    # ========================================
    # Step 4: Main Simulation Loop
    # ========================================
    step_count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # ----------------------------------------
            # Step 4a: Generate Random Action
            # ----------------------------------------
            # 랜덤 action 생성: 범위 [-1, 1]
            # 주의: 그리퍼는 실제로 [0.0, 0.04] 범위여야 함
            actions = 2 * torch.rand(
                env.action_space.shape,
                device=env_unwrapped.device
            ) - 1

            # 그리퍼 범위 조정: [-1,1] → [0.0, 0.04]
            # (랜덤 테스트이므로 전체 범위 사용)
            actions[:, LEFT_GRIPPER_IDX] = (
                torch.rand(env_unwrapped.num_envs, device=env_unwrapped.device)
                * (GRIPPER_OPEN - GRIPPER_CLOSE) + GRIPPER_CLOSE
            )
            actions[:, RIGHT_GRIPPER_IDX] = (
                torch.rand(env_unwrapped.num_envs, device=env_unwrapped.device)
                * (GRIPPER_OPEN - GRIPPER_CLOSE) + GRIPPER_CLOSE
            )

            # Debug output
            if args_cli.debug and step_count < 5:
                print(f"\n[RuleAgent] Step {step_count}")
                print_action_debug(actions, prefix="  ")

            # ----------------------------------------
            # Step 4b: Apply Action to Environment
            # ----------------------------------------
            # 이 호출이 핵심!
            # env.step() → env._pre_physics_step() → env._apply_action()
            # → robot.set_joint_position_target(action, _joint_idx)
            obs, reward, terminated, truncated, info = env.step(actions)

            # ----------------------------------------
            # Step 4c: Log Results
            # ----------------------------------------
            if args_cli.debug and step_count < 5:
                obs_parsed = parse_observation(obs)
                print(f"  Gripper positions after step:")
                print(f"    Left:  {obs_parsed.left_gripper_pos[0].item():.4f}")
                print(f"    Right: {obs_parsed.right_gripper_pos[0].item():.4f}")

            step_count += 1

            # Optional: Print periodic status
            if step_count % 100 == 0:
                print(f"[RuleAgent] Step {step_count}, Reward: {reward}")

    # ========================================
    # Step 5: Cleanup
    # ========================================
    env.close()
    print("[RuleAgent] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()
