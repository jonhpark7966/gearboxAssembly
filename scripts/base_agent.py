# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause
"""
Base Agent Module for Galaxea R1 Robot

이 모듈은 세 가지 에이전트(teleop, rule_based, VLA)에서 공통으로 사용하는
코드를 제공합니다.

핵심 컴포넌트:
1. Environment 생성 및 설정
2. Action 텐서 구성
3. Observation 파싱
4. 디버깅 유틸리티

Action Tensor Format (14D):
    [0:6]   left_arm_joint[1-6]     라디안
    [6:12]  right_arm_joint[1-6]    라디안
    [12]    left_gripper_axis1      0.0 (close) ~ 0.04 (open) meters
    [13]    right_gripper_axis1     0.0 (close) ~ 0.04 (open) meters

Observation Dict Keys:
    - head_rgb, left_hand_rgb, right_hand_rgb: (H, W, 3) uint8
    - head_depth, left_hand_depth, right_hand_depth: (H, W) float32
    - left_arm_joint_pos, right_arm_joint_pos: (6,) float32
    - left_gripper_joint_pos, right_gripper_joint_pos: scalar float32
"""

from __future__ import annotations

import argparse
import torch
import gymnasium as gym
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# Constants
# ============================================================================

# Action dimensions
LEFT_ARM_DIM = 6
RIGHT_ARM_DIM = 6
LEFT_GRIPPER_DIM = 1
RIGHT_GRIPPER_DIM = 1
TOTAL_ACTION_DIM = LEFT_ARM_DIM + RIGHT_ARM_DIM + LEFT_GRIPPER_DIM + RIGHT_GRIPPER_DIM  # 14

# Action indices
LEFT_ARM_START = 0
LEFT_ARM_END = 6
RIGHT_ARM_START = 6
RIGHT_ARM_END = 12
LEFT_GRIPPER_IDX = 12
RIGHT_GRIPPER_IDX = 13

# Gripper range (meters)
GRIPPER_OPEN = 0.04
GRIPPER_CLOSE = 0.0


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GripperConfig:
    """그리퍼 설정"""
    open_pos: float = GRIPPER_OPEN
    close_pos: float = GRIPPER_CLOSE
    pinch_open_dist: float = 0.06   # 손가락 열린 거리 (m)
    pinch_close_dist: float = 0.02  # 손가락 닫힌 거리 (m)


@dataclass
class ActionComponents:
    """분해된 Action 컴포넌트"""
    left_arm: torch.Tensor      # (num_envs, 6)
    right_arm: torch.Tensor     # (num_envs, 6)
    left_gripper: torch.Tensor  # (num_envs, 1)
    right_gripper: torch.Tensor # (num_envs, 1)

    def to_tensor(self) -> torch.Tensor:
        """14D action tensor로 결합"""
        return torch.cat([
            self.left_arm,
            self.right_arm,
            self.left_gripper,
            self.right_gripper,
        ], dim=-1)

    @classmethod
    def from_tensor(cls, action: torch.Tensor) -> "ActionComponents":
        """14D action tensor에서 분해"""
        return cls(
            left_arm=action[..., LEFT_ARM_START:LEFT_ARM_END],
            right_arm=action[..., RIGHT_ARM_START:RIGHT_ARM_END],
            left_gripper=action[..., LEFT_GRIPPER_IDX:LEFT_GRIPPER_IDX+1],
            right_gripper=action[..., RIGHT_GRIPPER_IDX:RIGHT_GRIPPER_IDX+1],
        )


@dataclass
class ObservationComponents:
    """파싱된 Observation 컴포넌트"""
    # Images
    head_rgb: torch.Tensor          # (num_envs, H, W, 3)
    left_hand_rgb: torch.Tensor     # (num_envs, H, W, 3)
    right_hand_rgb: torch.Tensor    # (num_envs, H, W, 3)

    # Joint positions
    left_arm_joint_pos: torch.Tensor    # (num_envs, 6)
    right_arm_joint_pos: torch.Tensor   # (num_envs, 6)
    left_gripper_pos: torch.Tensor      # (num_envs,)
    right_gripper_pos: torch.Tensor     # (num_envs,)

    # Joint velocities
    left_arm_joint_vel: torch.Tensor    # (num_envs, 6)
    right_arm_joint_vel: torch.Tensor   # (num_envs, 6)
    left_gripper_vel: torch.Tensor      # (num_envs,)
    right_gripper_vel: torch.Tensor     # (num_envs,)

    # Optional depth
    head_depth: Optional[torch.Tensor] = None
    left_hand_depth: Optional[torch.Tensor] = None
    right_hand_depth: Optional[torch.Tensor] = None

    def get_qpos(self) -> torch.Tensor:
        """14D qpos tensor 반환 (for VLA policy)"""
        return torch.cat([
            self.left_arm_joint_pos,
            self.right_arm_joint_pos,
            self.left_gripper_pos.unsqueeze(-1),
            self.right_gripper_pos.unsqueeze(-1),
        ], dim=-1)

    def get_images_for_vla(self) -> torch.Tensor:
        """VLA policy용 이미지 텐서 (num_envs, 3, C, H, W)"""
        # Permute from (N, H, W, C) to (N, C, H, W)
        head = self.head_rgb.permute(0, 3, 1, 2).unsqueeze(1)
        left = self.left_hand_rgb.permute(0, 3, 1, 2).unsqueeze(1)
        right = self.right_hand_rgb.permute(0, 3, 1, 2).unsqueeze(1)
        return torch.cat([head, left, right], dim=1).float()


# ============================================================================
# Environment Utilities
# ============================================================================

def create_base_parser(description: str) -> argparse.ArgumentParser:
    """공통 CLI 인자를 포함한 parser 생성"""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--num_envs", type=int, default=1,
        help="Number of environments"
    )
    parser.add_argument(
        "--task", type=str, default="Template-Galaxea-Lab-Agent-Direct-v0",
        help="Task name"
    )
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False,
        help="Disable fabric and use USD I/O operations"
    )
    parser.add_argument(
        "--no_randomize_objects", action="store_false", dest="randomize_objects",
        default=True, help="Disable object randomization on reset"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug output"
    )

    AppLauncher.add_app_launcher_args(parser)
    return parser


def create_environment(task: str, device: str, num_envs: int,
                       use_fabric: bool = True,
                       randomize_objects: bool = True):
    """환경 생성 및 설정

    Args:
        task: 태스크 이름
        device: torch device
        num_envs: 환경 개수
        use_fabric: Fabric 사용 여부
        randomize_objects: 오브젝트 랜덤화 여부

    Returns:
        env: Gym 환경
        env_cfg: 환경 설정
    """
    # Import here to avoid circular imports
    from isaaclab_tasks.utils import parse_env_cfg
    import Galaxea_Lab_External.tasks  # Register environment

    env_cfg = parse_env_cfg(
        task, device=device, num_envs=num_envs, use_fabric=use_fabric
    )

    # Object randomization 설정
    if hasattr(env_cfg, "randomize_objects"):
        env_cfg.randomize_objects = randomize_objects

    # 환경 생성
    env = gym.make(task, cfg=env_cfg)

    return env, env_cfg


def parse_observation(obs_dict: Dict) -> ObservationComponents:
    """환경 observation을 파싱하여 컴포넌트로 분해

    Args:
        obs_dict: env.step()에서 반환된 observation dict

    Returns:
        ObservationComponents 인스턴스
    """
    policy_obs = obs_dict.get("policy", obs_dict)

    return ObservationComponents(
        head_rgb=policy_obs["head_rgb"],
        left_hand_rgb=policy_obs["left_hand_rgb"],
        right_hand_rgb=policy_obs["right_hand_rgb"],
        left_arm_joint_pos=policy_obs["left_arm_joint_pos"],
        right_arm_joint_pos=policy_obs["right_arm_joint_pos"],
        left_gripper_pos=policy_obs["left_gripper_joint_pos"],
        right_gripper_pos=policy_obs["right_gripper_joint_pos"],
        left_arm_joint_vel=policy_obs["left_arm_joint_vel"],
        right_arm_joint_vel=policy_obs["right_arm_joint_vel"],
        left_gripper_vel=policy_obs["left_gripper_joint_vel"],
        right_gripper_vel=policy_obs["right_gripper_joint_vel"],
        head_depth=policy_obs.get("head_depth"),
        left_hand_depth=policy_obs.get("left_hand_depth"),
        right_hand_depth=policy_obs.get("right_hand_depth"),
    )


# ============================================================================
# Gripper Utilities
# ============================================================================

def grip_to_joint(grip: float, config: GripperConfig = GripperConfig()) -> float:
    """Grip value를 joint position으로 변환

    GripperRetargeter 출력:
        -1.0 = 닫힘 (pinch)
        +1.0 = 열림 (fingers apart)

    Gripper joint 범위:
        0.0 = 닫힘
        0.04 = 열림

    Args:
        grip: Grip value [-1, +1]
        config: Gripper 설정

    Returns:
        Joint position [0.0, 0.04]
    """
    # [-1, +1] → [0, 1]
    normalized = (grip + 1.0) / 2.0
    # [0, 1] → [close, open]
    return config.close_pos + normalized * (config.open_pos - config.close_pos)


def joint_to_grip(joint_pos: float, config: GripperConfig = GripperConfig()) -> float:
    """Joint position을 grip value로 변환 (역변환)

    Args:
        joint_pos: Joint position [0.0, 0.04]
        config: Gripper 설정

    Returns:
        Grip value [-1, +1]
    """
    # [close, open] → [0, 1]
    normalized = (joint_pos - config.close_pos) / (config.open_pos - config.close_pos)
    normalized = max(0.0, min(1.0, normalized))  # Clamp
    # [0, 1] → [-1, +1]
    return -1.0 + 2.0 * normalized


def finger_distance_to_grip(thumb_pos, index_pos,
                            config: GripperConfig = GripperConfig()) -> float:
    """손가락 거리를 grip value로 변환 (smooth, no hysteresis)

    Args:
        thumb_pos: Thumb tip 3D position (np.ndarray or list)
        index_pos: Index tip 3D position (np.ndarray or list)
        config: Gripper 설정

    Returns:
        Grip value [-1, +1]
    """
    import numpy as np

    distance = np.linalg.norm(np.array(thumb_pos) - np.array(index_pos))

    if distance <= config.pinch_close_dist:
        return -1.0
    elif distance >= config.pinch_open_dist:
        return 1.0
    else:
        normalized = (distance - config.pinch_close_dist) / \
                     (config.pinch_open_dist - config.pinch_close_dist)
        return -1.0 + 2.0 * normalized


# ============================================================================
# Action Construction
# ============================================================================

def construct_action(
    left_arm: torch.Tensor,
    right_arm: torch.Tensor,
    left_gripper: torch.Tensor,
    right_gripper: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """컴포넌트로부터 14D action tensor 생성

    Args:
        left_arm: Left arm joint positions (6,) or (num_envs, 6)
        right_arm: Right arm joint positions (6,) or (num_envs, 6)
        left_gripper: Left gripper position (1,) or (num_envs, 1)
        right_gripper: Right gripper position (1,) or (num_envs, 1)
        device: Torch device

    Returns:
        Action tensor (num_envs, 14)
    """
    # Ensure proper dimensions
    if left_arm.dim() == 1:
        left_arm = left_arm.unsqueeze(0)
    if right_arm.dim() == 1:
        right_arm = right_arm.unsqueeze(0)
    if left_gripper.dim() == 0:
        left_gripper = left_gripper.unsqueeze(0).unsqueeze(0)
    elif left_gripper.dim() == 1:
        left_gripper = left_gripper.unsqueeze(-1)
    if right_gripper.dim() == 0:
        right_gripper = right_gripper.unsqueeze(0).unsqueeze(0)
    elif right_gripper.dim() == 1:
        right_gripper = right_gripper.unsqueeze(-1)

    return torch.cat([
        left_arm.to(device),
        right_arm.to(device),
        left_gripper.to(device),
        right_gripper.to(device),
    ], dim=-1)


# ============================================================================
# Debug Utilities
# ============================================================================

def print_action_debug(action: torch.Tensor, prefix: str = ""):
    """Action tensor 디버깅 출력

    Args:
        action: 14D action tensor
        prefix: 출력 앞에 붙일 문자열
    """
    if action.dim() == 1:
        action = action.unsqueeze(0)

    print(f"{prefix}Action Debug:")
    print(f"  Shape: {action.shape}")
    print(f"  Left Arm [0:6]:   {action[0, :6].cpu().numpy()}")
    print(f"  Right Arm [6:12]: {action[0, 6:12].cpu().numpy()}")
    print(f"  Left Gripper [12]:  {action[0, 12].item():.4f}")
    print(f"  Right Gripper [13]: {action[0, 13].item():.4f}")


def print_observation_debug(obs: ObservationComponents, prefix: str = ""):
    """Observation 디버깅 출력"""
    print(f"{prefix}Observation Debug:")
    print(f"  Head RGB shape: {obs.head_rgb.shape}")
    print(f"  Left Arm Pos: {obs.left_arm_joint_pos[0].cpu().numpy()}")
    print(f"  Right Arm Pos: {obs.right_arm_joint_pos[0].cpu().numpy()}")
    print(f"  Left Gripper: {obs.left_gripper_pos[0].item():.4f}")
    print(f"  Right Gripper: {obs.right_gripper_pos[0].item():.4f}")


def print_joint_index_debug(env):
    """환경의 joint index 디버깅 출력

    핵심: action tensor의 순서와 _joint_idx의 순서가 일치하는지 확인

    Args:
        env: 환경 (unwrapped)
    """
    print("\n=== Joint Index Debug ===")
    print(f"_left_arm_joint_idx: {env._left_arm_joint_idx}")
    print(f"_right_arm_joint_idx: {env._right_arm_joint_idx}")
    print(f"_left_gripper_dof_idx: {env._left_gripper_dof_idx}")
    print(f"_right_gripper_dof_idx: {env._right_gripper_dof_idx}")
    print(f"_joint_idx (combined): {env._joint_idx}")
    print(f"len(_joint_idx): {len(env._joint_idx)}")
    print(f"Expected: 14 (6 + 6 + 1 + 1)")

    # Action → Joint 매핑 출력
    print("\nExpected Action → Joint Mapping:")
    print("  action[0:6]  → left_arm_joint[1-6]")
    print("  action[6:12] → right_arm_joint[1-6]")
    print("  action[12]   → left_gripper_axis1")
    print("  action[13]   → right_gripper_axis1")

    # 실제 매핑 확인
    print("\nActual _joint_idx contents:")
    robot = env.robot
    joint_names = robot.joint_names
    for i, idx in enumerate(env._joint_idx):
        if idx < len(joint_names):
            print(f"  _joint_idx[{i}] = {idx} → {joint_names[idx]}")
        else:
            print(f"  _joint_idx[{i}] = {idx} → INDEX OUT OF RANGE!")
    print("=" * 50)
