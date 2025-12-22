# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extended environment class with step-by-step evaluation functions."""

from __future__ import annotations

import torch
import isaacsim.core.utils.torch as torch_utils

from .galaxea_lab_external_env import GalaxeaLabExternalEnv


class EvaluatedGalaxeaLabExternalEnv(GalaxeaLabExternalEnv):
    """Extended environment with step-by-step evaluation functions.
    
    This class extends GalaxeaLabExternalEnv to add evaluation functions
    for each step of the assembly process, without modifying the original class.
    """
    
    def step(self, action):
        terminated_before, truncated_before = self._get_dones()
        terminated_before_reset = terminated_before.clone()
        truncated_before_reset = truncated_before.clone()
        
        obs_buf, reward_buf, _, _, extras = super().step(action)
        
        return obs_buf, reward_buf, terminated_before_reset, truncated_before_reset, extras

    def evaluate_gear_mounted_to_pin(self, gear_id: int) -> bool:
        """특정 기어(1~4)가 핀에 올바르게 장착되었는지 평가
        
        Args:
            gear_id: The gear ID (1-4) corresponding to sun_planetary_gear_1-4
            
        Returns:
            True if the gear is properly mounted, False otherwise
        """
        pin_world_positions, pin_world_quats, gear_world_positions, gear_world_quats, _, _, ring_gear_world_pos, ring_gear_world_quat, _, _ = self.get_key_points()
        
        gear_idx = gear_id - 1  # gear_id 1-4 -> gear_idx 0-3
        if gear_idx < 0 or gear_idx >= len(gear_world_positions):
            print(f"[EVAL] Invalid gear_id: {gear_id}, must be 1-4")
            return False
        
        gear_world_pos = gear_world_positions[gear_idx]
        gear_world_quat = gear_world_quats[gear_idx]
        
        if gear_id == 4:
            # Get planetary carrier position to check if ring gear is mounted
            _, _, _, _, planetary_carrier_pos, planetary_carrier_quat, ring_gear_world_pos, ring_gear_world_quat, _, _ = self.get_key_points()
            
            carrier_to_ring_distance = torch.norm(planetary_carrier_pos[:, :2] - ring_gear_world_pos[:, :2])
            carrier_to_ring_height_diff = planetary_carrier_pos[:, 2] - ring_gear_world_pos[:, 2]
            carrier_to_ring_angle = torch.acos(torch.dot(planetary_carrier_quat.squeeze(0), ring_gear_world_quat.squeeze(0)))
            
            # ring gear가 carrier에 장착되었으면 ring gear center를 사용하고
            # 아니면 planetary carrier의 center를 사용 
            if carrier_to_ring_distance < 0.005 and carrier_to_ring_angle < 0.1 and carrier_to_ring_height_diff < 0.004:
                target_pos = ring_gear_world_pos
                target_quat = ring_gear_world_quat
                target_name = "ring gear center"
            else:
                target_pos = planetary_carrier_pos
                target_quat = planetary_carrier_quat
                target_name = "planetary carrier center (3 pins center)"
            
            distance = torch.norm(gear_world_pos[:, :2] - target_pos[:, :2])
            height_diff = gear_world_pos[:, 2] - target_pos[:, 2]
            

            if target_name == "planetary carrier center (3 pins center)":
                height_threshold = 0.01  # 10mm for step 9
            else:
                height_threshold = 0.004  # 4mm for step 12+ (same as evaluate_score)
            
            if distance < 0.005 and height_diff < height_threshold:
                print(f"[EVAL] Gear 4 (gear_idx={gear_idx}) is mounted to {target_name}")
                print(f"       distance={distance.item():.6f}, height_diff={height_diff.item():.6f}")
                return True
            else:
                print(f"[EVAL] Gear 4 (gear_idx={gear_idx}) is NOT mounted to {target_name}")
                print(f"[EVAL DEBUG] distance={distance.item():.6f} (threshold: <0.005), height_diff={height_diff.item():.6f} (threshold: <{height_threshold})")
                print(f"[EVAL DEBUG] Failed checks: distance<0.005={distance.item() < 0.005}, height_diff<{height_threshold}={height_diff.item() < height_threshold}")
                print(f"[EVAL DEBUG] Note: Angle check is skipped at step 9. It will be checked later in step 12+ using original evaluate_score logic.")
                return False
        
        # For gears 1-3: check if mounted to any pin 
        for pin_idx in range(len(pin_world_positions)):
            pin_world_pos = pin_world_positions[pin_idx]
            pin_world_quat = pin_world_quats[pin_idx]
            distance = torch.norm(gear_world_pos[:, :2] - pin_world_pos[:, :2])
            height_diff = gear_world_pos[:, 2] - pin_world_pos[:, 2]
            angle = torch.acos(torch.dot(gear_world_quat.squeeze(0), pin_world_quat.squeeze(0)))
            
            if distance < 0.002 and angle < 0.1 and height_diff < 0.012:
                print(f"[EVAL] Gear {gear_id} (gear_idx={gear_idx}) is mounted to pin {pin_idx}")
                print(f"       distance={distance.item():.6f}, angle={angle.item():.6f}, height_diff={height_diff.item():.6f}")
                return True
        
        print(f"[EVAL] Gear {gear_id} (gear_idx={gear_idx}) is NOT mounted to any pin")
        print(f"[EVAL DEBUG] All pin checks failed. Final values for all pins:")
        for pin_idx in range(len(pin_world_positions)):
            pin_world_pos = pin_world_positions[pin_idx]
            distance = torch.norm(gear_world_pos[:, :2] - pin_world_pos[:, :2])
            height_diff = gear_world_pos[:, 2] - pin_world_pos[:, 2]
            # EXACT same as line 347 in galaxea_lab_external_env.py
            angle = torch.acos(torch.dot(gear_world_quat.squeeze(0), pin_world_quats[pin_idx].squeeze(0)))
            print(f"[EVAL DEBUG]   Pin {pin_idx}: d={distance.item():.6f}, a={angle.item():.6f}, h={height_diff.item():.6f}")
            print(f"[EVAL DEBUG]     Check: d<0.002={distance.item() < 0.002}, a<0.1={angle.item() < 0.1}, h<0.012={height_diff.item() < 0.012}")
        return False

    def evaluate_carrier_mounted_to_ring(self) -> bool:
        """ Planetary carrier가 ring gear에 올바르게 장착되었는지 평가
        
        Returns:
            True if the carrier is properly mounted to the ring gear, False otherwise
        """
        _, _, _, _, planetary_carrier_pos, planetary_carrier_quat, ring_gear_world_pos, ring_gear_world_quat, _, _ = self.get_key_points()
        
        distance = torch.norm(planetary_carrier_pos[:, :2] - ring_gear_world_pos[:, :2])
        height_diff = planetary_carrier_pos[:, 2] - ring_gear_world_pos[:, 2]
        dot_product = torch.clamp(torch.dot(planetary_carrier_quat.squeeze(0), ring_gear_world_quat.squeeze(0)), -1.0, 1.0)
        angle = torch.acos(dot_product)
        
        success = distance < 0.005 and angle < 0.1 and height_diff < 0.004
        
        if success:
            print(f"[EVAL] Planetary carrier is mounted to ring gear")
            print(f"       distance={distance.item():.6f}, angle={angle.item():.6f}, height_diff={height_diff.item():.6f}")
        else:
            print(f"[EVAL] Planetary carrier is NOT mounted to ring gear")
            print(f"       distance={distance.item():.6f}, angle={angle.item():.6f}, height_diff={height_diff.item():.6f}")
        
        return success

    def evaluate_reducer_mounted(self) -> bool:
        """Planetary reducer가 올바르게 장착되었는지 평가
        
        Returns:
            True if the reducer is properly mounted to any gear, False otherwise
        """
        _, _, gear_world_positions, gear_world_quats, _, _, _, _, reducer_world_pos, reducer_world_quat = self.get_key_points()
        
        # Check all gears for reducer mounting
        for gear_idx in range(len(gear_world_positions)):
            gear_world_pos = gear_world_positions[gear_idx]
            gear_world_quat = gear_world_quats[gear_idx]
            distance = torch.norm(gear_world_pos[:, :2] - reducer_world_pos[:, :2])
            height_diff = gear_world_pos[:, 2] - reducer_world_pos[:, 2]
            dot_product = torch.clamp(torch.dot(gear_world_quat.squeeze(0), reducer_world_quat.squeeze(0)), -1.0, 1.0)
            angle = torch.acos(dot_product)
            
            if distance < 0.005 and angle < 0.1 and height_diff < 0.002:
                print(f"[EVAL] Reducer is mounted to gear {gear_idx + 1}")
                print(f"       distance={distance.item():.6f}, angle={angle.item():.6f}, height_diff={height_diff.item():.6f}")
                return True
        
        print(f"[EVAL] Reducer is NOT mounted to any gear")
        return False

