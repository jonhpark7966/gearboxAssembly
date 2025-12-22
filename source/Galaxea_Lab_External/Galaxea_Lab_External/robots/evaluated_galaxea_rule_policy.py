# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extended rule policy class with step completion detection."""

import torch
from .galaxea_rule_policy import GalaxeaRulePolicy


class EvaluatedGalaxeaRulePolicy(GalaxeaRulePolicy):
    """ 
    GalaxeaRulePolicy에 단계별 완료 감지 기능 추가
    """

    def __init__(self, sim, scene, obj_dict):
        """
        Args:
            sim: Simulation context
            scene: Interactive scene
            obj_dict: Dictionary of objects
        """
        super().__init__(sim, scene, obj_dict)
        
        # Step completion tracking
        self.step_completed = False
        self.completed_step_name = None
        self.completed_gear_id = None
        self.prev_count = -1

    def get_action(self):
        """
        action 생성 및 단계별 완료 감지
        
        Returns:
            Tuple of (action, joint_ids) and sets step_completed flag
            when a step that requires evaluation is completed.
        """

        self.step_completed = False
        self.completed_step_name = None
        self.completed_gear_id = None
        
        action, joint_ids = super().get_action()
        
        # Detect step completion 
        if self.count != self.prev_count:
            self.prev_count = self.count
            
            # Check for completion of steps that require evaluation
            if self.count == self.count_step_2[-1]:
                self.step_completed = True
                self.completed_step_name = "mount_gear_1"
                self.completed_gear_id = 1
            elif self.count == self.count_step_4[-1]:
                self.step_completed = True
                self.completed_step_name = "mount_gear_2"
                self.completed_gear_id = 2
            elif self.count == self.count_step_7[-1]:
                self.step_completed = True
                self.completed_step_name = "mount_gear_3"
                self.completed_gear_id = 3
            elif self.count == self.count_step_9[-1]:
                self.step_completed = True
                self.completed_step_name = "mount_gear_4"
                self.completed_gear_id = 4
            elif self.count == self.count_step_12[-1]:
                self.step_completed = True
                self.completed_step_name = "mount_carrier_to_ring"
                self.completed_gear_id = None
            elif self.count == self.count_step_14[-1]:
                self.step_completed = True
                self.completed_step_name = "mount_reducer"
                self.completed_gear_id = None
        
        return action, joint_ids

    def restart_step(self, step_name: str):
        """
        특정 단계 재시작
 
        Args:
            step_name: Name of the step to restart
        """
        step_restart_map = {
            "mount_gear_1": self.count_step_2[0],
            "mount_gear_2": self.count_step_4[0],
            "mount_gear_3": self.count_step_7[0],
            "mount_gear_4": self.count_step_9[0],
            "mount_carrier_to_ring": self.count_step_12[0],
            "mount_reducer": self.count_step_14[0],
        }
        
        if step_name in step_restart_map:
            self.count = step_restart_map[step_name]
            self.prev_count = self.count - 1  
            
            self.current_target_position = None
            self.current_target_orientation = None
            self.current_target_joint_pos = None
            self.step_initial_joint_pos = None
            
            print(f"[RETRY] Restarted step '{step_name}' to count={self.count} (reset state variables)")
        else:
            print(f"[WARN] Unknown step name: {step_name}")

