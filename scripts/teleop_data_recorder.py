# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause
"""
HDF5 Data Recorder for Teleoperation

Records observations and actions during teleoperation sessions in HDF5 format
compatible with VLA/ACT training pipelines.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch

if TYPE_CHECKING:
    from typing import Optional


class TeleopDataRecorder:
    """Records teleoperation data to HDF5 files.

    HDF5 Structure:
        episode_N.h5
        ├── observations/
        │   ├── head_rgb              (T, H, W, 3) uint8
        │   ├── left_hand_rgb         (T, H, W, 3) uint8
        │   ├── right_hand_rgb        (T, H, W, 3) uint8
        │   ├── head_depth            (T, H, W) float32
        │   ├── left_hand_depth       (T, H, W) float32
        │   ├── right_hand_depth      (T, H, W) float32
        │   ├── left_arm_joint_pos    (T, 6) float32
        │   ├── right_arm_joint_pos   (T, 6) float32
        │   ├── left_gripper_pos      (T, 1) float32
        │   └── right_gripper_pos     (T, 1) float32
        ├── actions/
        │   ├── left_arm_action       (T, 6) float32
        │   ├── right_arm_action      (T, 6) float32
        │   ├── left_gripper          (T, 1) float32
        │   └── right_gripper         (T, 1) float32
        └── attrs: {sim: True, timestamp: ...}
    """

    def __init__(
        self,
        save_dir: str,
        record_rgb: bool = True,
        record_depth: bool = False,
    ):
        """Initialize the data recorder.

        Args:
            save_dir: Directory to save HDF5 files.
            record_rgb: Whether to record RGB images.
            record_depth: Whether to record depth images.
        """
        self.save_dir = save_dir
        self.record_rgb = record_rgb
        self.record_depth = record_depth

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Episode counter
        self.episode_idx = self._get_next_episode_idx()

        # Data buffers for current episode
        self._reset_buffers()

        # Recording state
        self.is_recording = False
        self.start_time: Optional[float] = None

    def _get_next_episode_idx(self) -> int:
        """Find the next available episode index."""
        existing = [f for f in os.listdir(self.save_dir) if f.startswith("episode_") and f.endswith(".h5")]
        if not existing:
            return 0
        indices = []
        for f in existing:
            try:
                idx = int(f.replace("episode_", "").replace(".h5", ""))
                indices.append(idx)
            except ValueError:
                pass
        return max(indices) + 1 if indices else 0

    def _reset_buffers(self) -> None:
        """Reset data buffers for a new episode."""
        self.data = {
            # Observations
            "/observations/head_rgb": [],
            "/observations/left_hand_rgb": [],
            "/observations/right_hand_rgb": [],
            "/observations/head_depth": [],
            "/observations/left_hand_depth": [],
            "/observations/right_hand_depth": [],
            "/observations/left_arm_joint_pos": [],
            "/observations/right_arm_joint_pos": [],
            "/observations/left_gripper_pos": [],
            "/observations/right_gripper_pos": [],
            # Actions
            "/actions/left_arm_action": [],
            "/actions/right_arm_action": [],
            "/actions/left_gripper": [],
            "/actions/right_gripper": [],
            # Timestamps
            "/timestamps": [],
        }

    def start_recording(self) -> None:
        """Start recording a new episode."""
        self._reset_buffers()
        self.is_recording = True
        self.start_time = time.time()
        print(f"[Recorder] Started recording episode {self.episode_idx}")

    def stop_recording(self, save: bool = True) -> Optional[str]:
        """Stop recording and optionally save the episode.

        Args:
            save: Whether to save the episode to disk.

        Returns:
            Path to saved file if saved, None otherwise.
        """
        if not self.is_recording:
            return None

        self.is_recording = False
        duration = time.time() - self.start_time if self.start_time else 0
        num_frames = len(self.data["/timestamps"])

        print(f"[Recorder] Stopped recording: {num_frames} frames, {duration:.1f}s")

        if save and num_frames > 0:
            filepath = self.save_episode()
            return filepath
        return None

    def record_frame(
        self,
        obs: dict,
        actions: torch.Tensor,
        env_idx: int = 0,
    ) -> None:
        """Record a single frame of observations and actions.

        Args:
            obs: Observation dictionary from environment.
            actions: Action tensor (14D: 6 left arm + 6 right arm + 1 left gripper + 1 right gripper).
            env_idx: Environment index for multi-env setups.
        """
        if not self.is_recording:
            return

        # Record timestamp
        self.data["/timestamps"].append(time.time())

        # Helper to convert tensor to numpy
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return np.array(x)

        # Record RGB observations
        if self.record_rgb:
            if "head_rgb" in obs:
                img = to_numpy(obs["head_rgb"])
                # Handle both (H, W, C) and (batch, H, W, C) formats
                if img.ndim == 4:
                    img = img[env_idx]
                if img.shape[-1] == 4:  # RGBA -> RGB
                    img = img[..., :3]
                self.data["/observations/head_rgb"].append(img.astype(np.uint8))

            if "left_hand_rgb" in obs:
                img = to_numpy(obs["left_hand_rgb"])
                if img.ndim == 4:
                    img = img[env_idx]
                if img.shape[-1] == 4:
                    img = img[..., :3]
                self.data["/observations/left_hand_rgb"].append(img.astype(np.uint8))

            if "right_hand_rgb" in obs:
                img = to_numpy(obs["right_hand_rgb"])
                if img.ndim == 4:
                    img = img[env_idx]
                if img.shape[-1] == 4:
                    img = img[..., :3]
                self.data["/observations/right_hand_rgb"].append(img.astype(np.uint8))

        # Record depth observations
        if self.record_depth:
            if "head_depth" in obs:
                depth = to_numpy(obs["head_depth"])
                if depth.ndim == 3:
                    depth = depth[env_idx]
                self.data["/observations/head_depth"].append(depth.astype(np.float32))
            if "left_hand_depth" in obs:
                depth = to_numpy(obs["left_hand_depth"])
                if depth.ndim == 3:
                    depth = depth[env_idx]
                self.data["/observations/left_hand_depth"].append(depth.astype(np.float32))
            if "right_hand_depth" in obs:
                depth = to_numpy(obs["right_hand_depth"])
                if depth.ndim == 3:
                    depth = depth[env_idx]
                self.data["/observations/right_hand_depth"].append(depth.astype(np.float32))

        # Record joint positions from observations
        # The environment uses separate keys for each joint group
        if "left_arm_joint_pos" in obs:
            joint_pos = to_numpy(obs["left_arm_joint_pos"])
            if joint_pos.ndim == 2:
                joint_pos = joint_pos[env_idx]
            self.data["/observations/left_arm_joint_pos"].append(joint_pos.astype(np.float32))

        if "left_gripper_joint_pos" in obs:
            gripper_pos = to_numpy(obs["left_gripper_joint_pos"])
            if gripper_pos.ndim >= 1:
                if gripper_pos.ndim == 2:
                    gripper_pos = gripper_pos[env_idx]
                gripper_pos = np.atleast_1d(gripper_pos)
            self.data["/observations/left_gripper_pos"].append(gripper_pos.astype(np.float32))

        if "right_arm_joint_pos" in obs:
            joint_pos = to_numpy(obs["right_arm_joint_pos"])
            if joint_pos.ndim == 2:
                joint_pos = joint_pos[env_idx]
            self.data["/observations/right_arm_joint_pos"].append(joint_pos.astype(np.float32))

        if "right_gripper_joint_pos" in obs:
            gripper_pos = to_numpy(obs["right_gripper_joint_pos"])
            if gripper_pos.ndim >= 1:
                if gripper_pos.ndim == 2:
                    gripper_pos = gripper_pos[env_idx]
                gripper_pos = np.atleast_1d(gripper_pos)
            self.data["/observations/right_gripper_pos"].append(gripper_pos.astype(np.float32))

        # Record actions
        actions_np = to_numpy(actions)
        if actions_np.ndim == 2:
            actions_np = actions_np[env_idx]

        # Action format: [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)]
        self.data["/actions/left_arm_action"].append(actions_np[0:6].astype(np.float32))
        self.data["/actions/right_arm_action"].append(actions_np[6:12].astype(np.float32))
        self.data["/actions/left_gripper"].append(actions_np[12:13].astype(np.float32))
        self.data["/actions/right_gripper"].append(actions_np[13:14].astype(np.float32))

    def save_episode(self) -> str:
        """Save the current episode to an HDF5 file.

        Returns:
            Path to the saved file.
        """
        filepath = os.path.join(self.save_dir, f"episode_{self.episode_idx}.h5")

        with h5py.File(filepath, "w", rdcc_nbytes=1024**2 * 2) as f:
            # Set attributes
            f.attrs["sim"] = True
            f.attrs["timestamp"] = datetime.now().isoformat()
            f.attrs["num_frames"] = len(self.data["/timestamps"])

            # Save each dataset
            for key, value in self.data.items():
                if len(value) > 0:
                    arr = np.array(value)
                    f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)

        print(f"[Recorder] Saved episode to {filepath}")

        # Increment episode counter for next recording
        self.episode_idx += 1

        return filepath

    def discard_episode(self) -> None:
        """Discard the current episode without saving."""
        self._reset_buffers()
        self.is_recording = False
        print(f"[Recorder] Discarded episode {self.episode_idx}")

    @property
    def num_frames(self) -> int:
        """Get the number of frames recorded in the current episode."""
        return len(self.data["/timestamps"])
