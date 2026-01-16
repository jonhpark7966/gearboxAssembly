# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause
"""
LeRobot Client Agent for Galaxea R1 Robot

ZMQ를 통해 LeRobot inference 서버와 통신하여
SmolVLA 또는 Pi0.5 모델로 로봇을 제어합니다.

Usage:
    # 먼저 LeRobot 서버를 시작한 후:
    ./isaaclab.sh -p scripts/lerobot_client_agent.py \
        --server_host localhost \
        --server_port 5555 \
        --task Template-Galaxea-Lab-Agent-Direct-v0 \
        --num_envs 1

Data Flow:
    1. Isaac Lab 환경에서 observation 수집
    2. 이미지를 JPEG로 압축, base64 인코딩
    3. ZMQ로 서버에 전송
    4. 서버에서 inference 수행
    5. 14D action 수신 (Isaac Lab 형식)
    6. env.step(action) 실행
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

# ============================================================================
# CLI Arguments
# ============================================================================
parser = argparse.ArgumentParser(description="LeRobot Client Agent for Galaxea R1")
parser.add_argument(
    "--server_host", type=str, default="localhost",
    help="LeRobot inference server host"
)
parser.add_argument(
    "--server_port", type=int, default=5555,
    help="LeRobot inference server port"
)
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
parser.add_argument(
    "--timeout_ms", type=int, default=5000,
    help="Inference request timeout in milliseconds"
)
parser.add_argument(
    "--record_video", type=str, default=None,
    help="Path to save head camera video recording (e.g., ./output.mp4)"
)
parser.add_argument(
    "--record_fps", type=int, default=30,
    help="FPS for video recording (default: 30)"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

print(f"[LeRobotClient] args_cli: {args_cli}")

# ============================================================================
# Launch Application
# ============================================================================
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# Post-Launch Imports
# ============================================================================
import base64
import signal
import subprocess
import time
from io import BytesIO
from pathlib import Path

import cv2
import gymnasium as gym
import msgpack
import numpy as np
import torch
import zmq
from PIL import Image

# Global flag for graceful shutdown
_shutdown_requested = False

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import Galaxea_Lab_External.tasks  # Register environment

# Import base agent utilities
from base_agent import (
    TOTAL_ACTION_DIM,
    parse_observation,
    print_action_debug,
    print_joint_index_debug,
)


# ============================================================================
# Video Recorder Class
# ============================================================================
class VideoRecorder:
    """헤드캠 비디오 녹화"""

    def __init__(self, output_path: str, fps: int = 30, resolution: tuple = None):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.resolution = resolution
        self.writer = None
        self.frame_count = 0

    def add_frame(self, img_tensor: torch.Tensor):
        """프레임 추가 (H, W, C) 또는 (1, H, W, C) 형식"""
        # Tensor to numpy
        if isinstance(img_tensor, torch.Tensor):
            img = img_tensor.cpu().numpy()
        else:
            img = img_tensor

        # Remove batch dimension if present
        if img.ndim == 4:
            img = img[0]

        # Ensure uint8
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Initialize writer on first frame
        if self.writer is None:
            h, w = img_bgr.shape[:2]
            if self.resolution:
                w, h = self.resolution
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (w, h))
            print(f"[VideoRecorder] Recording to {self.output_path} at {self.fps}fps, resolution {w}x{h}")

        # Resize if needed
        if self.resolution:
            img_bgr = cv2.resize(img_bgr, self.resolution)

        self.writer.write(img_bgr)
        self.frame_count += 1

    def close(self):
        """녹화 종료 및 H.264 변환"""
        if self.writer:
            self.writer.release()
            print(f"[VideoRecorder] Saved {self.frame_count} frames to {self.output_path}")

            # ffmpeg로 H.264 변환 (브라우저/플레이어 호환성)
            temp_path = self.output_path
            final_path = self.output_path.with_suffix('.mp4')

            # 임시 파일명으로 변경
            temp_raw = self.output_path.with_name(self.output_path.stem + '_raw.mp4')
            temp_path.rename(temp_raw)

            try:
                print(f"[VideoRecorder] Converting to H.264 with ffmpeg...")
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(temp_raw),
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    str(final_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    temp_raw.unlink()  # 임시 파일 삭제
                    print(f"[VideoRecorder] Successfully converted to {final_path}")
                else:
                    print(f"[VideoRecorder] ffmpeg conversion failed: {result.stderr}")
                    temp_raw.rename(final_path)  # 실패 시 원본 유지

            except FileNotFoundError:
                print("[VideoRecorder] ffmpeg not found, keeping original mp4v format")
                temp_raw.rename(final_path)


# ============================================================================
# LeRobot Client Class
# ============================================================================
class LeRobotClient:
    """ZMQ 기반 LeRobot 추론 클라이언트"""

    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 5000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms

        # ZMQ 설정
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")

        self.connected = False
        self.model_type = None

    def ping(self) -> bool:
        """서버 연결 확인"""
        try:
            self.socket.send(msgpack.packb({"type": "ping"}))
            response = msgpack.unpackb(self.socket.recv(), raw=False)
            if response.get("status") == "pong":
                self.connected = True
                self.model_type = response.get("model_type", "unknown")
                return True
            return False
        except zmq.ZMQError as e:
            print(f"[LeRobotClient] Ping failed: {e}")
            return False

    def reset(self):
        """정책 상태 리셋 (에피소드 시작 시 호출)"""
        try:
            self.socket.send(msgpack.packb({"type": "reset"}))
            response = msgpack.unpackb(self.socket.recv(), raw=False)
            return response.get("status") == "success"
        except zmq.ZMQError as e:
            print(f"[LeRobotClient] Reset failed: {e}")
            return False

    def predict(self, obs_parsed) -> np.ndarray:
        """추론 요청

        Args:
            obs_parsed: base_agent.parse_observation()의 반환값

        Returns:
            action: 14D numpy array (Isaac Lab 형식)
        """
        # 이미지를 JPEG로 인코딩
        images = {}
        for key, img_tensor in [
            ("head", obs_parsed.head_rgb),
            ("left_hand", obs_parsed.left_hand_rgb),
            ("right_hand", obs_parsed.right_hand_rgb),
        ]:
            img = img_tensor[0].cpu().numpy()  # (H, W, C)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG", quality=95)
            images[key] = base64.b64encode(buffer.getvalue()).decode("ascii")

        # State 구성 (28D - 전체 robot state)
        # Dataset format: [left_pos(6), left_vel(6), left_grip(1), left_grip_vel(1),
        #                  right_pos(6), right_vel(6), right_grip(1), right_grip_vel(1)]
        left_arm_pos = obs_parsed.left_arm_joint_pos[0].cpu().numpy().tolist()    # 6
        left_arm_vel = obs_parsed.left_arm_joint_vel[0].cpu().numpy().tolist()    # 6
        left_grip_pos = obs_parsed.left_gripper_pos[0].cpu().item()               # 1
        left_grip_vel = obs_parsed.left_gripper_vel[0].cpu().item()               # 1
        right_arm_pos = obs_parsed.right_arm_joint_pos[0].cpu().numpy().tolist()  # 6
        right_arm_vel = obs_parsed.right_arm_joint_vel[0].cpu().numpy().tolist()  # 6
        right_grip_pos = obs_parsed.right_gripper_pos[0].cpu().item()             # 1
        right_grip_vel = obs_parsed.right_gripper_vel[0].cpu().item()             # 1

        state = (
            left_arm_pos +           # [0-5]   left arm positions
            left_arm_vel +           # [6-11]  left arm velocities
            [left_grip_pos] +        # [12]    left gripper position
            [left_grip_vel] +        # [13]    left gripper velocity
            right_arm_pos +          # [14-19] right arm positions
            right_arm_vel +          # [20-25] right arm velocities
            [right_grip_pos] +       # [26]    right gripper position
            [right_grip_vel]         # [27]    right gripper velocity
        )  # Total: 28D

        request = {
            "type": "inference",
            "observation": {
                "images": images,
                "state": state,
            }
        }

        try:
            self.socket.send(msgpack.packb(request))
            response = msgpack.unpackb(self.socket.recv(), raw=False)

            if response["status"] == "success":
                return np.array(response["action"], dtype=np.float32)
            else:
                raise RuntimeError(f"Inference failed: {response.get('error_message')}")

        except zmq.ZMQError as e:
            raise RuntimeError(f"Communication error: {e}")

    def shutdown(self):
        """서버 종료 요청"""
        try:
            self.socket.send(msgpack.packb({"type": "shutdown"}))
            self.socket.recv()
        except:
            pass

    def close(self):
        """리소스 정리"""
        self.socket.close()
        self.context.term()


# ============================================================================
# Main Function
# ============================================================================
def main():
    """LeRobot Client Agent 메인 루프"""
    global _shutdown_requested

    # Resources to cleanup
    video_recorder = None
    client = None
    env = None

    def signal_handler(signum, frame):
        global _shutdown_requested
        print(f"\n[LeRobotClient] Received signal {signum}, shutting down gracefully...")
        _shutdown_requested = True

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
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

        print(f"[LeRobotClient] Environment type: {type(env)}")
        print(f"[LeRobotClient] Observation space: {env.observation_space}")
        print(f"[LeRobotClient] Action space: {env.action_space}")

        # Debug: Joint index 확인
        if args_cli.debug:
            print_joint_index_debug(env_unwrapped)

        # ========================================
        # Step 3: Connect to LeRobot Server
        # ========================================
        print(f"[LeRobotClient] Connecting to server at {args_cli.server_host}:{args_cli.server_port}")
        client = LeRobotClient(
            host=args_cli.server_host,
            port=args_cli.server_port,
            timeout_ms=args_cli.timeout_ms
        )

        # 연결 확인 (최대 10회 시도)
        for attempt in range(10):
            if client.ping():
                print(f"[LeRobotClient] Connected! Model type: {client.model_type}")
                break
            print(f"[LeRobotClient] Connection attempt {attempt + 1}/10 failed, retrying...")
            time.sleep(1)
        else:
            print("[LeRobotClient] Failed to connect to server")
            return

        # ========================================
        # Step 4: Reset Environment
        # ========================================
        obs, _ = env.reset()
        client.reset()
        print("[LeRobotClient] Environment reset complete")

        # ========================================
        # Step 4.5: Initialize Video Recorder (if enabled)
        # ========================================
        if args_cli.record_video:
            video_recorder = VideoRecorder(
                output_path=args_cli.record_video,
                fps=args_cli.record_fps
            )
            print(f"[LeRobotClient] Video recording enabled: {args_cli.record_video}")

        # ========================================
        # Step 5: Main Simulation Loop
        # ========================================
        step_count = 0
        episode_count = 0
        total_inference_time = 0.0

        while simulation_app.is_running() and not _shutdown_requested:
            # ----------------------------------------
            # Step 5a: Parse Observation
            # ----------------------------------------
            obs_parsed = parse_observation(obs)

            # ----------------------------------------
            # Step 5a.1: Record Head Camera Frame (if enabled)
            # ----------------------------------------
            if video_recorder:
                video_recorder.add_frame(obs_parsed.head_rgb)

            # ----------------------------------------
            # Step 5b: Request Inference from Server
            # ----------------------------------------
            try:
                start_time = time.time()
                action_np = client.predict(obs_parsed)
                inference_time = (time.time() - start_time) * 1000
                total_inference_time += inference_time

            except RuntimeError as e:
                print(f"[LeRobotClient] Inference error: {e}")
                # 에러 시 현재 위치 유지 (zero action)
                action_np = np.zeros(TOTAL_ACTION_DIM, dtype=np.float32)

            # ----------------------------------------
            # Step 5c: Apply Action to Environment
            # ----------------------------------------
            action = torch.from_numpy(action_np).unsqueeze(0).to(env_unwrapped.device)

            if args_cli.debug and step_count < 5:
                print(f"\n[LeRobotClient] Step {step_count}")
                print(f"  Inference time: {inference_time:.1f}ms")
                print_action_debug(action, prefix="  ")

            obs, reward, terminated, truncated, info = env.step(action)

            # ----------------------------------------
            # Step 5d: Handle Episode End
            # ----------------------------------------
            if terminated.any() or truncated.any():
                episode_count += 1
                avg_inference = total_inference_time / max(step_count, 1)
                print(f"[LeRobotClient] Episode {episode_count} ended at step {step_count}")
                print(f"[LeRobotClient] Average inference time: {avg_inference:.1f}ms")

                obs, _ = env.reset()
                client.reset()
                step_count = 0
                total_inference_time = 0.0
                continue

            step_count += 1

            # Periodic status
            if step_count % 100 == 0:
                avg_inference = total_inference_time / step_count
                print(f"[LeRobotClient] Step {step_count}, Reward: {reward}, "
                      f"Avg inference: {avg_inference:.1f}ms")

    finally:
        # ========================================
        # Step 6: Cleanup (always executed)
        # ========================================
        print("[LeRobotClient] Cleaning up...")
        if video_recorder:
            video_recorder.close()
        if client:
            client.close()
        if env:
            env.close()
        print("[LeRobotClient] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()
