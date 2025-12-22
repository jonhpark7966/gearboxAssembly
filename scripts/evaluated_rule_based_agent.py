# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Rule-based agent with step-by-step evaluation & retry
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rule-based agent with evaluation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--max_retries", type=int, default=1, help="Maximum number of retries per step.")
parser.add_argument("--evaluation_delay_steps", type=int, default=10, help="Number of steps to wait before evaluation.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

print(f"args_cli: {args_cli}")
print(f"Python path: {sys.path}")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab.sim as sim_utils

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

import Galaxea_Lab_External.tasks
from Galaxea_Lab_External.robots.evaluated_galaxea_rule_policy import EvaluatedGalaxeaRulePolicy
from Galaxea_Lab_External.tasks.direct.galaxea_lab_external.evaluated_galaxea_lab_external_env import EvaluatedGalaxeaLabExternalEnv


def replace_rule_policy_with_evaluated(env):
    """eval 및 재시도 로직을 위해 새로운 policy로 교체"""
    unwrapped_env = env.unwrapped
    
    old_policy = unwrapped_env.rule_policy
    new_policy = EvaluatedGalaxeaRulePolicy(
        sim_utils.SimulationContext.instance(),
        unwrapped_env.scene,
        unwrapped_env.obj_dict
    )
    
    new_policy.set_initial_root_state(old_policy.initial_root_state)
    new_policy.prepare_mounting_plan()
    new_policy.count = old_policy.count
    
    unwrapped_env.rule_policy = new_policy
    print("[INFO] Replaced rule_policy with EvaluatedGalaxeaRulePolicy")



def patch_env_with_evaluation(env):
    """원래 환경 객체에 evaluation method를 추가"""
    unwrapped_env = env.unwrapped
    
    eval_env = EvaluatedGalaxeaLabExternalEnv.__new__(EvaluatedGalaxeaLabExternalEnv)

    unwrapped_env.evaluate_gear_mounted_to_pin = eval_env.evaluate_gear_mounted_to_pin.__get__(unwrapped_env, type(unwrapped_env))
    unwrapped_env.evaluate_carrier_mounted_to_ring = eval_env.evaluate_carrier_mounted_to_ring.__get__(unwrapped_env, type(unwrapped_env))
    unwrapped_env.evaluate_reducer_mounted = eval_env.evaluate_reducer_mounted.__get__(unwrapped_env, type(unwrapped_env))
    
    print("[INFO] Added evaluation methods to environment")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print("env type: ", type(env))

    patch_env_with_evaluation(env)
    
    env.reset()

    replace_rule_policy_with_evaluated(env)

    # print info 
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    
    # Step evaluation mapping
    STEP_EVALUATION_MAP = {
        "mount_gear_1": lambda env, gear_id: env.unwrapped.evaluate_gear_mounted_to_pin(1),
        "mount_gear_2": lambda env, gear_id: env.unwrapped.evaluate_gear_mounted_to_pin(2),
        "mount_gear_3": lambda env, gear_id: env.unwrapped.evaluate_gear_mounted_to_pin(3),
        "mount_gear_4": lambda env, gear_id: env.unwrapped.evaluate_gear_mounted_to_pin(4),
        "mount_carrier_to_ring": lambda env, gear_id: env.unwrapped.evaluate_carrier_mounted_to_ring(),
        "mount_reducer": lambda env, gear_id: env.unwrapped.evaluate_reducer_mounted(),
    }
    
    # Step state tracking
    step_state = {
        "waiting_for_evaluation": False, # eval 대기 중인지
        "evaluation_delay_steps": args_cli.evaluation_delay_steps, # eval 하기 전 대기할 step 수
        "current_step_name": None, # 현재 평가 중인 step 이름
        "retry_count": {}, # 각 step별 재시도 횟수
        "max_retries": args_cli.max_retries, # 최대 재시도 횟수
    }
    
    # Episode success tracking
    episode_state = {
        "all_steps_completed": False,
        "episode_failed": False,
        "completed_steps": set(),  # Track which steps have been successfully completed
        "required_steps": {"mount_gear_1", "mount_gear_2", "mount_gear_3", "mount_gear_4", "mount_carrier_to_ring", "mount_reducer"},
    }
    
    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            rule_policy = env.unwrapped.rule_policy

            # 1) 평가 대기 상태일 때는 env.step()을 호출하지 않고,
            #    현재 상태를 고정한 채로 delay 카운트와 평가만 수행.
            if step_state["waiting_for_evaluation"]:
                step_state["evaluation_delay_steps"] -= 1
                if step_state["evaluation_delay_steps"] <= 0:
                    step_name = step_state["current_step_name"]
                    eval_func = STEP_EVALUATION_MAP.get(step_name)

                    if eval_func:
                        print(f"[EVAL] Evaluating {step_name}...")
                        try:
                            success = eval_func(env, rule_policy.completed_gear_id)
                            print(f"[EVAL] {step_name}: {'✓ SUCCESS' if success else '✗ FAILED'}")
                        except Exception as e:
                            print(f"[EVAL ERROR] Exception in evaluation function: {e}")
                            import traceback
                            traceback.print_exc()
                            success = False

                        if success:
                            step_state["waiting_for_evaluation"] = False
                            step_state["current_step_name"] = None
                            if step_name in step_state["retry_count"]:
                                del step_state["retry_count"][step_name]

                            # Track successful step completion
                            episode_state["completed_steps"].add(step_name)
                            print(f"[STEP] ✓ {step_name} completed successfully, proceeding to next step")

                            # Check if all required steps are completed
                            if episode_state["completed_steps"] == episode_state["required_steps"]:
                                episode_state["all_steps_completed"] = True
                                print(f"[EPISODE] ✓ All steps completed successfully!")
                        else:
                            # Failure: retry
                            if step_name not in step_state["retry_count"]:
                                step_state["retry_count"][step_name] = 0

                            step_state["retry_count"][step_name] += 1

                            if step_state["retry_count"][step_name] < step_state["max_retries"]:
                                # Restart current step
                                rule_policy.restart_step(step_name)
                                step_state["current_step_name"] = None  # Reset to allow re-detection
                                # Reset trigger flag to allow re-detection
                                trigger_key = f"{step_name}_triggered"
                                if trigger_key in step_state:
                                    del step_state[trigger_key]
                                print(
                                    f"[RETRY] {step_name} "
                                    f"(attempt {step_state['retry_count'][step_name]}/{step_state['max_retries']})"
                                )
                            else:
                                # Max retries exceeded - episode failed
                                print(
                                    f"[EPISODE] ✗ Max retries ({step_state['max_retries']}) "
                                    f"exceeded for {step_name}. Episode failed."
                                )
                                print(
                                    "[EPISODE] Resetting environment immediately to start fresh episode..."
                                )

                                # Reset environment completely (random reset)
                                env.reset()
                                replace_rule_policy_with_evaluated(env)

                                # Reset all state
                                episode_state["all_steps_completed"] = False
                                episode_state["episode_failed"] = False
                                episode_state["completed_steps"] = set()
                                step_state["waiting_for_evaluation"] = False
                                step_state["current_step_name"] = None
                                step_state["retry_count"] = {}
                                for key in list(step_state.keys()):
                                    if key.endswith("_triggered"):
                                        del step_state[key]

                                print(
                                    "[EPISODE] Environment reset complete. "
                                    "Starting fresh episode with clean data..."
                                )

                    else:
                        print(f"[WARN] No evaluation function for step: {step_name}")
                        step_state["waiting_for_evaluation"] = False

                # 평가 대기 중에는 env.step()을 호출하지 않고 다음 루프로 넘어간다.
                continue

            # 2) 평가 대기 상태가 아닐 때만 env.step()을 호출해서 정책을 진행.
            dummy_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs, reward, terminated, truncated, info = env.step(dummy_action)

            count_int = int(rule_policy.count)
            step_2_end = int(rule_policy.count_step_2[-1].item())
            step_4_end = int(rule_policy.count_step_4[-1].item())
            step_7_end = int(rule_policy.count_step_7[-1].item())
            step_9_end = int(rule_policy.count_step_9[-1].item())
            step_12_end = int(rule_policy.count_step_12[-1].item())
            step_14_end = int(rule_policy.count_step_14[-1].item())

            if (
                count_int >= step_2_end
                and not step_state["waiting_for_evaluation"]
                and step_state["current_step_name"] != "mount_gear_1"
                and not step_state.get("mount_gear_1_triggered", False)
            ):
                print(f"[STEP] Detected mount_gear_1 completion: count={count_int}, step_2_end={step_2_end}")
                step_state["mount_gear_1_triggered"] = True
                rule_policy.step_completed = True
                rule_policy.completed_step_name = "mount_gear_1"
                rule_policy.completed_gear_id = 1
            elif (
                count_int >= step_4_end
                and not step_state["waiting_for_evaluation"]
                and step_state["current_step_name"] != "mount_gear_2"
                and not step_state.get("mount_gear_2_triggered", False)
            ):
                print(f"[STEP] Detected mount_gear_2 completion: count={count_int}, step_4_end={step_4_end}")
                step_state["mount_gear_2_triggered"] = True
                rule_policy.step_completed = True
                rule_policy.completed_step_name = "mount_gear_2"
                rule_policy.completed_gear_id = 2
            elif (
                count_int >= step_7_end
                and not step_state["waiting_for_evaluation"]
                and step_state["current_step_name"] != "mount_gear_3"
                and not step_state.get("mount_gear_3_triggered", False)
            ):
                print(f"[STEP] Detected mount_gear_3 completion: count={count_int}, step_7_end={step_7_end}")
                step_state["mount_gear_3_triggered"] = True
                rule_policy.step_completed = True
                rule_policy.completed_step_name = "mount_gear_3"
                rule_policy.completed_gear_id = 3
            elif (
                count_int >= step_9_end
                and not step_state["waiting_for_evaluation"]
                and step_state["current_step_name"] != "mount_gear_4"
                and not step_state.get("mount_gear_4_triggered", False)
            ):
                print(f"[STEP] Detected mount_gear_4 completion: count={count_int}, step_9_end={step_9_end}")
                step_state["mount_gear_4_triggered"] = True
                rule_policy.step_completed = True
                rule_policy.completed_step_name = "mount_gear_4"
                rule_policy.completed_gear_id = 4
            elif (
                count_int >= step_12_end
                and not step_state["waiting_for_evaluation"]
                and step_state["current_step_name"] != "mount_carrier_to_ring"
                and not step_state.get("mount_carrier_to_ring_triggered", False)
            ):
                print(
                    f"[STEP] Detected mount_carrier_to_ring completion: "
                    f"count={count_int}, step_12_end={step_12_end}"
                )
                step_state["mount_carrier_to_ring_triggered"] = True
                rule_policy.step_completed = True
                rule_policy.completed_step_name = "mount_carrier_to_ring"
                rule_policy.completed_gear_id = None
            elif (
                count_int >= step_14_end
                and not step_state["waiting_for_evaluation"]
                and step_state["current_step_name"] != "mount_reducer"
                and not step_state.get("mount_reducer_triggered", False)
            ):
                print(f"[STEP] Detected mount_reducer completion: count={count_int}, step_14_end={step_14_end}")
                step_state["mount_reducer_triggered"] = True
                rule_policy.step_completed = True
                rule_policy.completed_step_name = "mount_reducer"
                rule_policy.completed_gear_id = None

            # 단계별 완료 확인: 이 iteration에서 step 완료가 감지되면,
            # 다음 루프부터 평가 대기 모드로 진입.
            if hasattr(rule_policy, "step_completed") and rule_policy.step_completed:
                step_name = rule_policy.completed_step_name
                step_state["waiting_for_evaluation"] = True
                step_state["evaluation_delay_steps"] = args_cli.evaluation_delay_steps
                step_state["current_step_name"] = step_name
                print(f"[STEP] Starting evaluation wait: {args_cli.evaluation_delay_steps} steps")
                rule_policy.step_completed = False  # Reset flag

            # 에피소드 종료 처리 (평가 대기 모드가 아닐 때만)
            if terminated.any() or truncated.any():
                print(f"[INFO] Episode terminated: terminated={terminated}, truncated={truncated}")

                # Log episode result
                if episode_state["all_steps_completed"]:
                    print(f"[EPISODE] ✓ Episode completed successfully!")
                elif episode_state["episode_failed"]:
                    print(f"[EPISODE] ✗ Episode failed.")
                else:
                    print(
                        "[EPISODE] Episode ended but not all steps completed "
                        "(timeout or other reason)."
                    )

                # Reset for next episode (same for all cases)
                env.reset()
                replace_rule_policy_with_evaluated(env)
                episode_state["all_steps_completed"] = False
                episode_state["episode_failed"] = False
                episode_state["completed_steps"] = set()
                step_state["waiting_for_evaluation"] = False
                step_state["current_step_name"] = None
                step_state["retry_count"] = {}
                for key in list(step_state.keys()):
                    if key.endswith("_triggered"):
                        del step_state[key]
                print(f"[EPISODE] Environment reset. Starting new episode...")

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
