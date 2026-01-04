# Galaxea R1 Agent Architecture

## Overview

이 문서는 Galaxea R1 로봇 에이전트의 아키텍처와 데이터 흐름을 설명합니다.
그리퍼 문제 디버깅을 위해 action이 어떻게 생성되고 적용되는지 추적할 수 있도록 작성되었습니다.

---

## 1. 파일 구조

```
scripts/
├── teleop_r1_agent.py      # XR 텔레오퍼레이션 (Vision Pro)
├── rule_based_agent.py     # 규칙 기반 랜덤 액션
├── VLA_agent.py            # VLA (ACT) 모델 기반
└── base_agent.py           # [NEW] 공통 베이스 코드

source/Galaxea_Lab_External/.../tasks/direct/galaxea_lab_agent/
├── galaxea_lab_agent_env.py      # 환경 클래스
└── galaxea_lab_agent_env_cfg.py  # 환경 설정
```

---

## 2. Action 데이터 흐름 (핵심!)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ACTION DATA FLOW                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [Agent Script]                                                              │
│       │                                                                      │
│       │  actions = torch.tensor([14D])                                       │
│       │  Shape: (num_envs, 14)                                              │
│       │  Format: [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)]
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  env.step(actions)                                                   │    │
│  │  File: galaxea_lab_agent_env.py:605                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       │  self.env_step_action = action  (Line 641)                          │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  for _ in range(decimation=5):   (Line 652)                         │    │
│  │      self._apply_action()                                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  _apply_action()  (Line 165)                                         │    │
│  │                                                                       │    │
│  │  action = self.env_step_action                                       │    │
│  │  self.robot.set_joint_position_target(action, self._joint_idx)       │    │
│  │                                       ▲            ▲                 │    │
│  │                                       │            │                 │    │
│  │                              (14D tensor)    (joint indices)         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  self.scene.write_data_to_sim()  (Line 657)                         │    │
│  │  self.sim.step(render=False)     (Line 659)                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Joint Index 매핑 (중요!)

### 3.1 환경 초기화 시 Joint Index 설정
**File:** `galaxea_lab_agent_env.py:47-70`

```python
# __init__()에서 joint index 검색
self._left_arm_joint_idx, _ = self.robot.find_joints("left_arm_joint.*")      # 6개
self._right_arm_joint_idx, _ = self.robot.find_joints("right_arm_joint.*")    # 6개
self._left_gripper_dof_idx, _ = self.robot.find_joints("left_gripper_axis1")  # 1개
self._right_gripper_dof_idx, _ = self.robot.find_joints("right_gripper_axis1") # 1개

# 전체 joint index 결합
self._joint_idx = self._left_arm_joint_idx + self._right_arm_joint_idx + \
                  self._left_gripper_dof_idx + self._right_gripper_dof_idx
```

### 3.2 Action Tensor와 Joint Index의 대응

| Action Index | Joint Name Pattern | 개수 | 범위 |
|--------------|-------------------|------|------|
| 0-5 | left_arm_joint[1-6] | 6 | 라디안 |
| 6-11 | right_arm_joint[1-6] | 6 | 라디안 |
| 12 | left_gripper_axis1 | 1 | 0.0 ~ 0.04 m |
| 13 | right_gripper_axis1 | 1 | 0.0 ~ 0.04 m |

### 3.3 잠재적 문제점 확인 필요

```python
# 환경 로그에서 확인해야 할 것:
print(f"_left_arm_joint_idx: {self._left_arm_joint_idx}")      # [??, ??, ??, ??, ??, ??]
print(f"_right_arm_joint_idx: {self._right_arm_joint_idx}")    # [??, ??, ??, ??, ??, ??]
print(f"_left_gripper_dof_idx: {self._left_gripper_dof_idx}")  # [??]
print(f"_right_gripper_dof_idx: {self._right_gripper_dof_idx}") # [??]

# 질문: find_joints()가 반환하는 순서가 action tensor 순서와 일치하는가?
```

---

## 4. 에이전트별 Action 생성 방식

### 4.1 Teleop Agent (teleop_r1_agent.py)

```
XR Hand Tracking (16D output)
    │
    ├── [0:3]   Left Position (world frame)
    ├── [3:7]   Left Quaternion (wxyz)
    ├── [7:8]   Left Grip (-1=close, +1=open)
    ├── [8:11]  Right Position (world frame)
    ├── [11:15] Right Quaternion (wxyz)
    └── [15:16] Right Grip (-1=close, +1=open)
    │
    ▼
┌─────────────────────────────────────┐
│  Differential IK Controller         │
│  - left_arm_joint[1-6] → link6     │
│  - right_arm_joint[1-6] → link6    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Gripper Mapping                    │
│  grip [-1,+1] → joint [0.0, 0.04]  │
│                                     │
│  normalized = (grip + 1) / 2        │
│  joint = close + normalized*(open-close)
└─────────────────────────────────────┘
    │
    ▼
actions = torch.cat([
    left_joint_targets,    # (6,)
    right_joint_targets,   # (6,)
    left_gripper_target,   # (1,) ← 여기!
    right_gripper_target,  # (1,) ← 여기!
], dim=-1)
```

### 4.2 Rule-Based Agent (rule_based_agent.py)

```python
# 단순히 랜덤 액션 생성
actions = 2 * torch.rand(env.action_space.shape, device=device) - 1
# Shape: (num_envs, 14)
# Range: [-1, 1] for all dimensions
```

### 4.3 VLA Agent (VLA_agent.py)

```python
# ACT Policy로부터 action 예측
qpos = torch.cat([
    left_arm_joint_pos,           # (6,)
    right_arm_joint_pos,          # (6,)
    left_gripper_joint_pos,       # (1,)
    right_gripper_joint_pos,      # (1,)
], dim=-1)  # Total: 14D

images = torch.cat([head_rgb, left_hand_rgb, right_hand_rgb], dim=1)

action = policy.predict(qpos, images)  # Returns 14D action
```

---

## 5. 그리퍼 문제 디버깅 체크리스트

### 5.1 Command Generation (확인 완료 ✓)
- [x] 손가락 거리 → grip value (-1/+1) 변환 정상
- [x] grip value → joint position (0.0~0.04) 변환 정상
- [x] 로그에서 target 값 변화 확인됨

### 5.2 Action Application (확인 필요 ⚠️)

```python
# 디버깅 추가 위치: galaxea_lab_agent_env.py

def _apply_action(self) -> None:
    action = self.env_step_action

    # === DEBUG: Action과 Joint Index 확인 ===
    print(f"[DEBUG] action.shape: {action.shape}")
    print(f"[DEBUG] action[0, 12:14]: {action[0, 12:14]}")  # 그리퍼 타겟
    print(f"[DEBUG] self._joint_idx: {self._joint_idx}")
    print(f"[DEBUG] len(_joint_idx): {len(self._joint_idx)}")

    # === DEBUG: 적용 전 현재 위치 ===
    curr_pos = self.robot.data.joint_pos[0, self._left_gripper_dof_idx[0]]
    print(f"[DEBUG] gripper BEFORE: {curr_pos.item():.4f}")

    self.robot.set_joint_position_target(action, self._joint_idx)

    # === DEBUG: 적용 후 타겟 확인 ===
    # (write_data_to_sim 후에 확인해야 함)
```

### 5.3 확인해야 할 질문들

1. **Joint Index 순서 일치 여부**
   - `self._joint_idx`의 순서가 action tensor의 순서와 일치하는가?
   - `find_joints("left_arm_joint.*")`가 joint1, 2, 3, 4, 5, 6 순서로 반환하는가?

2. **Gripper Axis2 존재 여부**
   - 로봇에 `left_gripper_axis2`가 있는지?
   - axis1만 제어하면 axis2는 어떻게 되는지?

3. **Joint Limit 확인**
   - 그리퍼 조인트의 position limit이 올바르게 설정되어 있는지?
   - USD 파일에서 joint limit 확인 필요

4. **Mimic Joint 설정**
   - 그리퍼가 mimic joint로 설정되어 있다면, axis1과 axis2가 연동되는지?

---

## 6. 코드 따라가기 순서

### Step 1: Action 생성 확인
```
teleop_r1_agent.py:912-920
```
- `actions = torch.cat([...])`의 shape과 값 확인

### Step 2: env.step() 진입
```
galaxea_lab_agent_env.py:605 → step()
```
- `action = action.to(self.device)` 확인

### Step 3: Action 저장
```
galaxea_lab_agent_env.py:641
```
- `self.env_step_action = action`

### Step 4: Physics Step Loop
```
galaxea_lab_agent_env.py:652-666
```
- `for _ in range(self.cfg.decimation):` → 5번 반복
- `self._apply_action()` 호출

### Step 5: Action 적용 (핵심!)
```
galaxea_lab_agent_env.py:165-174
```
```python
def _apply_action(self) -> None:
    action = self.env_step_action
    self.robot.set_joint_position_target(action, self._joint_idx)
```

### Step 6: 시뮬레이션 스텝
```
galaxea_lab_agent_env.py:657-659
```
- `self.scene.write_data_to_sim()` - 타겟을 시뮬레이터에 씀
- `self.sim.step(render=False)` - 물리 시뮬레이션 실행

---

## 7. 로봇 Actuator 설정

**File:** `galaxea_robots.py:105-113`

```python
"r1_grippers": ImplicitActuatorCfg(
    joint_names_expr=[".*_gripper_axis1"],
    effort_limit_sim=100.0,
    velocity_limit_sim=0.07,    # 느림 (0.5초에 전체 범위)
    stiffness=25000.0,          # 매우 높음
    damping=1000.0,             # 높음
    friction=0.2,
    armature=0.2,
),
```

**주의:** `velocity_limit_sim=0.07`은 느리지만, 충분한 시간이 주어지면 작동해야 함.
User의 테스트에서 충분한 시간을 줬는데도 안 열림 → 속도/댐핑 문제 아님

---

## 8. 의심되는 문제 원인

### 가설 1: Joint Index 불일치
- `find_joints()`의 반환 순서가 예상과 다를 수 있음
- Action tensor의 index 12, 13이 실제로 gripper에 매핑되지 않을 수 있음

### 가설 2: Gripper Axis2 미제어
- `gripper_axis2`가 있는데 제어 안 함
- Mimic joint가 아니라면 axis2가 닫힌 상태로 유지되어 axis1이 열리지 못함

### 가설 3: USD 파일 내 Joint Constraint
- USD 파일에 그리퍼 조인트에 대한 추가 제약이 있을 수 있음
- Joint drive 설정 확인 필요

### 가설 4: Action이 덮어쓰기됨
- `step()` 함수 끝부분(Line 708-720)에서 action을 다시 쓰는 코드가 있음
- `self.env_step_joint_ids` 조건에 따라 action이 변경될 수 있음

---

## 9. 디버깅 코드 (이미 추가됨)

### 9.1 환경 초기화 시 출력 (`__init__`)
환경 시작 시 다음 정보가 출력됩니다:
- 로봇의 모든 joint 이름
- `_joint_idx` 값과 각 인덱스가 어떤 joint에 매핑되는지
- axis2 joint 존재 여부 (경고)

### 9.2 Action 적용 시 출력 (`_apply_action`)
처음 20회 `_apply_action()` 호출마다:
- 그리퍼 Target 값 (action[12], action[13])
- 그리퍼 Actual 값 (robot.data.joint_pos)
- Error (Target - Actual)

---

## 10. 테스트 방법 및 확인 사항

### Step 1: 환경 실행
```bash
./isaaclab.sh -p scripts/rule_based_agent.py \
    --task Template-Galaxea-Lab-Agent-Direct-v0 \
    --debug
```

### Step 2: 환경 초기화 출력 확인

```
============================================================
=== JOINT INDEX VERIFICATION (Gripper Debug) ===
============================================================
All robot joint names: ['left_arm_joint1', 'left_arm_joint2', ...]
Total joints in robot: XX

_joint_idx (combined): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
len(_joint_idx): 14

Expected mapping:
  action[0:6]  -> left_arm_joint[1-6]
  action[6:12] -> right_arm_joint[1-6]
  action[12]   -> left_gripper_axis1
  action[13]   -> right_gripper_axis1

Actual _joint_idx -> joint_name mapping:
  action[ 0] -> _joint_idx[0]=XX -> left_arm_joint1
  ...
  action[12] -> _joint_idx[12]=XX -> left_gripper_axis1  <-- 이것 확인!
  action[13] -> _joint_idx[13]=XX -> right_gripper_axis1 <-- 이것 확인!
============================================================
```

### Step 3: 확인해야 할 것들

1. **action[12]가 정확히 left_gripper_axis1에 매핑되는가?**
   - 만약 다른 joint에 매핑되면 → 버그 발견!

2. **axis2 joints가 존재하는가?**
   - 존재하면 경고 메시지가 출력됨
   - axis2가 있는데 제어 안 하면 → 물리적으로 막힐 수 있음

3. **Target과 Actual의 차이가 시간이 지나도 줄어들지 않는가?**
   - 줄어들지 않으면 → actuator 문제 또는 물리적 제약

### Step 4: 가능한 문제 원인

| 증상 | 가능한 원인 |
|------|------------|
| action[12]가 다른 joint에 매핑 | `find_joints()` 반환 순서 문제 |
| axis2 존재 + 경고 | axis2가 gripper 열림을 막음 |
| Target 변화하는데 Actual 안 변함 | actuator 설정 또는 USD 제약 |
| Target도 Actual도 안 변함 | action이 제대로 전달 안 됨 |
