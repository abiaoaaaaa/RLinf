# Zadai EEF 数据转换与 Demo Buffer 使用指南

这套工具将 Zadai 200 条双臂 joint-space 示教转换为 20 维绝对 EEF 数据，支持人工标注 RL 关键阶段，并生成 RLinf RLT Stage2 可以加载的 demo buffer。所有推荐命令都写入新目录，不修改原始数据。

## 流程与数据所有权

```text
16D joint LeRobot v3 原始数据（长期保留）
  │
  ├─ FK 转换
  ▼
20D absolute EEF LeRobot v3（规范数据源，长期保留）
  │
  ├─ 结构转换 ───────────────► RLinf LeRobot v2.1 ─► Pi0.5 + RLT Stage1
  │                                  │
  ├─ 人工键盘标注                     └─ Stage1 checkpoint
  ▼                                                        │
RL 区间与成功/失败 JSON（长期保留）                          │
  │                                                        │
  └──────────────────── 特征物化 ◄─────────────────────────┘
                           │
                           ▼
                  Stage2 EEF demo buffer（派生缓存）

旧 16D online replay buffer ── FK 转换 ──► 20D EEF replay buffer（新目录）
```

将 EEF LeRobot 数据集和人工标注 JSON 作为事实源。`z_rl` 和 `ref_chunk` 依赖具体的 Stage1 checkpoint，只是可以重新生成的派生缓存。

## 数据格式

原始 16 维顺序：

```text
[Joint1_L ... Joint7_L, Joint1_R ... Joint7_R, gripper_L, gripper_R]
```

转换后的 20 维绝对 EEF 顺序：

```text
[left_xyz(3), left_rot6d(6), left_gripper,
 right_xyz(3), right_rot6d(6), right_gripper]
```

- 位姿参考系：`base_link`
- 左右末端 frame：`left_tool`、`right_tool`
- 位置单位：米
- 关节单位：弧度
- 旋转表示：旋转矩阵前两列组成的 rot6d
- 数据集保存：绝对 EEF 目标
- Pi0.5 训练输入：以 action chunk 起始观测为参考的 body-frame SE(3) 增量

保存绝对目标可以保留可审计的物理语义。训练时的增量表示由 OpenPI transform 生成，不应提前破坏原始目标。

## 前置条件

从 RLinf 仓库根目录执行所有命令：

```bash
cd /home/zzx23457/code_xlb/RLinf
export PYTHONPATH="$PWD:${PYTHONPATH}"
```

不同步骤需要不同依赖：

| 步骤 | 必需依赖 | 其他条件 |
|---|---|---|
| joint → EEF 数据集 | `numpy`、`pyarrow`、`tqdm`、Pinocchio | 能加载 APEX MJCF |
| v3 → RLinf v2.1 | `pyarrow` | 视频 hardlink 要求源和目标位于同一文件系统 |
| norm stats | RLinf OpenPI 环境、`pyarrow` | 不需要解码视频 |
| 人工标注 | `opencv-python`、`pyarrow` | 需要桌面显示或 X11 转发 |
| demo buffer 特征物化 | 完整 RLinf OpenPI 环境、PyTorch | 需要 EEF Stage1 checkpoint，建议使用 CUDA |
| online buffer → EEF | PyTorch、Pinocchio、`pyarrow` | 使用与数据集转换相同的 MJCF |

安装 OpenPI/RLinf 训练环境可参考：

```bash
bash requirements/install.sh embodied --model openpi --env maniskill_libero
```

Pinocchio 必须来自能正确执行 `import pinocchio` 的 APEX/机器人环境。不要仅因为系统存在 ROS 包就假设当前 Python 能加载对应绑定。

## 1. 将 Joint 数据集转换为 EEF

程序：[`convert_joint_lerobot_to_eef.py`](convert_joint_lerobot_to_eef.py)

推荐一次生成 EEF v3 事实源和 RLinf v2.1 训练副本：

```bash
python3 toolkits/marvin/convert_joint_lerobot_to_eef.py \
  --src /home/zzx23457/Data/lerobot/zadai_200 \
  --dst /home/zzx23457/Data/lerobot/zadai_200_eef_rot6d \
  --rlinf-v21-dst /home/zzx23457/Data/lerobot/zadai_200_eef_rot6d_rlinf_v21 \
  --mjcf /home/zzx23457/ws_apex/Apex_Deploy-main/robot_node/marvin_description/mjcf/matrix/m6_696.xml \
  --media-mode hardlink
```

该命令会完成以下操作：

1. 对观测关节和动作目标分别执行 FK。
2. 写入 20D `observation.state` 和 `action`。
3. 保留 joint position、EEF quaternion、EEF twist 等审计字段。
4. 在新目录生成 v3 数据和 RLinf 兼容的 v2.1 数据。
5. 复用视频，不进行重新编码。

媒体模式：

| 模式 | 特点 | 使用建议 |
|---|---|---|
| `hardlink` | 不重复占用视频空间；源和目标必须在同一文件系统 | 默认选择 |
| `symlink` | 不复制视频，但移动源目录后会失效 | 固定存储位置时使用 |
| `copy` | 完全独立，额外占用磁盘空间 | 需要可迁移副本时使用 |

默认情况下目标目录已存在会报错。不要传 `--overwrite`，除非你明确要替换目标副本；该参数不会删除 `--src`，但会删除已有的目标目录。

如果已经有 EEF v3 数据，也可以单独转换成 RLinf v2.1：

```bash
python3 toolkits/lerobot/convert_zadai_v3_to_rlinf_v21.py \
  /home/zzx23457/Data/lerobot/zadai_200_eef_rot6d \
  /home/zzx23457/Data/lerobot/zadai_200_eef_rot6d_rlinf_v21 \
  --video-mode hardlink
```

## 2. 计算 Pi0.5 EEF 归一化统计

程序：[`calculate_zadai_eef_norm_stats.py`](calculate_zadai_eef_norm_stats.py)

```bash
python3 toolkits/marvin/calculate_zadai_eef_norm_stats.py \
  --dataset /home/zzx23457/Data/lerobot/zadai_200_eef_rot6d_rlinf_v21 \
  --output-dir /home/zzx23457/Data/lerobot/zadai_200_eef_norm_stats \
  --action-horizon 50 \
  --model-action-dim 32
```

输出文件：

```text
/home/zzx23457/Data/lerobot/zadai_200_eef_norm_stats/norm_stats.json
```

该程序不解码相机视频。它按 LeRobot action chunk 规则补齐 episode 末尾动作，然后计算 body-frame SE(3) 增量的 OpenPI 统计。`--action-horizon` 必须与 Stage1 配置的 `action_horizon` 一致；当前配置为 50。

## 3. 启动 EEF Pi0.5 + RLT Stage1 联合训练

配置：[`examples/sft/config/zadai_200_eef_sft_openpi_pi05.yaml`](../../examples/sft/config/zadai_200_eef_sft_openpi_pi05.yaml)

```bash
export ZADAI_EEF_DATA=/home/zzx23457/Data/lerobot/zadai_200_eef_rot6d_rlinf_v21
export ZADAI_EEF_NORM_STATS=/home/zzx23457/Data/lerobot/zadai_200_eef_norm_stats/norm_stats.json
export ZADAI_PI05_INIT_CKPT=/path/to/initial/pi05/checkpoint

bash examples/sft/run_vla_sft.sh zadai_200_eef_sft_openpi_pi05
```

这个配置同时训练 Pi0.5 action expert 和 RLT encoder。人工数据标注不依赖该训练，可以在 Stage1 运行前、运行中或运行后完成。

可选的 compressor-only 精炼配置是 `zadai_200_eef_rlt_stage1_sft_openpi_pi05.yaml`。默认联合训练流程不要求额外执行它。

## 4. 人工标注 RL 关键阶段和任务结果

程序：[`label_eef_rl_demo_buffer.py`](label_eef_rl_demo_buffer.py)

```bash
python3 toolkits/marvin/label_eef_rl_demo_buffer.py annotate \
  --dataset /home/zzx23457/Data/lerobot/zadai_200_eef_rot6d_rlinf_v21 \
  --annotations /home/zzx23457/Data/lerobot/zadai_200_eef_rl_annotations.json
```

从指定 episode 开始：

```bash
python3 toolkits/marvin/label_eef_rl_demo_buffer.py annotate \
  --dataset /home/zzx23457/Data/lerobot/zadai_200_eef_rot6d_rlinf_v21 \
  --annotations /home/zzx23457/Data/lerobot/zadai_200_eef_rl_annotations.json \
  --start-episode 80
```

### 键盘操作

| 按键 | 功能 |
|---|---|
| `Space` | 播放或暂停 |
| `A` / `D` | 后退或前进一帧 |
| `J` / `L` | 后退或前进一秒 |
| `[` | 标记一个 RL 区间的开始 |
| `]` | 标记当前 RL 区间的结束 |
| `S` | 在当前帧标记成功终点，并关闭已打开的 RL 区间 |
| `F` | 在当前帧标记失败终点，并关闭已打开的 RL 区间 |
| `N` / `B` | 保存并进入下一条或上一条 episode |
| `X` | 清除当前 episode 的标注 |
| `Q` / `Esc` | 保存并退出 |

一个 episode 可以包含多个 RL 区间。标注会原子写入独立 JSON；程序中断后可以继续。它不会修改 parquet、元数据或视频。

## 5. 将标注数据物化为 Stage2 Demo Buffer

标注完成后，可以等待 EEF Stage1 checkpoint 确定，再生成 RLT buffer：

```bash
python3 toolkits/marvin/label_eef_rl_demo_buffer.py export \
  --dataset /home/zzx23457/Data/lerobot/zadai_200_eef_rot6d_rlinf_v21 \
  --annotations /home/zzx23457/Data/lerobot/zadai_200_eef_rl_annotations.json \
  --output /home/zzx23457/code_xlb/RLinf/replay_buffers/zadai_200_eef_demo/rank_0 \
  --stage1-checkpoint /path/to/eef_stage1/checkpoints/global_step_N/actor \
  --norm-stats /home/zzx23457/Data/lerobot/zadai_200_eef_norm_stats/norm_stats.json \
  --device cuda \
  --chunk-size 30 \
  --ref-chunk-size 50
```

输出是 RLinf `TrajectoryReplayBuffer` rank 目录：

```text
rank_0/
├── metadata.json
├── trajectory_index.json
├── trajectory_0_offline_eef_demo.pt
├── ...
├── source_annotations.json
└── eef_buffer_manifest.json
```

每 30 个低层动作生成一个 transition。程序会写入：

| 字段 | 内容 |
|---|---|
| `curr_obs` / `next_obs` | `z_rl`、20D EEF `proprio`、EEF `ref_chunk` |
| `actions` | 30 × 20D 归一化 EEF 专家动作（Stage2 policy domain） |
| `intervene_flags` | 全部为 `True`，供 demo BC 分支识别 |
| `rewards` | 成功终止 chunk 的最后一步为 1，其余为 0 |
| `dones` | 标注终点所在 chunk 的最后一步为 `True` |

失败片段默认不进入专家 demo buffer。需要将失败轨迹提供给 critic 时，显式增加：

```text
--include-failures
```

> [!WARNING]
> `z_rl` 和 `ref_chunk` 与 `--stage1-checkpoint` 绑定。更换或继续训练 encoder 后，应重新执行 `export`。不要将来自不同 Stage1 encoder 的 `z_rl` 无条件混入同一个 Stage2 buffer。

在 Stage2 配置中指向生成结果：

```bash
export ZADAI_EEF_DEMO_BUFFER_PATH=/home/zzx23457/code_xlb/RLinf/replay_buffers/zadai_200_eef_demo/rank_0
```

### Stage2 接入与启动

EEF 使用增量配置 `examples/embodiment/config/zadai_200_eef_rlt_stage2_ac_mlp.yaml`。它继承已验证的 joint Stage2 调度、replay 和优化参数，只覆盖 EEF 的状态、动作、checkpoint 和安全约束；原 joint 配置保持 16D 且仍可独立启动。

| 配置或组件 | EEF 要求 |
|---|---|
| `actor.model.action_dim` | `20` |
| `actor.model.proprio_dim` | `20` |
| `actor.model.num_action_chunks` | 与 demo 的 `--chunk-size` 一致，当前为 `30` |
| `actor.model.ref_num_action_chunks` | 不小于 `num_action_chunks`；与 Stage1 reference horizon 对齐 |
| `rollout.rlt_feature_model.openpi.config_name` | `pi05_zadai_eef_rot6d` |
| `rollout.rlt_feature_model.openpi.action_env_dim` | `20` |
| `rollout.rlt_feature_model.openpi_data.norm_stats_path` | 指向 EEF `norm_stats.json` |
| `rollout.rlt_feature_model.openpi.rlt_action_space` | `eef` |
| 环境 action adapter | 20D normalized EEF → 20D absolute EEF → VLAHost `xyz+xyzw` |

启动前设置最终 Stage1 actor 目录、同源 norm stats 和两个独立 buffer 路径：

```bash
export ZADAI_EEF_STAGE1_CHECKPOINT=/path/to/eef_stage1/checkpoints/global_step_N/actor
export ZADAI_EEF_NORM_STATS=/home/zzx23457/Data/lerobot/zadai_200_eef_norm_stats/norm_stats.json
export ZADAI_EEF_DEMO_BUFFER_PATH=/home/zzx23457/code_xlb/RLinf/replay_buffers/zadai_200_eef_demo/rank_0
export ZADAI_EEF_REPLAY_BUFFER_PATH=/home/zzx23457/code_xlb/RLinf/replay_buffers/zadai_200_eef_online
export ZADAI_VLAHOST_URL=http://ROBOT_HOST:8010

bash examples/embodiment/run_embodiment.sh zadai_200_eef_rlt_stage2_ac_mlp
```

VLAHost 的 `/state` 应返回：

```json
{"capabilities": {"control_spaces": ["joint", "eef"]}}
```

EEF action 由 VLAHost 以 30 Hz 调用 APEX 已有的 `control/call_qp_planning` service；`marvin_qp_controller` 继续以 500 Hz 求解并输出 joint command。service 请求持有 150 ms 短租约，防止 `teleop_manager` 的普通 `control/ik_request` 覆盖 policy target；脚踏接管前 VLAHost 会发送全 unavailable 请求立即释放租约。joint action 仍发布到原 user joint topics。不要同时在同一个 action/chunk 中发送 joint 与 EEF arm target。

旋转安全限制同时在 RLinf 和 VLAHost 生效：单步上限为 `0.12 rad`，最大角速度为 `1.5 rad/s`。30 Hz action 因此实际限制为 `min(0.12, 1.5 / 30) = 0.05 rad/step`，并按 SO(3) 最短旋转角裁剪。

### 真机启动前验收

先运行只读预检。它只读取 `/state`，不发送 action：

```bash
python3 toolkits/realworld_check/validate_zadai_eef_control.py \
  preflight --url http://ROBOT_HOST:8010
```

确认双臂周围净空、急停和脚踏可用后，才运行 3 mm 往返测试：

```bash
python3 toolkits/realworld_check/validate_zadai_eef_control.py \
  small-step \
  --url http://ROBOT_HOST:8010 \
  --axis z \
  --distance-m 0.003 \
  --acknowledge-robot-clear
```

脚踏接管和 EEF 恢复测试会先发送当前测量位姿的 hold chunk，等待人工踩下并释放脚踏，再使用当前 `refresh_generation` 提交新的 EEF hold chunk：

```bash
python3 toolkits/realworld_check/validate_zadai_eef_control.py \
  intervention \
  --url http://ROBOT_HOST:8010 \
  --acknowledge-robot-clear
```

这些真机命令必须由现场人员执行；程序默认模式不会驱动机器人。

## 6. 将旧 Online Buffer 转换为 EEF

程序：[`convert_online_buffer_joint_to_eef.py`](convert_online_buffer_joint_to_eef.py)

```bash
python3 toolkits/marvin/convert_online_buffer_joint_to_eef.py \
  --src /home/zzx23457/code_xlb/RLinf/replay_buffers/zadai_200_rlt_stage2/rank_0 \
  --dst /home/zzx23457/code_xlb/RLinf/replay_buffers/zadai_200_rlt_stage2_eef/rank_0 \
  --mjcf /home/zzx23457/ws_apex/Apex_Deploy-main/robot_node/marvin_description/mjcf/matrix/m6_696.xml
```

程序将动作和 reference 字段写成 20D normalized EEF policy domain；`proprio` 保持 20D physical absolute EEF：

- `actions` 和 `forward_inputs/action`
- `curr_obs`、`next_obs` 中的 `proprio`、`ref_chunk`
- `forward_inputs` 中对应的 RLT transition 字段
- 与动作同形状的 `intervene_flags`、`versions`

奖励、终止标记和其他 RL 元数据保持不变。程序拒绝已存在的 `--dst`，并通过临时目录完成后再重命名。`--src` 始终只读。

> [!WARNING]
> 旧 online buffer 没有保存相机图像，因此无法使用新的 EEF Stage1 encoder 重新生成 `z_rl`。脚本会原样复制旧 `z_rl`，并在 `eef_buffer_manifest.json` 中记录限制。如果联合训练改变了 RLT encoder，转换后的旧 buffer 不能直接与新在线数据安全混用。只有复用兼容的旧 encoder，或找回与 transition 对齐的原始图像重新编码时，才可消除这一分布差异。

因此，新的 EEF Stage1 全参训练完成后，推荐先使用第 5 节从原始图像重新生成 demo buffer；旧 online buffer 转换结果仅作为有明确 encoder 兼容性依据时的迁移工具，不应默认混入新训练。

## 验证输出

### 验证 LeRobot EEF 数据

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("/home/zzx23457/Data/lerobot/zadai_200_eef_rot6d_rlinf_v21")
info = json.loads((root / "meta/info.json").read_text())
assert info["features"]["observation.state"]["shape"] == [20]
assert info["features"]["action"]["shape"] == [20]
assert len(list((root / "data").rglob("episode_*.parquet"))) == 200
print("EEF dataset OK")
PY
```

### 验证 Replay Buffer

```bash
python3 - <<'PY'
import json
from pathlib import Path
import torch

root = Path("/path/to/eef_buffer/rank_0")
metadata = json.loads((root / "metadata.json").read_text())
files = sorted(root.glob("trajectory_*.pt"))
assert len(files) == metadata["size"]
sample = torch.load(files[0], map_location="cpu", weights_only=False)
assert sample["actions"].shape[-1] % 20 == 0
assert sample["curr_obs"]["proprio"].shape[-1] == 20
assert sample["curr_obs"]["ref_chunk"].shape[-1] == 20
print("EEF replay buffer OK", metadata)
PY
```

## 常见问题

### 为什么数据集保存绝对 EEF，而不是直接保存增量？

绝对目标具有稳定的物理意义，便于 FK 对照、回放、重新定义增量和更换训练策略。body-frame 增量依赖 chunk 起点和训练时 horizon，属于模型输入表示，不适合作为唯一事实源。

### 人工标注是否依赖 Stage1？

不依赖。人工标注只读取视频和 episode 帧号。只有生成可直接训练的 RLT demo buffer 时，才需要 Stage1 checkpoint 计算 `z_rl` 和 `ref_chunk`。

### 能否把不同 Stage1 checkpoint 生成的 demo buffer 混合？

默认不要这样做。即使维度相同，`z_rl` 的数值空间也可能不同。应使用最终确定的 Stage1 checkpoint 重新物化所有离线 demo。

### 目标目录已存在怎么办？

选择一个新的版本化目录，例如：

```text
zadai_200_eef_demo_stage1_step_10000/rank_0
```

不要覆盖仍可能用于复现实验的旧目录。确认不再需要时，再由人工单独处理。

### Pinocchio 无法导入怎么办？

先在当前解释器验证：

```bash
python3 -c "import pinocchio; print(pinocchio.__version__)"
```

如果失败，请切换到 APEX 使用的 Python/conda 环境。数据集转换和 online buffer 转换必须使用同一份 MJCF 和一致的 joint/frame 定义。
