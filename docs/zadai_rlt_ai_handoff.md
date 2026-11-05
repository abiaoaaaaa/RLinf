# Zadai RLT：新 AI 对话快速接手

> 远程训练统一入口为 `ssh zkrh-56`，RLinf 路径为
> `/ssd/xlb/code/RLinf`，公共硬件、存储和 GPU 安全规则见
> [`../../../REMOTE_SERVER.md`](../../../REMOTE_SERVER.md)。不要误用远端
> `/ssd/RLinf`，也不要把大文件写入根盘。

> 适用仓库：`/home/xlb/data/code/code_new/code_xlb/RLinf`
> 文档整理日期：2026-09-26；最近实验快照：2026-09-21
> 目的：让没有历史对话的新 AI 在几分钟内理解本机真机 RLT 项目、常用命令和用户习惯。

## 1. 新对话先做什么

不要仅凭本文直接修改或启动真机。先执行只读检查：

```bash
cd /home/xlb/data/code/code_new/code_xlb/RLinf
git status --short

sed -n '1,180p' docs/zadai_rlt_ai_handoff.md
sed -n '1,140p' examples/embodiment/config/zadai_439_rlt_stage2_ac_mlp_native.yaml
rg -n "num_action_chunks|ref_num_action_chunks|update_epoch|critic_actor_ratio|train_every_transitions|max_updates_per_train_step" \
  examples/embodiment/config/zadai_200_rlt_stage2_ac_mlp.yaml

find /home/xlb/data/logs -type d -path '*zadai_439*checkpoints/global_step_*' \
  -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -10
```

接手规则：

- 当前主仓库就是上面的 `RLinf`，不要误用其他 RLinf 副本。
- 工作树长期包含用户尚未提交的改动。禁止 `git reset --hard`、覆盖或清理无关文件。
- 以当前 YAML 和代码为行为真值；文档用于导航。发现冲突时先说明具体差异。
- 程序、配置、测试和文档可以纳入 Git；模型权重、数据集、Replay、原始 rollout 和运行日志不得纳入 Git。
- 真机动作有安全风险。除非用户明确要求，不代替用户启动、复位或发送机器人动作。
- “分析日志”默认只读；不要为了分析而重新运行训练或改写 Replay。

## 2. 当前默认实验

用户当前优先使用：**439 Stage1 + Native（非 RTC）+ residual MLP Actor**。

| 项目 | 当前值 |
| --- | --- |
| 主配置 | `zadai_439_rlt_stage2_ac_mlp_native` |
| Stage1 checkpoint | `/home/xlb/data/models/zadai_439_rlt_stage1/global_step_20000` |
| norm stats | `/home/xlb/data/models/zadai_439_rlt_stage1/assets/zadai_joint_clean_439/norm_stats.json` |
| 固定专家 buffer | `/home/xlb/data/replay_buffers/zadai_9_16_joint_expert_38_439/rank_0` |
| 在线 Replay | `/home/xlb/data/replay_buffers/zadai_439_rlt_stage2_native/rank_0` |
| 原始 rollout | `/home/xlb/data/replay_buffers/zadai_439_rlt_stage2_native_raw/rank_0` |
| 最近运行目录 | `/home/xlb/data/logs/zadai_439_rlt_stage2_native_fir_run1` |
| 最近已保存 checkpoint | `.../zadai_439_rlt_stage2_ac_mlp_native/checkpoints/global_step_390` |
| VLAHost | 本机 `http://127.0.0.1:8010` |
| 机器人 | `192.168.14.190`，本机网卡曾配置 `192.168.14.100/24` |

上表的 checkpoint 和样本数是**快照**，使用前必须重新查找最新目录。2026-09-21
停止时，在线 Replay `metadata.json` 为 1174 条；`global_step_390` 内的
`update_step=916`，未保存的运行日志一度到 `update_step=948`。

不要混淆：

- `runner global step`：一次外层采集/训练循环；checkpoint 名使用它。
- `update_step`：Actor/Critic 已执行的梯度更新计数，单独持久化。
- 低层控制 step：30 Hz 的一个动作点。
- 一个 transition：一个完整的 30-step action chunk，约 1 秒。
- `openpi.num_steps=10`：flow 推理迭代次数，不是 chunk 长度。

## 3. 当前动作和训练逻辑

当前 439 Native 配置的关键事实：

```text
OpenPI/Pi0.5 产生 reference chunk（30 x 16）
        +
RLT MLP Actor 产生有界 residual（30 x 16）
        ↓
手臂 residual 上限 ±0.03 normalized；两个夹爪 residual=0
        ↓
[1, 2, 3, 2, 1] / 9 centered FIR，仅平滑 chunk 内 residual
        ↓
ZadaiEnv 转物理 Joint 并做安全限制
        ↓
POST /action_chunk 给 VLAHost，以 30 Hz 执行30步
```

Actor 输入是 Stage1 的 `z_rl`（2048维）、当前 proprio（16维）和 OpenPI
`ref_chunk`（30×16）。当前是真正的 residual policy，不是旧
`absolute_compat` 输出语义。

主要超参数以 YAML 为准，目前 Native 覆盖值包括：

- `gamma=0.99864`，`tau=0.005`，`alpha=0`；
- Actor LR `3e-5`，Critic LR `3e-4`；
- BC weight `5.0`；Q weight 在满足在线成功数后最高 `0.05`；
- 至少250条 critical transition 才 warmup，初始200次更新只做 BC；
- 每新增4条 transition，调度 `update_epoch=8` 次 Critic 更新；
- `critic_actor_ratio=4`，所以通常对应8次 Critic、2次 Actor 更新；
- 单个 runner step 最多运行20次 Critic 更新，未执行预算留到后续循环。

固定专家 buffer 和在线 Replay 以 50:50 组成训练 batch。人工干预 transition
同时进入在线 Replay 和运行时 demo buffer；固定专家目录本身不会被在线追加覆盖。

FIR 只解决 chunk 内高频正负摆动，不处理跨 chunk 边界。2026-09-21 的实测分析：

- Actor 原始 chunk 加速度 RMS 约 `27–30 rad/s²`；FIR 后约 `3.9–4.3 rad/s²`；
- 最新 Actor 相对 OpenPI 平均偏移约 `0.0248 rad`；
- 新增26个完整 RL 回合中9个成功，但全部含人工干预；完全自主8回合、成功0；
- 曾出现一次约 `0.363 rad` 的跨 chunk 边界跳变，FIR 不会修复它；
- Q 项和 BC 项的加权量已接近，不能仅因 `q_weight=0.05` 就认为 Q 影响很小。

后续分析应分别报告：Actor raw、FIR/routed、实际 executed、跨 chunk boundary、
是否人工干预、自主成功率；不要只看聚合 reward。

## 4. ROS、Apex 和 VLAHost

ROS 2 Humble 必须使用系统 Python 3.10，不能在 conda 或 RLinf Python 3.11
虚拟环境里启动。每个 ROS 终端先清理：

```bash
conda deactivate 2>/dev/null || true
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash
source /home/xlb/data/code/code_new/Apex_Deploy/install/setup.bash
```

启动 Apex 硬件、相机、脚踏和录制：

```bash
ros2 launch node_manager bringup_all_dm_m696.launch.py \
  use_rviz:=false \
  webrtc:=false \
  footpad_device_path:=/dev/input/event5 \
  bag_storage_root:=/home/xlb/data/Data/rosbag/9_9_zadai
```

另一个清洁 ROS 终端启动本项目 VLAHost：

```bash
ros2 launch vlahost vlahost_server.launch.py \
  host:=0.0.0.0 \
  port:=8010 \
  end_effector:=gripper

curl -sS http://127.0.0.1:8010/state
```

RLinf 与 VLAHost 同机，因此默认 URL 是 `127.0.0.1:8010`，不是机器人 IP。
`192.168.14.190` 是机器人地址。曾出现的 `/go_home` 502 应先检查本机 VLAHost、
ROS/Apex 和机器人网络，而不是把 RLinf URL 改成机器人 IP。

脚踏录制可用但 RLT 干预无效时，优先检查：ROS 终端是否误用了 conda Python、
footpad reader 是否发布正确 topic、VLAHost intervention session 是否启用。终端打印
`^[[1;2D` 只是键盘转义序列，不证明脚踏事件进入了 VLAHost。

## 5. RLinf 通用环境和键盘

RLinf 使用独立的 Python 3.11 环境：

```bash
cd /home/xlb/data/code/code_new/code_xlb/RLinf
source .venv_openpi/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ZADAI_VLAHOST_URL=http://127.0.0.1:8010
```

不设置 `PYTHONPATH` 会出现 `ModuleNotFoundError: No module named 'rlinf'`。

默认 RLT 键盘：

```text
/dev/input/by-id/usb-HS6209_A4tech_2.4G_Wireless_Device-event-kbd
```

需要时授权或覆盖：

```bash
sudo setfacl -m u:"$USER":r \
  /dev/input/by-id/usb-HS6209_A4tech_2.4G_Wireless_Device-event-kbd
export RLINF_KEYBOARD_DEVICE=/dev/input/by-id/其他键盘-event-kbd
```

| 按键 | 当前行为 |
| --- | --- |
| `b` | 开始 critical/RLT segment，开始记录 transition；learner ready 后 Actor 接管。 |
| `c` | 成功，reward=1，结束 segment，返回 OpenPI；不自动 reset。 |
| `a` | 失败，reward=0，结束 segment，返回 OpenPI；不自动 reset。 |
| `q` | 中止，reward=0，结束 segment，返回 OpenPI；不自动 reset。 |
| `Ctrl+1` | 安全 chunk 边界暂停，打开夹爪、go home、等待5秒，再恢复推理。 |

脚踏人工干预与上述阶段键是两套信号：脚踏不会替代 `b/c/a/q`。一次长 OpenPI
rollout 内可以多次执行 `b → a/c/q → 人工调整 → b`，提高采样效率。

## 6. 正式续训命令模板

先确认最新完整 checkpoint；不要只加载 `actor/model_state_dict`，完整恢复还包括
Critic、target、optimizer、`update_step` 和成功计数。以下是最近一次已验证的恢复模板：

```bash
cd /home/xlb/data/code/code_new/code_xlb/RLinf
source .venv_openpi/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ZADAI_VLAHOST_URL=http://127.0.0.1:8010
export ZADAI_439_NATIVE_REPLAY_BUFFER_PATH=/home/xlb/data/replay_buffers/zadai_439_rlt_stage2_native
export ZADAI_439_NATIVE_RAW_ROLLOUT_PATH=/home/xlb/data/replay_buffers/zadai_439_rlt_stage2_native_raw
export ZADAI_RLT_RUNTIME_LOG=/home/xlb/data/logs/zadai_439_rlt_stage2_native_fir_run2/rlt_runtime.jsonl
export ZADAI_RLT_ROUTE_LOG="$ZADAI_RLT_RUNTIME_LOG"
mkdir -p /home/xlb/data/logs/zadai_439_rlt_stage2_native_fir_run2

python examples/embodiment/train_embodied_agent.py \
  --config-path "$PWD/examples/embodiment/config" \
  --config-name zadai_439_rlt_stage2_ac_mlp_native \
  runner.resume_dir=/home/xlb/data/logs/zadai_439_rlt_stage2_native_fir_run1/zadai_439_rlt_stage2_ac_mlp_native/checkpoints/global_step_390 \
  runner.max_steps=1000 \
  runner.save_interval=10 \
  runner.val_check_interval=-1 \
  runner.logger.log_path=/home/xlb/data/logs/zadai_439_rlt_stage2_native_fir_run2 \
  env.train.max_steps_per_rollout_epoch=300 \
  algorithm.replay_buffer.auto_save=true \
  algorithm.replay_buffer.auto_resume=true \
  algorithm.replay_buffer.auto_save_path=/home/xlb/data/replay_buffers/zadai_439_rlt_stage2_native \
  rollout.enable_torch_compile=false
```

每次实验使用新的 `runner.logger.log_path` 和 runtime JSONL 路径，但继续训练时可以
复用同一在线 Replay 与 raw archive。若要严格 A/B 对照，应先复制/冻结 Replay，或
给新实验设置新目录。

## 7. 只测试 OpenPI 基线

下面命令禁止训练、checkpoint 和 Replay 恢复；`b` 也不会让未就绪 Actor 接管：

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path "$PWD/examples/embodiment/config" \
  --config-name zadai_439_rlt_stage2_ac_mlp_native \
  runner.max_steps=1000 \
  runner.save_interval=-1 \
  runner.val_check_interval=-1 \
  runner.logger.log_path=/tmp/zadai_439_openpi_native_smoke \
  env.train.max_steps_per_rollout_epoch=300 \
  algorithm.rlt_schedule.warmup_min_size=999999 \
  algorithm.rlt_schedule.warmup_post_collect_updates=999999 \
  algorithm.replay_buffer.auto_save=false \
  algorithm.replay_buffer.auto_resume=false \
  algorithm.demo_buffer=null \
  algorithm.raw_rollout_archive.enabled=false \
  rollout.enable_torch_compile=false
```

RTC 是另一个可选配置 `zadai_439_rlt_stage2_ac_mlp_rtc`。当前正式路线默认 Native；
除非用户明确要求，不把 RTC 数据目录和 Native 数据目录混用。

## 8. 专家数据标注与导出

当前 Joint 源数据：

```text
/home/xlb/data/Data/lerobot/9_16_zadai_joint_clean/videos
```

标注按键：空格播放/暂停，`A/D`逐帧，`J/L`一秒，`[`开始区间，`]`结束区间，
`S`成功，`F`失败，`E`整回合作为成功专家，`N`保存下一条，`B`上一条，`X`清空，
`Q/Esc`保存退出。

```bash
python toolkits/marvin/label_eef_rl_demo_buffer.py annotate \
  --dataset /home/xlb/data/Data/lerobot/9_16_zadai_joint_clean/videos \
  --annotations /home/xlb/data/annotations/zadai_9_16_joint_expert_38.json \
  --max-expert-episodes 38
```

导出439专用专家 buffer：

```bash
python toolkits/marvin/label_eef_rl_demo_buffer.py export \
  --dataset /home/xlb/data/Data/lerobot/9_16_zadai_joint_clean/videos \
  --annotations /home/xlb/data/annotations/zadai_9_16_joint_expert_38.json \
  --output /home/xlb/data/replay_buffers/zadai_9_16_joint_expert_38_439/rank_0 \
  --stage1-checkpoint /home/xlb/data/models/zadai_439_rlt_stage1/global_step_20000 \
  --norm-stats /home/xlb/data/models/zadai_439_rlt_stage1/assets/zadai_joint_clean_439/norm_stats.json \
  --action-space joint \
  --expected-episodes 38 \
  --chunk-size 30 \
  --chunk-stride 5 \
  --ref-chunk-size 50 \
  --num-steps 10 \
  --device cuda
```

`chunk-stride=5` 表示每隔5帧产生一个重叠训练样本，不会把执行 chunk 改成5步；
执行和 Actor 输出仍是30步。`num-steps=10` 是 Pi0.5 flow 推理步数。

## 9. 数据与日志分别保存什么

| 位置 | 内容 | 能否跨模型复用 |
| --- | --- | --- |
| 在线 Replay `...stage2_native/rank_0` | `z_rl`、proprio、ref/action、reward/done、干预标志；轻量训练数据 | 绑定 Stage1 encoder，不能直接换编码器 |
| 固定专家 buffer `...expert_38_439/rank_0` | 由439 Stage1物化的专家 transition | 仅供匹配的439编码器 |
| raw archive `...native_raw/rank_0` | 三相机 JPEG、状态、Pi0.5完整输出、reference、Actor raw、routed、executed、干预、reward、RTC元数据 | 可重新编码，长期复用价值最高 |
| `metrics.log`/TensorBoard | runner、Replay、loss、Q、BC和时间指标 | 诊断用 |
| `rlt_runtime.jsonl` | 路由、Joint处理前后、VLAHost状态、chunk安全限制、训练事件 | 动作连续性和故障诊断 |
| checkpoint | Actor、Critic、target、优化器、训练状态及快照 Replay | 完整续训 |

`algorithm.demo_buffer.auto_save=true` 并不等于把所有在线数据追加进固定专家目录。
当前在线干预会进入运行时 demo buffer并随 checkpoint 保存；原始固定专家目录应保持不变。

## 10. 常见故障快速判断

| 现象 | 首先检查 |
| --- | --- |
| `No module named rlinf` | 是否在仓库根目录并设置 `PYTHONPATH` |
| Hydra 提示 key 不存在 | 临时新增字段需 `+key=value`；更应先确认代码是否真的读取该字段 |
| `/go_home` 返回502 | 本机8010 VLAHost、ROS环境、Apex节点、机器人网络，不要把URL改成机器人IP |
| ROS `rclpy._rclpy_pybind11`/cpython-314 | ROS终端仍在 conda；清理环境后使用系统Python3.10 |
| 机器人持续推理但不动 | VLAHost ownership/intervention heartbeat、policy chunk是否 active、Joint/EEF模式 |
| 脚踏无效 | footpad topic、device path/key code、VLAHost intervention session、ROS domain一致性 |
| `KeyError: rtc_chunk_id` | 新旧 Replay schema混合；当前代码已有兼容处理，先确认运行代码和测试 |
| 感觉切换时跳变 | 分开检查 chunk内 FIR 与跨chunk boundary；查看 `rlt_route_action` 和 `action_chunk_submit` |
| 没有感觉到 RLT 接管 | `ready_for_online`、`update_step`、是否按 `b`、`actor_switch_rate` |

## 11. 新 AI 分析日志时的最低输出要求

回答“分析最新数据”时至少给出：

1. 数据时间范围、是否仍在运行、最新 runner step/checkpoint/update step；
2. Replay 新增条数、完整回合数、成功数、含干预成功与自主成功分别统计；
3. Actor raw、routed/FIR、executed 的 residual、单步变化、加速度和跨 chunk 边界；
4. Q/BC 的原始与加权量、梯度、Critic loss，不能只报 `q_weight`；
5. 安全裁剪次数、推理耗时、30 Hz实际频率；
6. 明确区分事实、推测和下一步建议。

常用只读入口：

```text
运行指标：<log_path>/metrics.log
TensorBoard：<log_path>/tensorboard/
动作细节：<log_path>/rlt_runtime.jsonl
训练样本：/home/xlb/data/replay_buffers/zadai_439_rlt_stage2_native/rank_0
完整轨迹：/home/xlb/data/replay_buffers/zadai_439_rlt_stage2_native_raw/rank_0
```

## 12. 深入阅读顺序

1. [`zadai_277_439_rlt_runbook.md`](zadai_277_439_rlt_runbook.md)：四种配置、运行和数据目录。
2. [`zadai_277_rlt_training_architecture.md`](zadai_277_rlt_training_architecture.md)：runner、warmup、Replay和梯度更新结构。
3. 当前配置：`examples/embodiment/config/zadai_439_rlt_stage2_ac_mlp_native.yaml`及其父配置。
4. 核心实现：`rlinf/algorithms/rlt/route.py`、`transition.py`、
   `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py`、`rlinf/workers/env/env_worker.py`、
   `rlinf/envs/realworld/zadai/zadai_env.py`。
