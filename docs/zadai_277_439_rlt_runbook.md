# Zadai 277/439 Stage2 RLT 运行手册

本文记录 277、439 两个 Stage1 模型在 Stage2 RLT 中使用 RTC 优化或原生同步 OpenPI 推理的配置与启动方式。

新 AI 对话或当前 `/home/xlb` 真机环境请先读
[`zadai_rlt_ai_handoff.md`](zadai_rlt_ai_handoff.md)。该文档集中记录当前默认路线、
实际路径、常用命令、最近实验快照和接手检查；本文保留较完整的多配置说明。

## 1. 四种配置

| Stage1 模型 | 推理模式 | Hydra 配置名 |
| --- | --- | --- |
| 439 | RTC 异步优化 | `zadai_439_rlt_stage2_ac_mlp_rtc` |
| 439 | 原生同步 | `zadai_439_rlt_stage2_ac_mlp_native` |
| 277 | RTC 异步优化 | `zadai_277_rlt_stage2_ac_mlp_rtc` |
| 277 | 原生同步 | `zadai_277_rlt_stage2_ac_mlp_native` |

四份配置均继承 `zadai_200_rlt_stage2_ac_mlp.yaml`，共用相同的 Actor、Critic、warmup、采样和更新参数。

RTC 版本启用：

- 非 critical chunk 异步预取；
- 10 步推理延迟元数据；
- 50 步完整动作规划中的过期前缀跳过；
- approximate overlap/flow guidance；
- 非 critical trajectory 空载同步，避免传输无用完整轨迹。

Native 版本关闭 RTC、异步预取和非 critical trajectory 跳过，使用同步 OpenPI 推理，作为对照基线。

## 2. 模型与归一化文件

### 439

```text
checkpoint:
/home/xlb/data/models/zadai_439_rlt_stage1/global_step_20000

norm_stats:
/home/xlb/data/models/zadai_439_rlt_stage1/assets/zadai_joint_clean_439/norm_stats.json
```

可用环境变量覆盖：

```bash
export ZADAI_439_STAGE1_CHECKPOINT=/path/to/global_step_20000
export ZADAI_439_NORM_STATS_PATH=/path/to/norm_stats.json
```

### 277

```text
checkpoint:
/home/xlb/data/checkpoints/zadai_277_rlt_stage1/global_step_20000/actor

norm_stats:
/home/xlb/data/checkpoints/zadai_277_rlt_stage1/norm_stats.json
```

可用环境变量覆盖：

```bash
export ZADAI_277_STAGE1_CHECKPOINT=/path/to/actor
export ZADAI_277_NORM_STATS_PATH=/path/to/norm_stats.json
```

## 3. 通用启动环境

先启动机器人、相机和本机 VLAHost，并确认：

```bash
curl -sS http://127.0.0.1:8010/state
```

然后在 RLinf 终端执行：

```bash
cd /home/xlb/data/code/code_new/code_xlb/RLinf
source .venv_openpi/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

RLinf 默认优先监听专用的 HS6209 A4tech 2.4G 键盘：

```bash
/dev/input/by-id/usb-HS6209_A4tech_2.4G_Wireless_Device-event-kbd
```

首次插入或重启后，如果当前用户无读取权限，执行：

```bash
sudo setfacl -m u:"$USER":r \
  /dev/input/by-id/usb-HS6209_A4tech_2.4G_Wireless_Device-event-kbd
```

如需临时改用其他键盘，仍可覆盖：

```bash
export RLINF_KEYBOARD_DEVICE=/dev/input/by-id/其他设备-event-kbd
```

## 4. 正式 Stage2 RLT 启动命令

### 4.1 439 + RTC

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path "$PWD/examples/embodiment/config" \
  --config-name zadai_439_rlt_stage2_ac_mlp_rtc \
  runner.logger.log_path=/home/xlb/data/logs/zadai_439_rlt_stage2_rtc
```

### 4.2 439 + Native

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path "$PWD/examples/embodiment/config" \
  --config-name zadai_439_rlt_stage2_ac_mlp_native \
  runner.logger.log_path=/home/xlb/data/logs/zadai_439_rlt_stage2_native
```

### 4.3 277 + RTC

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path "$PWD/examples/embodiment/config" \
  --config-name zadai_277_rlt_stage2_ac_mlp_rtc \
  runner.logger.log_path=/home/xlb/data/logs/zadai_277_rlt_stage2_rtc
```

### 4.4 277 + Native

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path "$PWD/examples/embodiment/config" \
  --config-name zadai_277_rlt_stage2_ac_mlp_native \
  runner.logger.log_path=/home/xlb/data/logs/zadai_277_rlt_stage2_native
```

### 4.5 439 Native 的 residual Actor 说明

`zadai_439_rlt_stage2_ac_mlp_native` 现在使用真正的 residual policy：

```text
最终动作 = clip(OpenPI reference + 0.03 * tanh(Actor residual), -1, 1)
```

14 个手臂关节的 residual 上限为归一化 `0.03`，两个夹爪为 `0`；新 Actor
零初始化，因此训练开始时严格复现 OpenPI。前 200 次更新只做 BC；即使更新数达到
阈值，在累计 10 个在线成功 episode 之前 Q 权重仍为 0，随后用 10 个在线成功
episode 将 Q 权重渐增到 `0.05`。计数随 checkpoint 保存和恢复。

不要用 `runner.resume_dir` 加载旧的 `absolute_compat` Stage2 actor checkpoint；两者
输出语义不同。原专家 buffer 和在线 replay 保存的都是绝对执行动作及 reference，
可以继续作为 residual Actor 的训练目标复用。首次 residual 训练应新建日志目录，但
可以保留配置中的 `replay_buffer.auto_resume=true` 来恢复已有 replay。

### 4.6 不接真机完成 residual Actor warmup

以下入口只创建 ActorGroup，读取现有在线 replay 与专家 buffer；不会创建 EnvWorker、
RolloutWorker，也不会访问 VLAHost 或机器人：

```bash
cd /home/xlb/data/code/code_new/code_xlb/RLinf
source .venv_openpi/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python examples/embodiment/train_offline_rl.py \
  --config-path "$PWD/examples/embodiment/config" \
  --config-name zadai_439_rlt_stage2_ac_mlp_native_offline_warmup \
  runner.logger.log_path=/home/xlb/data/logs/zadai_439_rlt_offline_warmup
```

它从 `update_step=0` 训练到 `200`，每轮最多执行 20 次更新，并保存：

```text
/home/xlb/data/logs/zadai_439_rlt_offline_warmup/
  zadai_439_rlt_stage2_residual_offline_warmup/
  checkpoints/global_step_200/
```

若 replay/demo 未达到配置的最小样本数，入口会报错退出，不会保存空 warmup
checkpoint。随后真机训练使用该目录作为 `runner.resume_dir`。

## 5. Replay 与专家 Buffer 隔离

四个在线 replay 目录彼此独立：

```text
/home/xlb/data/replay_buffers/zadai_439_rlt_stage2_rtc
/home/xlb/data/replay_buffers/zadai_439_rlt_stage2_native
/home/xlb/data/replay_buffers/zadai_277_rlt_stage2_rtc
/home/xlb/data/replay_buffers/zadai_277_rlt_stage2_native
```

专家 buffer 按 Stage1 模型分别生成，因为其中的编码特征必须与对应 Stage1 编码器一致：

```text
439: /home/xlb/data/replay_buffers/zadai_9_16_joint_expert_30_439/rank_0
277: /home/xlb/data/replay_buffers/zadai_9_16_joint_expert_30/rank_0
```

正式训练前必须确认相应目录已生成。不要让 277 和 439 共用编码后的专家 buffer。

在线 replay 路径可分别通过以下变量覆盖：

```bash
export ZADAI_439_RTC_REPLAY_BUFFER_PATH=/path/to/replay
export ZADAI_439_NATIVE_REPLAY_BUFFER_PATH=/path/to/replay
export ZADAI_277_RTC_REPLAY_BUFFER_PATH=/path/to/replay
export ZADAI_277_NATIVE_REPLAY_BUFFER_PATH=/path/to/replay
```

## 6. 原始在线 Rollout 归档

四份配置默认还会把**每一个 chunk**（包括非 critical/OpenPI 阶段）写入独立的原始归档。它不替代训练 Replay：训练仍从轻量编码 Replay 采样，原始归档用于离线重编码、诊断和以后更换模型。

默认目录：

```text
/home/xlb/data/replay_buffers/zadai_439_rlt_stage2_rtc_raw/rank_0
/home/xlb/data/replay_buffers/zadai_439_rlt_stage2_native_raw/rank_0
/home/xlb/data/replay_buffers/zadai_277_rlt_stage2_rtc_raw/rank_0
/home/xlb/data/replay_buffers/zadai_277_rlt_stage2_native_raw/rank_0
```

每个目录包含 `manifest.json`、`index.jsonl` 和 `chunks/chunk_*.pt`。每个 chunk 保存：

- chunk 起点和终点的机器人 Joint 状态、任务文本、top/wrist_L/wrist_R JPEG 图像；
- `pi05_model_output`：Pi0.5 未裁剪的完整模型输出（通常 50 步，仍处在模型输出域）；
- `pi05_reference`：完成输出变换并跳过 RTC 过期前缀后，实际参与路由的 reference 动作；
- `rlt_actor_raw`、`routed`、经过环境安全限制后的 `executed` 动作；
- `rl_phase_requested`、`rlt_actor_active`、干预标志和 `intervention` 动作；
- reward、done/termination/truncation、RTC 延迟和推理耗时元数据。

图像 JPEG 压缩与 `.pt` 写入在后台线程执行；队列满时会等待而不会丢数据，并在每个 runner step 结束时强制 flush。可分别覆盖目录：

```bash
export ZADAI_439_RTC_RAW_ROLLOUT_PATH=/path/to/raw
export ZADAI_439_NATIVE_RAW_ROLLOUT_PATH=/path/to/raw
export ZADAI_277_RTC_RAW_ROLLOUT_PATH=/path/to/raw
export ZADAI_277_NATIVE_RAW_ROLLOUT_PATH=/path/to/raw
```

临时测试若不希望落盘，启动命令增加：

```bash
algorithm.raw_rollout_archive.enabled=false
```

## 7. 仅测试推理

如果专家 buffer 尚未生成，只想比较真机推理，在任一启动命令末尾添加：

```bash
  algorithm.demo_buffer=null \
  algorithm.replay_buffer.auto_save=false \
  algorithm.replay_buffer.auto_resume=false \
  algorithm.rlt_schedule.warmup_min_size=999999 \
  algorithm.rlt_schedule.warmup_post_collect_updates=999999 \
  runner.save_interval=-1 \
  runner.val_check_interval=-1
```

这会保持 OpenPI reference policy 控制机器人，不执行 Stage2 Actor/Critic 更新，也不读写 replay。

## 8. RLT 键盘阶段切换

当前使用 `keyboard_reward_wrapper: rlt_policy_switch`，监听 Linux evdev 键盘事件，不依赖终端窗口焦点。

| 按键 | 行为 |
| --- | --- |
| `b` | 进入 critical 阶段并开始记录在线 transition；warmup 完成前仍由 OpenPI reference 控制，完成后切换到 RLT Actor。 |
| `c` | critical 段成功，奖励为 1，结束该 RLT segment 并返回 OpenPI reference，不重置机器人。 |
| `a` | critical 段失败，奖励为 0，结束该 RLT segment 并返回 OpenPI reference，不重置机器人。 |
| `q` | 中止当前 critical 段，奖励为 0，结束该 RLT segment 并返回 OpenPI reference，不重置机器人。 |
| `Ctrl+1` | 退出 critical 段（若已进入则以奖励 0 结束），暂停新推理，打开双夹爪，执行 `go_home`，额外等待 5 秒后从回零观测恢复 OpenPI 推理。 |

`a`、`c`、`q` 只在 Replay 中写入当前 RLT segment 的 terminal，不会产生物理 episode 的 `done`。因此一次长 OpenPI rollout 中可以重复执行：

```text
b 开始 RLT segment
  -> a/c/q 结束 segment
  -> OpenPI/人工干预调整机器人状态
  -> b 开始下一个 RLT segment
```

只有环境自身超时、异常终止或外部停止才会结束物理 episode。按键只会在下一个策略 chunk 边界改变策略控制，`a`、`q` 不是机器人硬件急停。

`Ctrl+1` 也不是硬件急停。它按以下安全顺序执行：

```text
停止进入新的 RLT critical 段
  -> 阻止新的 RTC 异步预取
  -> 到达没有旧预取动作的 chunk 边界
  -> POST /hold
  -> POST /action 打开双夹爪（0.0）
  -> POST /go_home
  -> 等待 reset_wait_s=5.0 秒
  -> 读取新的 /state 并恢复 OpenPI 推理
```

如果按键时 RTC 已经生成了下一 chunk，为避免回零后误执行旧观测对应的动作，系统会先消耗这个已预取 chunk，并禁止它再次预取，然后在下一个边界回零。因此从按键到开始回零通常可能多等待至多约一个 30 步 chunk（30 Hz 时约 1 秒）。Native 模式或尚未预取时，在当前 chunk 边界直接执行。

## 9. 启动前检查

```bash
test -f /home/xlb/data/models/zadai_439_rlt_stage1/global_step_20000/actor/model_state_dict/full_weights.pt
test -f /home/xlb/data/models/zadai_439_rlt_stage1/assets/zadai_joint_clean_439/norm_stats.json
test -f /home/xlb/data/checkpoints/zadai_277_rlt_stage1/global_step_20000/actor/model_state_dict/full_weights.pt
test -f /home/xlb/data/checkpoints/zadai_277_rlt_stage1/norm_stats.json
```

正式RL训练还应检查相应专家 buffer：

```bash
test -d /home/xlb/data/replay_buffers/zadai_9_16_joint_expert_30_439/rank_0
test -d /home/xlb/data/replay_buffers/zadai_9_16_joint_expert_30/rank_0
```

## 10. Joint 动作安全限制与日志

当前发送给 VLAHost 前的物理 Joint 限制为：

```text
相邻控制点：每个机械臂关节最多变化 0.22 rad
单个 30 步 chunk：每个机械臂关节相对 chunk 起点最多偏移 3.00 rad
```

这两个阈值来自 `9_16_zadai_joint_clean` 的 79 个 episode、62,245 帧：
示教最大相邻动作变化为 0.197 rad，最大 30 步累计变化为 2.721 rad；
阈值向上取整，分别保留约 12% 和 10% 余量。统计严格限制在 episode 内，
不把回合切换时的复位跳变计入。

Stage2 RLT actor 在到达物理限位前还会经过参考动作跟踪和残差平滑：

```text
递归跟踪单步上限：0.07 normalized，约 0.22 rad
相对 OpenPI 最大残差：0.10 normalized，约 0.314 rad
残差相邻步变化上限：0.02 normalized，约 0.063 rad
夹爪残差：0，仍完全沿用 OpenPI
```

残差限制描述的是“RL 相对 OpenPI 能改多少”，数据集没有直接对应的真值，
因此它不使用示教动作最大值硬匹配；最终仍由上述物理 joint 限位兜底。

每次 `action_chunk_submit` 会在 `rlt_runtime.jsonl` 中追加以下聚合诊断，不保存三路图像或完整逐步动作：

```text
safety_clip_step_count
safety_clip_step_fraction
safety_clip_count_per_dim
max_requested_step_delta_per_dim
max_sent_step_delta_per_dim
max_requested_chunk_delta_per_dim
max_sent_chunk_delta_per_dim
max_abs_safety_delta_per_dim
```

16 维数组的顺序为左臂 7 关节、右臂 7 关节、左夹爪、右夹爪。前 14 维 Joint 的单位是 rad；最后两维夹爪使用 `[0, 1]` 位置单位，且当前不应用 Joint delta 限制。旧日志不会补充这些字段，必须在修改后的代码上重新测试。
