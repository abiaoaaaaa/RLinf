# Zadai / Marvin 数据与检查点盘点

盘点日期：2026-09-04

## 335-Episode Pi0.5 RLT Stage1（2026-09-07）

使用当前 335 条有效 episode 的数据集，不要使用包含黑腕相机异常数据的
354 条备份。两台机器运行同一脚本，各自读取本机数据和 JAX checkpoint；
不要在机器间传输数据集。

本地输入：

```text
/home/zzx23457/Data/lerobot/zadai_354
/home/zzx23457/Data/models/zadai_335_jax/149999
/home/zzx23457/Data/models/zadai_335_jax/norm_stats.json
```

服务器输入：

```text
/ssd/hhw/tianji/zadai_354
/ssd/hhw/openpi-hzh/checkpoints/pi05_hhw_tj_zadai_354_uniform/pi0.5fine_tuning_tj_zadai_354/149999
/ssd/hhw/tianji/zadai_354/norm_stats.json
```

先在两端分别设置输入路径并执行准备：

```bash
export ZADAI_335_SOURCE=/path/to/local/zadai_354
export ZADAI_335_JAX_CKPT=/path/to/local/149999
export ZADAI_335_NORM_STATS=/path/to/local/norm_stats.json
bash toolkits/marvin/prepare_zadai_335_rlt_stage1.sh prepare
```

该命令将视频硬链接到仓库内的 `artifacts/zadai_335/`，转换 Parquet 布局，
将 JAX 参数转换成 bfloat16 SafeTensors，并校验 335 episodes、424905 frames、
模型结构和 norm stats。它不会写入 OpenPI 源码目录。

只在服务器先跑一个优化步骤的 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash toolkits/marvin/prepare_zadai_335_rlt_stage1.sh smoke
```

smoke test 通过后启动完整 Stage1：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash toolkits/marvin/prepare_zadai_335_rlt_stage1.sh train
```

训练使用 `zadai_335_rlt_stage1_sft_openpi_pi05`，冻结 Pi0.5 主干，只训练
RLT token encoder/decoder。启动前应确认两端代码清单哈希一致，且服务器 GPU
0–3 空闲。

## 结论摘要

- 原始 Joint 数据存在，主要数据集为 200 episodes、16D action/state。
- EEF 数据存在，主要数据集为 200 episodes、20D action/state。
- EEF RL 区间标注存在，但当前没有发现已经导出的独立 EEF RLT demo buffer。
- Joint Stage2 模型和 online replay buffer 存在，最高发现 `global_step_270`。
- Joint Stage1 和 EEF Stage1 checkpoint 都存在。
- 当前 residual tracking 分支使用 `absolute_compat`，可以复用旧 Joint Stage2 actor 和 online buffer。

## 原始 Joint 数据

| 路径 | Episodes | Frames | Action / state | 备注 |
|---|---:|---:|---|---|
| `/home/zzx23457/Data/lerobot/zadai_total/zadai_200` | 200 | 307134 | 16D / 16D | 当前发现的完整 Joint 数据目录 |
| `/home/zzx23457/Data/lerobot/zadai_200_with_effort` | 200 | 307134 | 16D / 16D | 额外包含 velocity、effort |
| `/home/zzx23457/Data/lerobot/zadai_200.zip` | - | - | - | Joint 数据压缩包 |

其他 Joint 子集或合并数据包括 `8_18_zadai_50_with_effort`、`8_20_zadai`、`8_20_zadai_100_with_effort`、`8_21_zadai_50`、`9_3_zadai`、`9_3_zadai_with_effort` 和 `merged_all_bags`。

注意：配置/README 中的 `/home/zzx23457/Data/lerobot/zadai_200` 当前未发现；实际完整目录是 `zadai_total/zadai_200`。

## EEF 数据

| 路径 | Episodes | Frames | Action / state | 备注 |
|---|---:|---:|---|---|
| `/home/zzx23457/Data/lerobot/zadai_200_eef_rot6d` | 200 | 307134 | 20D / 20D | EEF v3 数据 |
| `/home/zzx23457/Data/lerobot/zadai_200_eef_rot6d_m6_696` | 200 | 307134 | 20D / 20D | M6-696 工具版本 |
| `/home/zzx23457/Data/lerobot/zadai_200_eef_rot6d_rlinf_v21` | 200 | 307134 | 20D / 20D | RLinf v2.1 训练副本 |
| `/home/zzx23457/Data/lerobot/zadai_200_eef_rot6d_rlinf_v21_m6_696` | 200 | 307134 | 20D / 20D | RLinf v2.1 + M6-696 |

对应统计文件为 `zadai_200_eef_norm_stats/` 和 `zadai_200_eef_norm_stats_m6_696/`；RL 标注文件为 `zadai_200_eef_rl_annotations.json` 和 `zadai_200_eef_rl_annotations_m6_696.json`。

## EEF RL 标注与 Demo Buffer

当前标注统计：

- 标注 episode 条目：135；含 RL 区间的 episode：133；RL 区间总数：133。
- RL 区间帧数约 38679；当前所有标注 outcome 都是 `success`。

标注文件不是 RLT demo buffer。当前没有发现导出的：

```text
/home/zzx23457/code_xlb/RLinf/replay_buffers/zadai_200_eef_demo/rank_0
```

`logs/zadai_eef_replay/` 和 `logs/zadai_eef_replay_ep0/` 是动作/数据验证日志，不是 RLinf `TrajectoryReplayBuffer`。

## Joint Stage1 / Stage2 checkpoint

Joint Stage1：

```text
/home/zzx23457/code_xlb/RLinf/logs/20260825-23:18:34-zadai_200_rlt_stage1_sft_openpi_pi05/
zadai_200_rlt_stage1/checkpoints/global_step_7000/actor
```

Joint Stage2 最高发现：

```text
/home/zzx23457/code_xlb/RLinf/logs/continue_20260827_zadai_200_rlt_stage2/
zadai_200_rlt_stage2_ac_mlp/checkpoints/global_step_270
```

EEF Stage1：

```text
/home/zzx23457/Data/checkpoints/zadai_eef_stage1_global_step_15000
/home/zzx23457/code_xlb/RLinf/checkpoints/zadai_200_eef_stage1_step15000
```

当前没有发现完整保存的 EEF Stage2 checkpoint，主要发现的是 EEF Stage2 runtime 日志。

## Joint Stage2 online replay

Joint Stage2 replay 以 SAC 组件形式保存在 Stage2 checkpoint 中，例如：

```text
/home/zzx23457/code_xlb/RLinf/logs/continue_20260827_zadai_200_rlt_stage2/
zadai_200_rlt_stage2_ac_mlp/checkpoints/global_step_270/actor/
sac_components/replay_buffer/rank_0
```

不同 snapshot 的 trajectory 数量包括：早期 `global_step_30` 的 31、280、381 条；`global_step_60` 的 341 或 500 条；`global_step_90` 到 `global_step_270` 每个 snapshot 约 500 条。

典型字段如下：

```text
actions                 [1, 1, 480]
forward_inputs/action    [1, 1, 480]
forward_inputs/z_rl      [1, 1, 2048]
forward_inputs/proprio   [1, 1, 16]
forward_inputs/ref_chunk [1, 1, 30, 16]
curr_obs / next_obs
rewards                  [1, 1, 30]
terminations / truncations / dones
intervene_flags / versions
```

这些是 Joint Stage2 RLT replay，不是普通 LeRobot 数据集。

## Online-only residual 训练

当前 Joint Stage2 配置移除了 demo buffer，只使用 online replay：

```yaml
algorithm:
  demo_buffer: null
```

建议恢复完整 Stage2 checkpoint，而不是只加载 actor：

```yaml
runner:
  resume_dir: /home/zzx23457/code_xlb/RLinf/logs/continue_20260827_zadai_200_rlt_stage2/
    zadai_200_rlt_stage2_ac_mlp/checkpoints/global_step_270
```

完整恢复会包含 actor、critic、target model、optimizer、训练步数和 checkpoint 内的 replay buffer。需要确认 `ZADAI_REPLAY_BUFFER_PATH` 与实际希望继续写入的 online buffer 路径一致。不要将 Joint replay 与 EEF replay 混用。

## Residual tracking 兼容策略

当前 Joint 配置使用：

```yaml
tracking_mode: absolute_compat
tracking_max_step_delta: 0.02
```

旧 actor 输出绝对 Joint target，转换为：

```text
residual[t] = actor_target[t] - ref[t]
```

然后执行递归 tracking。因此旧 replay 中的绝对 action 仍与当前 replay schema 兼容；新的 rollout 也继续保存绝对执行 action。

真正让 actor 从网络层直接输出 residual 的 `tracking_mode: residual` 需要重新定义 BC target，不能直接把旧 actor 输出按 residual 解释。
