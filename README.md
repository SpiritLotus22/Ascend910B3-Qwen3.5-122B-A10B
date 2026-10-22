# Qwen3.5-122B-A10B ± DFlash 对比测试

单机 8 卡 910B3 / sglang 0.5.22 / CANN 9.0.0。

## 结构

```
main.sh              唯一入口
scripts/
  common.sh          共享配置与函数（被 source，不单独执行）
  serve_baseline.sh  启动不带 DFlash 的服务
  serve_dflash.sh    启动带 DFlash 的服务
  bench.sh           压测当前服务
  stop.sh            停服务并等显存释放
  campaign.sh        端到端跑完整个对比实验
  compare.py         多轮聚合对比
logs/    results/
```

`serve_baseline.sh` 和 `serve_dflash.sh` 刻意保持为两个文件：它们的 diff 恰好就是
实验变量本身，「两组是否只差投机解码参数」可以靠读 diff 直接确认。

## 用法

```bash
./main.sh help

# 手动单步
./main.sh serve dflash 8
./main.sh status
./main.sh bench
./main.sh stop

# 端到端跑完（推荐，脱机运行）
nohup ./main.sh campaign > logs/campaign.out 2>&1 &

# 对比
./main.sh compare 'results/baseline-*' 'results/dflash-b8-*' 'results/dflash-b16-*'

# 清理（永不删 results/）
./main.sh clean logs
./main.sh clean kernel
```

## 控制变量

两个配置之间**唯一**允许不同的是 `--speculative-*` 参数。其余固定在 `common.sh`：
TP=8、`mem-fraction-static=0.8`、`max-running-requests=32`、
`mamba-radix-cache-strategy=extra_buffer`、`mamba-ssm-dtype=bfloat16`、attention backend。

`bench.sh` 开跑前会调 `/get_server_info` **自动校验**这些参数与预期一致，不一致直接退出。
这类差异不会报错，只会静默污染结论，所以必须自动查。

`--max-running-requests 32` 必须显式写：投机解码会把它默认重置成 48
（`speculative_hook.py:1000`）。

压测侧固定：`--seed 42`、`--random-range-ratio 1.0`、`--warmup-requests 4`、每轮 `--flush-cache`。

### 只需要两个对照组

已核对 0.5.22 源码并经实测确认（`./main.sh status` 可见）：

- overlap scheduler **不会**因投机解码而关闭。只有 `device=cpu`
  （`speculative_hook.py:65-73`）和 `mps`（`platform_hook.py:48`）才强制关，NPU 不在其中。
- mixed chunked prefill 默认就是关的（`fields/schedule.py:202`），且 DFLASH 在
  `supports_mixed_chunk()` 白名单里（`spec_info.py:152`）。

官方文档里「disables overlap scheduler & mixed chunked prefill」是 EAGLE 时代的旧约束，
本版本已不成立。所以 baseline 和 DFlash 调度条件天然一致，不需要额外的 no-overlap 组。

## 数据集说明

默认用 `random`，但要清楚它测的是什么：

- `random` 采样**真实 ShareGPT 文本**，不是随机 token。
- 当 prompt 短于目标长度时会**平铺重复**同一段 token（`datasets/random.py:130-134`）。
  ShareGPT 典型 prompt 长 100–300 token，而默认 `IN_LEN=1024`，所以输入含大量重复内容。
  重复文本极易被 draft 预测，**实测接受长度偏乐观**，报告中须注明。
- 真正的对抗下界是 `random-ids`（伪随机 token ramp）。
- 所有数据集默认强制 `ignore_eos`（`serving.py:311` 等四处），每请求精确生成到
  `output_len`，跨配置可比。

后续可补：`sharegpt`（真实文本、输出长度取自数据集，需下载约 650MB）、
`generated-shared-prefix`（共享前缀场景）、`speed-bench`（专为投机解码设计，带
low/high entropy 分类，但仓库缺 `prepare_speed_bench.sh`，JSONL 需自造）。

不适用：`mmmu`/`image` 需多模态 processor；`mooncake` 的 prompt 是 `"hi hi hi…"`
填充；`longbench_v2` 默认输出仅 10 token。

## 本机适配点

- 模型卡给的是 B200 命令，`fa4`/`trtllm_mha`/`flashinfer`/`cuda-graph-*`/
  `--enable-flashinfer-allreduce-fusion` 均为 CUDA 专用，NPU 上必须去掉。
- 模型卡写的 `--mamba-scheduler-strategy` 在 0.5.22 里叫 `--mamba-radix-cache-strategy`。
- **`--mamba-ssm-dtype bfloat16` 必须显式指定。** 模型 config 写的是 float32，但昇腾算子
  `chunk_gated_delta_rule` 的 FP32 `initial_state` 只在 arch35（Ascend950）支持，
  910B3 会在**加载完成后的第一次前向**抛
  `RuntimeError: FP32 initial_state is only supported on Ascend950 (arch35)`。
  sglang 的 NPU 后端不会自动降级。
- TP 只能是 8：BF16 权重约 244GB，TP=4 每卡需 61GB，放不下 64GB 的卡。
- 910B3 不支持原生 FP8，用 BF16 权重。
- 空闲时每卡 HBM 本底约 3.4GB，`IDLE_HBM_MB=8000` 就是为避开它。
- NPU attention 算子每轮 decode 最多 16 token（`num_speculative_tokens + 1 ≤ 16`），
  `block_size=16` 可能越界，campaign 里有降级到 12 的预案。
- NPU 上 greedy 走 argmax 验证；采样路径在 910C 上实测多约 6ms TPOT。压测固定 greedy。
- NPU 后端按显存分档设默认值，910B3（64GB 档）会拿到 `chunked_prefill_size=8192`、
  `page_size=128`、decode `cuda_graph max_bs=256`(tp≥4)、`disable_custom_all_reduce=True`。
  见 `hardware_backend/npu/utils.py:73`。
- `kernel_meta/` 是 CANN 的 TBE 算子 JIT 产物，落在进程 CWD 下，每次启服务都会再生成。
  用 `./main.sh clean kernel` 清理。

## 精度验证

BF16 的 SSM state 是模型作者指定之外的降级（config 要求 float32），需要确认没有精度塌陷：

```bash
python -m sglang.test.few_shot_gsm8k --num-questions 200 --parallel 16 --port 30000
```

已测 baseline：**Accuracy 0.970, Invalid 0.000**，无塌陷。

边界：GSM8K 输入输出都短，而 BF16 的风险是 recurrent state 误差随序列长度累积，
所以这只覆盖短上下文。长文本场景需另测。另外无法做 FP32 对照（算子层面不支持），
因此这是「绝对值合理性检查」，不是「降级影响量化」。
