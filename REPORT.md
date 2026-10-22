# Qwen3.5-122B-A10B + DFlash 投机解码评测报告

**环境**：单机 8×Ascend 910B3（64GB/卡）· sglang 0.5.22 · CANN 9.0.0 · TP=8 · BF16
**测试日期**：2026-10-21
**Target**：`Qwen/Qwen3.5-122B-A10B`　**Draft**：`z-lab/Qwen3.5-122B-A10B-DFlash`（0.8B）

---

## 结论摘要

1. **DFlash 在 910B3 上可用，收益窗口是并发 ≤ 4。** 并发 1 时 TPOT 加速 1.61x、吞吐 1.55x；并发 8 以上基本打平。
2. **`block_size=16` 在 910B3 上会产生错误输出，必须 ≤ 15。** 该配置下服务正常启动、不报错、benchmark 指标全线亮眼（吞吐 4x），但模型实际输出已损坏。
3. **DFlash 的性能方差显著高于 baseline**，单次测量不可信，容量规划不能只看均值。
4. **TTFT 普遍劣化**，对首 token 延迟敏感的场景需谨慎。
5. **中高并发区的结论尚不可靠**：加速比随并发非单调（c=16 触底 1.01x，c=32 又回升至 1.21x），原因未查明；DFlash 首轮压测系统性偏慢 9–33%，原因亦未查明。详见第七节。

---

## 一、关键发现：block_size=16 静默失效

这是本次评测最重要的发现，单独列出。

### 现象

`--speculative-dflash-block-size 16` 时，服务启动正常、无任何报错、健康检查通过、压测顺利跑完 3 轮，指标表现极为亮眼：

| 指标 | baseline | dflash-b16 | 「加速」 |
|---|---|---|---|
| TPOT p50 (c=1) | 14.11 ms | 2.50 ms | 5.62x |
| 输出吞吐 (c=1) | 67.7 tok/s | 306.8 tok/s | 4.52x |
| accept rate | — | 0.95–0.99 | — |

### 实际输出

greedy（temperature=0）下与 baseline 对比：

```
prompt: "List the first 10 prime numbers, separated by commas."
baseline: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29
b16     : 2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!... (持续到 max_tokens)

prompt: "Explain the difference between TCP and UDP in exactly two sentences."
baseline: TCP (Transmission Control Protocol) is a connection-oriented protocol that...
b16     : TCP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!...
```

5 条测试 prompt 全部如此：生成第一个 token 后即退化为 `!` 无限重复。

### 根因

NPU 的 attention 算子每轮 decode 最多处理 16 个 token，即 `num_speculative_tokens + 1 ≤ 16`。`block_size=16` 时实测 `speculative_num_draft_tokens=16`，加验证位为 17，**越界**。

越界后 verify 比对的是未初始化内存，因此「全部接受」——这解释了虚高的 accept rate 0.95–0.99，也解释了 4 倍吞吐：**它不是推理得快，而是在高速输出垃圾**。`!` 通常对应低位 token id，是未初始化读取的典型征兆。

### 教训

这个失败模式的危险之处在于**所有常规信号都是绿的**：不崩溃、不报错、指标漂亮、accept rate 高得诱人。如果只看 benchmark 数字，会直接得出「block=16 性能提升 4 倍」并写进报告。

**性能测试必须配正确性验证**，否则测的可能是一个坏掉的系统跑得有多快。

相关数据已隔离至 `results/INVALID/`，附根因说明，不参与任何结论。

---

## 二、baseline vs dflash-b8

每个配置每个并发点 **3 轮**，下表为 median，方括号内为 min–max。
数据集 `random`，input 1024 / output 512，greedy，`--seed 42`。

### TPOT p50 (ms，越低越好)

| conc | baseline | dflash-b8 | 加速 |
|---|---|---|---|
| 1 | 14.11 [14.00–14.11] | 8.76 [7.79–8.98] | **1.61x** |
| 2 | 14.82 [14.82–14.83] | 10.48 [10.23–11.24] | 1.41x |
| 4 | 16.78 [16.78–16.88] | 13.07 [13.06–14.05] | 1.28x |
| 8 | 20.16 [20.15–20.33] | 19.34 [19.25–21.40] | 1.04x |
| 16 | 25.12 [25.08–25.38] | 26.09 [25.89–29.44] | 0.96x |
| 32 | 34.29 [34.14–35.01] | 30.14 [28.06–30.66] | 1.14x |

### 输出吞吐 (tok/s，越高越好)

| conc | baseline | dflash-b8 | 加速 |
|---|---|---|---|
| 1 | 67.7 [67.6–68.2] | 104.9 [103.1–112.7] | **1.55x** |
| 2 | 123.9 [123.9–124.0] | 175.3 [152.8–175.9] | 1.41x |
| 4 | 219.7 [215.3–220.2] | 277.6 [203.4–280.1] | 1.26x |
| 8 | 366.4 [348.3–368.0] | 374.9 [266.2–399.7] | 1.02x |
| 16 | 577.0 [559.3–578.1] | 581.3 [404.4–600.5] | 1.01x |
| 32 | 798.7 [757.3–799.6] | 970.3 [753.3–1017.5] | 1.21x |

### TTFT p99 (ms，越低越好)

| conc | baseline | dflash-b8 | 比值 |
|---|---|---|---|
| 1 | 381.9 [380.6–385.7] | 428.1 [414.1–3737.1] | 0.89x |
| 2 | 731.5 [721.6–740.9] | 727.0 [711.6–5092.2] | 1.01x |
| 4 | 765.2 [743.5–922.7] | 798.6 [779.4–5467.6] | 0.96x |
| 8 | 958.0 [879.1–4477.5] | 3922.0 [907.7–9553.2] | 0.24x |
| 16 | 1429.1 [1404.4–1846.9] | 2948.5 [1432.6–10154.2] | 0.48x |
| 32 | 16981.1 [16957.0–18336.2] | 5902.6 [4084.9–11557.7] | 2.88x |

### ITL vs TPOT：投机解码的特征

| conc | baseline ITL/TPOT | dflash-b8 ITL/TPOT |
|---|---|---|
| 1 | 14.10 / 14.11 | **4.97 / 8.76** |
| 8 | 20.13 / 20.16 | 8.20 / 19.34 |
| 32 | 33.32 / 34.29 | 12.05 / 30.14 |

baseline 两者几乎相等（每步出 1 token）。DFlash 的 ITL 远低于 TPOT——被接受的 token 几乎零成本吐出，但平摊到每 token 的真实成本（TPOT）包含了 draft 前向与 verify 开销。**用户感知的流式顺滑度由 ITL 决定，实际算力成本由 TPOT 决定**，两者在投机解码下不再等价。

---

## 三、复测推翻了首轮结论

首轮单次测量显示 conc=8/16 吞吐跌至 **0.76x / 0.72x**，据此曾判断「中并发净亏 24–28%，收益窗口仅并发 ≤ 4」。

三轮 median 显示实际是 **1.02x / 1.01x**——**那个「反转」是噪声**。

原因从 min–max 范围可见：

| conc | baseline 吞吐跨度 | dflash-b8 吞吐跨度 |
|---|---|---|
| 8 | 348–368（±3%） | 266–400（**±20%**） |
| 16 | 559–578（±2%） | 404–600（**±19%**） |

首轮恰好取到了 DFlash 的低端值。**投机解码本身的性能方差显著高于 baseline**，这是独立于加速比的一项发现：它意味着 DFlash 的性能对调度时序、batch 组成更敏感，单次 benchmark 不足以定论。

TTFT p99 的跨度更极端（c=1: 414–3737 ms），因此上表中 0.24x / 0.48x 同样不能当作定论。

---

## 四、正确性验证

### 精度（baseline）

```
python -m sglang.test.few_shot_gsm8k --num-questions 200 --parallel 16
→ Accuracy 0.970, Invalid 0.000
```

无精度塌陷。这一步针对的是 `--mamba-ssm-dtype bfloat16` 这一降级（见第五节）。

**边界**：GSM8K 输入输出都短，而 BF16 的风险是 recurrent state 误差随序列长度累积，故仅覆盖短上下文。无法做 FP32 对照（算子层面不支持），因此这是「绝对值合理性检查」，不是「降级影响量化」。

### 输出一致性（dflash-b8）

greedy 下 5 条 prompt 与 baseline 对比：1 条完全一致，4 条有个别词差异，但**输出连贯、语义等价**：

```
baseline: ...guarantees reliable, ordered delivery of data through error checking...
b8      : ...guarantees reliable, ordered delivery of data packets through error checking...

baseline: 如果草稿模型生成的序列被大模型全部接受，... 减少大模型的调用次数。
b8      : 如果草稿模型生成的序列被大模型完全接受，... 减少大模型的计算次数。
```

**但这不能归因给 DFlash**：实测 b8 自身两次相同调用也不一致，说明引擎默认非确定（sglang 有 `--enable-deterministic-inference`，默认 `False`，描述为 "batch invariant ops"，即批式归约顺序随 batch 组成变化）。

与 b16 的区别是决定性的：

| | b16 | b8 |
|---|---|---|
| 输出 | `2!!!!!!...` 无限重复 | 连贯、高质量、语义等价 |
| 判定 | 彻底损坏 | 正常的非确定性分歧 |

---

## 五、910B3 适配要点

| 问题 | 处理 |
|---|---|
| 模型 config 要求 `mamba_ssm_dtype: float32`，但昇腾算子 `chunk_gated_delta_rule` 的 FP32 `initial_state` 仅 arch35(Ascend950) 支持 | 必须显式 `--mamba-ssm-dtype bfloat16`。否则在**加载完成后的首次前向**才抛错（约 9 分钟后），sglang NPU 后端不会自动降级 |
| 模型卡给的是 B200 命令 | `fa4` / `trtllm_mha` / `flashinfer` / `cuda-graph-*` / `--enable-flashinfer-allreduce-fusion` 均为 CUDA 专用，须去掉 |
| 模型卡写 `--mamba-scheduler-strategy` | 0.5.22 中已更名为 `--mamba-radix-cache-strategy` |
| TP 只能为 8 | BF16 权重约 244GB，TP=4 时每卡需 61GB，放不下 64GB 卡 |
| 投机解码会把 `max_running_requests` 默认重置为 48 | 必须显式 `--max-running-requests 32`，否则两组不可比（`speculative_hook.py:1000`） |
| 官方文档称投机解码会关闭 overlap scheduler 和 mixed chunked prefill | **本版本已不成立**。实测 `disable_overlap_schedule=False`、`enable_mixed_chunk=False`（默认即关），且 DFLASH 在 `supports_mixed_chunk()` 白名单内（`spec_info.py:152`）。两组调度条件天然一致，无需额外对照组 |
| `block_size=16` 越界 | 必须 ≤ 15。见第一节 |

---

## 六、结论与建议

### 可直接采用

- **并发 ≤ 4 的场景建议开启 DFlash**（block_size=8），TPOT 收益 1.28–1.61x。
- **交互式/流式场景收益被低估**：ITL 从 14.1ms 降至 5.0ms（c=1），用户感知的顺滑度提升明显大于 TPOT 数字。
- **并发 ≥ 8 时收益基本消失**（1.01–1.04x），但也不亏。是否开启取决于 TTFT 能否接受。

### 必须注意

- **`block_size` 不得超过 15**，否则静默产出错误输出。
- **TTFT 普遍劣化且方差极大**，SLO 含 TTFT 约束时需单独压测验证。
- **DFlash 性能方差高**，容量规划应参考 min 值而非 median。

### 本报告的局限

1. **accept length 偏乐观。** 实测 4.47–4.54，但 `random` 数据集在 prompt 短于目标长度时会**平铺重复** ShareGPT 文本（`datasets/random.py:130-134`）；ShareGPT 典型 prompt 长 100–300 token 而设定为 1024，故输入含大量重复内容，极易被 draft 预测。**真实业务下的收益窗口可能比并发 4 更窄。**
2. **单一 workload 形状。** 仅测 input 1024 / output 512。输出更短时 TTFT 劣化占比更大；代码/数学类内容的 accept length 通常更高（z-lab 数据：HumanEval 6.09 vs MT-Bench 4.54）。
3. **未验证 baseline 的非确定性。** 只确认了 b8 自身非确定，未回头确认 baseline 是否同样如此，故「非确定性是引擎级的」这一判断尚缺直接证据。
4. **长上下文未测。** BF16 SSM state 的误差累积风险在长序列下才会显现。

---

## 七、未解决的问题

本节列出已观察到但**尚未查明原因**的现象。它们不影响「并发 ≤ 4 有收益」这一主结论，但影响对中高并发区的判断。

### 7.1 加速比随并发非单调（主要疑点）

| conc | 1 | 2 | 4 | 8 | 16 | **32** |
|---|---|---|---|---|---|---|
| 吞吐加速 | 1.55x | 1.41x | 1.26x | 1.02x | 1.01x | **1.21x** |
| TPOT 加速 | 1.61x | 1.41x | 1.28x | 1.04x | 0.96x | **1.14x** |

加速比从 c=1 单调下降至 c=16 触底，**在 c=32 又回升**。这不符合「算力饱和后投机解码收益递减」的预期——按该模型，c=32 应该比 c=16 更差。

先澄清一点：首轮报告中提到的「c=32 吞吐打平但 TPOT 反而更好」这一反常组合，在三轮数据下**已经消失**——现在 c=32 两项指标都是 DFlash 占优（1.21x / 1.14x），彼此一致。**遗留的问题是非单调性本身。**

**候选解释（有证据但未证实）**：c=32 时 baseline 出现了更严重的排队。统计服务端日志的 `#queue-req` 分布：

| 配置 | 队列深度分布 |
|---|---|
| baseline | 0(×2863), 1(×116), **7(×40), 15(×20), 23(×20)** |
| dflash-b8 | 0(×2151), 1(×103), 2(×31), 3(×25), 7(×6) |

baseline 的队列最深到 23，DFlash 基本不超过 3。对应地，baseline 在 c=32 的 TTFT p99 高达 17.0s（p50 仅 2.1s），DFlash 为 5.9s。

一个可能的机制是：DFlash 每个 decode step 吐出多个 token，请求更快离开 running 集合、释放 slot，因而排队更浅；baseline 在 prefill 侧堆积，吞吐被排队拖累。**但这只是机制猜想，未做实验验证。**

另一个未排除的因素：`--max-running-requests 32` 恰好等于测试并发 32，调度器正好处在饱和边界，可能存在边界行为。

**要查清需要**：扫描 c=24/48/64 看曲线形状，并把 `max_running_requests` 与测试并发解耦。

### 7.2 DFlash 首轮系统性偏慢，原因不明

按轮次拆开看吞吐（round1 相对后续轮的偏差）：

| conc | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| baseline r1 偏差 | +0.7% | −0.1% | −2.2% | −5.3% | −3.3% | −5.3% |
| **dflash-b8 r1 偏差** | **−8.6%** | **−13.1%** | **−27.4%** | **−33.4%** | **−32.7%** | **−26.0%** |

baseline 首轮只慢 0–5%，而 **DFlash 首轮慢 9–33%，且偏差随并发放大**。这是一个明确的、可复现的系统性差异，但原因未查明。

**这正是首轮报告结论出错的直接原因**：当时 baseline 和 DFlash 都只跑了 round1，而 DFlash 的 round1 被重罚，于是测出 c=16 吞吐 0.72x 的「中并发净亏」。

**候选解释（均未验证）**：
- `--warmup-requests 4` 对 DFlash 不足。DFlash 的 shape 空间更大（draft 前向、不同接受长度下的 verify），4 个预热请求可能覆盖不到。
- CANN 对启动时图捕获未覆盖的 shape 做 JIT 编译（即 `kernel_meta/` 的来源），首轮承担编译开销。
- 某种与 `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` 相关的延迟初始化。

**对当前数据的影响有限**：3 轮取 median 会自动排除单个离群值。实测排除 round1 后加速比变化很小（c=32: 1.21x→1.24x，c=8: 1.02x→1.05x），故主结论不受影响。但这意味着**报告中的 DFlash 数字略偏保守**。

**要查清需要**：把 `--warmup-requests` 加到 32+ 重测首轮，看偏差是否消失。

### 7.3 DFlash 的高方差未能归因

即使排除 round1，DFlash 的轮间波动仍明显大于 baseline（见第三节）。TTFT p99 尤其不稳定（c=1 跨度 414–3737ms）。目前只能观察到现象，无法区分是调度时序、batch 组成敏感性，还是 7.2 的同一成因在后续轮次的残留。

### 7.4 `block_size` 的有效上限未精确定位

已确认 8 可用、16 损坏。中间的 9–15 完全未测。按 `num_speculative_tokens + 1 ≤ 16` 推算上限应为 15，但该推算来自 vLLM-Ascend 文档而非本环境实测，且 sglang 的 `speculative_num_draft_tokens` 与 `block_size` 的映射关系未逐值确认。**不建议在未实测的情况下使用 8 以外的值。**

### 7.5 b16 为何不被任何检查拦截

`block_size=16` 越界后服务照常启动、健康检查通过、冒烟测试也通过（冒烟只检查输出非空，而 `2!!!!...` 非空）。这说明当前脚本的校验强度不足。

已知的改进方向：冒烟测试应检测输出是否退化（如同一字符高比例重复），而不只是判空。尚未实现。

---

## 八、建议的下一步

按优先级（前两项针对第七节的未解决问题）：

1. **加大 `--warmup-requests` 至 32+ 重测**，验证 7.2 的首轮偏慢是否为预热不足。成本最低、收益最直接——若成立，说明现有 DFlash 数字整体偏保守。
2. **扫并发 24/48/64，并把 `max_running_requests` 与测试并发解耦**，查清 7.1 的非单调性。这决定了中高并发区的结论能不能用。
3. 用 `sharegpt` 数据集复测（真实文本、输出长度取自数据集本身，不平铺），校准真实 accept length。
4. 扫输出长度 128 / 512 / 2048，定位 TTFT 劣化被摊薄的临界点。
5. 逐值实测 `block_size` 9–15，定位真实上限（见 7.4）。
6. 长上下文精度验证（needle-in-a-haystack 或长文摘要）。

---

## 附：复现

```bash
cd test_dflash
./main.sh serve dflash 8     # 或 serve baseline
./main.sh status             # 确认实际生效参数
./main.sh bench
./main.sh stop

# 端到端（脱机运行）
nohup ./main.sh campaign > logs/campaign.out 2>&1 &

./main.sh compare 'results/baseline-*' 'results/dflash-b8-*'
```

原始数据：`results/*.jsonl`（有效）、`results/INVALID/`（b16，无效）。
