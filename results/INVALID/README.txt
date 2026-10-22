这些是 dflash-b16 (block_size=16) 的压测结果，已判定为【无效数据】，不得用于任何结论。

原因：greedy (temperature=0) 下输出与 baseline 完全不一致，且生成第一个 token 后即
退化为 "!" 无限重复，模型输出彻底损坏。

  prompt: "List the first 10 prime numbers, separated by commas."
  baseline: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29
  b16     : 2!!!!!!!!!!!!!!!!!!!!!!... (持续到 max_tokens)

根因：NPU attention 算子每轮 decode 最多处理 16 个 token
(num_speculative_tokens + 1 <= 16)。block_size=16 时 speculative_num_draft_tokens=16，
加验证位为 17，越界。服务能正常启动、不报错，但 verify 读到未初始化数据，
因此 accept rate 虚高至 0.95-0.99，"吞吐 4x" 实为高速输出垃圾。

结论：910B3 上 block_size 必须 <= 15。本目录数据仅作为"静默失效"的案例留档。
