#!/usr/bin/env bash
# 实验组：Qwen3.5-122B-A10B + z-lab DFlash draft model。
#
# 用法:
#   ./serve_dflash.sh                 # block_size=8，高并发推荐值
#   BLOCK_SIZE=16 ./serve_dflash.sh   # block_size=16，低并发接受长度更高
#
# 与 serve_baseline.sh 的唯一差别就是下面这几个 --speculative-* 参数，
# 其余参数全部来自 common.sh，以保证控制变量。

source "$(dirname "$0")/common.sh"

BLOCK_SIZE="${BLOCK_SIZE:-8}"

[[ -f "$DRAFT_PATH/config.json" ]] || die "draft 模型不存在: $DRAFT_PATH"

# DFlash 的 draft 要从 target 固定层取 hidden states，必须配原版 target。
# 这里顺带确认 draft 的 target_layer_ids 和 target 的层数对得上。
activate_env
python - "$DRAFT_PATH" "$MODEL_PATH" <<'PY'
import json, sys
draft = json.load(open(f"{sys.argv[1]}/config.json"))
target = json.load(open(f"{sys.argv[2]}/config.json"))
need = draft["num_target_layers"]
have = target.get("num_hidden_layers") or target["text_config"]["num_hidden_layers"]
assert need == have, f"draft 期望 target 有 {need} 层，实际 {have} 层，模型不匹配"
print(f"  draft/target 层数匹配: {have}；训练时 block_size = {draft['dflash_config']['block_size']}")
PY

# SGLang 的 DFlash 路径依赖这个 overlap plan stream 开关
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

start_server "dflash-b${BLOCK_SIZE}" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "$DRAFT_PATH" \
    --speculative-dflash-block-size "$BLOCK_SIZE"
