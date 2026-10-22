#!/usr/bin/env bash
# 对当前正在运行的服务做并发扫描压测。先启好服务再跑这个。
#
# 用法:
#   ./bench.sh                       # TAG 从 /get_server_info 推导
#   ./bench.sh dflash-b8             # 手动指定 TAG（结果文件名用）
#   ROUND=2 ./bench.sh               # 标记为第 2 轮，写入独立结果文件
#   CONCURRENCY="1 8 32" ./bench.sh  # 自定义并发点
#   IN_LEN=2048 OUT_LEN=512 ./bench.sh
#
# 关于 random 数据集：它采样真实 ShareGPT 文本，**不是**随机 token。
# 且当 prompt 短于目标长度时会平铺重复同一段 token（datasets/random.py:130-134）：
#     ratio = (input_lens[i] + prompt_len - 1) // prompt_len
#     input_ids = (prompt_token_ids * ratio)[: input_lens[i]]
# ShareGPT 典型 prompt 长 100-300 token，而默认 IN_LEN=1024，所以输入含大量
# 重复内容。重复文本极易被 draft 预测，**实测接受长度偏乐观**，报告中须注明。
#
# 真正的对抗下界是 random-ids（伪随机 token ramp）。
# 另：所有数据集默认强制 ignore_eos（serving.py:311 等），每请求精确生成到
# output_len，这对跨配置比较是好事，保持默认。

source "$(dirname "$0")/common.sh"

CONCURRENCY="${CONCURRENCY:-1 2 4 8 16 32}"
IN_LEN="${IN_LEN:-1024}"
OUT_LEN="${OUT_LEN:-512}"
SEED="${SEED:-42}"
ROUND="${ROUND:-1}"

activate_env

curl -fsS --max-time 5 "http://$HOST:$PORT/health_generate" >/dev/null \
    || die "$HOST:$PORT 上没有就绪的服务"

# tag 优先用显式传入，否则从活着的服务推导（不读可能过期的状态文件）
TAG="${1:-}"
if [[ -z "$TAG" ]]; then
    TAG="$(server_tag_from_api)"
    [[ -n "$TAG" ]] || die "无法从 /get_server_info 推导 tag，请显式传入"
fi

# 对比成立的前提：两组只差投机解码参数，其余必须一致
assert_shared_args

STAMP="$(date +%m%d-%H%M%S)"
OUT_JSONL="$RESULT_DIR/${TAG}-round${ROUND}-${STAMP}.jsonl"
SERVER_LOG="$(cat "$LOG_DIR/server.logpath" 2>/dev/null || true)"

log "压测配置: TAG=$TAG  round=$ROUND  input=$IN_LEN  output=$OUT_LEN  concurrency=[$CONCURRENCY]"
log "结果写入: $OUT_JSONL"

# 记下压测前的日志行数，之后只统计这一轮新增的 accept length
log_line_before=1
[[ -n "$SERVER_LOG" && -f "$SERVER_LOG" ]] && log_line_before=$(( $(wc -l < "$SERVER_LOG") + 1 ))

for c in $CONCURRENCY; do
    # 请求数随并发放大，保证每点样本足够，同时低并发不至于跑太久
    n=$(( c * 10 )); (( n < 40 )) && n=40

    log "--- concurrency=$c, num_prompts=$n ---"

    # --flush-cache 清掉 radix cache，避免上一轮的前缀命中污染本轮 TTFT
    python -m sglang.bench_serving \
        --backend sglang \
        --host "$HOST" --port "$PORT" \
        --dataset-name random \
        --random-input-len "$IN_LEN" \
        --random-output-len "$OUT_LEN" \
        --random-range-ratio 1.0 \
        --num-prompts "$n" \
        --max-concurrency "$c" \
        --warmup-requests 4 \
        --flush-cache \
        --seed "$SEED" \
        --output-file "$OUT_JSONL" \
        2>&1 | grep -E 'Successful|Request throughput|Output token throughput|TTFT|TPOT|ITL|E2E|latency' || true
done

# ---------- 从服务端日志里抓接受长度 ----------
if [[ -n "$SERVER_LOG" && -f "$SERVER_LOG" ]]; then
    echo
    log "服务端 accept length（本轮日志采样）:"
    # baseline 不输出该指标，grep 无匹配会返回 1，在 pipefail 下会终止整个脚本。
    # 用 `|| true` 兜住：没有 accept length 是预期情况，不是错误。
    tail -n "+$log_line_before" "$SERVER_LOG" \
        | { grep -oE 'accept[_ ]len(gth)?[:= ]+[0-9.]+' || true; } \
        | awk -F'[:= ]+' '{s+=$NF; n++} END {
            if (n) printf "  平均 %.3f（%d 个采样点）\n", s/n, n;
            else   print "  日志里没有 accept length 字段（baseline 不输出该指标属正常）"
          }'
fi

# ---------- 汇总 ----------
echo
log "汇总 $OUT_JSONL:"
python3 - "$OUT_JSONL" <<'PY'
import json, sys

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
cols = ["max_concurrency", "output_throughput", "median_ttft_ms", "p99_ttft_ms",
        "median_tpot_ms", "p99_tpot_ms", "median_itl_ms", "median_e2e_latency_ms"]
hdr  = ["conc", "out_tok/s", "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99", "itl_p50", "e2e_p50"]

print("  " + "".join(f"{h:>12}" for h in hdr))
for r in sorted(rows, key=lambda x: x.get("max_concurrency") or 0):
    cells = []
    for c in cols:
        v = r.get(c)
        cells.append(f"{v:>12.2f}" if isinstance(v, (int, float)) else f"{str(v):>12}")
    print("  " + "".join(cells))
PY

echo
log "完成: $OUT_JSONL"
