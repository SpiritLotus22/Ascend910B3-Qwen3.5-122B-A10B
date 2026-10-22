#!/usr/bin/env bash
# Qwen3.5-122B-A10B ± DFlash 对比测试：统一入口。
# 实际逻辑在 scripts/ 下，这里只做分发。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$ROOT/scripts"

usage() {
    cat <<'EOF'
用法: ./main.sh <命令> [参数]

  serve baseline            启动不带 DFlash 的服务
  serve dflash [8|16]       启动带 DFlash 的服务（默认 block=8）
  bench [tag]               压测当前服务（tag 缺省时从服务 API 推导）
  stop                      停服务并等显存释放
  status                    打印当前服务的关键参数
  campaign                  端到端跑完整个对比实验（脱机运行）
  compare <g1> <g2> [g3]    对比多个配置的结果，每个参数是一个 glob
  clean [logs|kernel|all]   清理（默认 logs）；永不删 results/
  help                      本帮助

环境变量（见 scripts/common.sh）:
  TP_SIZE MEM_FRACTION MAX_RUNNING PORT
  CONCURRENCY IN_LEN OUT_LEN ROUNDS

脱机跑完整实验:
  nohup ./main.sh campaign > logs/campaign.out 2>&1 &
EOF
}

cmd_serve() {
    case "${1:-}" in
        baseline) shift; bash "$SCRIPTS/serve_baseline.sh" "$@" ;;
        dflash)   shift; BLOCK_SIZE="${1:-8}" bash "$SCRIPTS/serve_dflash.sh" ;;
        *) echo "用法: ./main.sh serve baseline|dflash [8|16]" >&2; exit 1 ;;
    esac
}

cmd_status() {
    # shellcheck disable=SC1091
    source "$SCRIPTS/common.sh"
    if ! curl -fsS --max-time 5 "http://$HOST:$PORT/health_generate" >/dev/null 2>&1; then
        echo "没有服务在 $HOST:$PORT 上运行"
        return 1
    fi
    echo "推导 tag: $(server_tag_from_api)"
    curl -fsS --max-time 10 "http://$HOST:$PORT/get_server_info" | python3 -c "
import json, sys
d = json.load(sys.stdin)
keys = ['speculative_algorithm', 'speculative_dflash_block_size', 'tp_size',
        'mem_fraction_static', 'max_running_requests', 'mamba_ssm_dtype',
        'disable_overlap_schedule', 'enable_mixed_chunk', 'chunked_prefill_size']
for k in keys:
    print(f'  {k:32} = {d.get(k, \"<无>\")}')
"
    [[ -f "$LOG_DIR/server.pid" ]] && echo "  pid = $(cat "$LOG_DIR/server.pid")"
    [[ -f "$LOG_DIR/server.logpath" ]] && echo "  log = $(cat "$LOG_DIR/server.logpath")"
    return 0
}

cmd_clean() {
    # shellcheck disable=SC1091
    source "$SCRIPTS/common.sh"
    local what="${1:-logs}"

    # results/ 是数小时 NPU 机时换来的实验数据，任何 clean 子命令都不碰它
    case "$what" in
        logs|kernel|all) ;;
        results) die "clean 不支持删 results/。那是实验数据，要删请手动 rm。" ;;
        *) echo "用法: ./main.sh clean [logs|kernel|all]" >&2; exit 1 ;;
    esac

    if [[ "$what" == "logs" || "$what" == "all" ]]; then
        if curl -fsS --max-time 5 "http://$HOST:$PORT/health_generate" >/dev/null 2>&1; then
            die "有服务正在运行，拒绝清理日志（会删掉正在写入的文件）。先 ./main.sh stop"
        fi
        local n; n=$(find "$LOG_DIR" -maxdepth 1 -type f \( -name '*.log' -o -name '*.out' \) | wc -l)
        find "$LOG_DIR" -maxdepth 1 -type f \( -name '*.log' -o -name '*.out' \) -delete
        echo "已删除 $n 个日志文件"
    fi

    if [[ "$what" == "kernel" || "$what" == "all" ]]; then
        # CANN 的 TBE 算子 JIT 产物，落在进程 CWD 下，每次启服务都会再生成
        if [[ -d "$ROOT/kernel_meta" ]]; then
            rm -rf "$ROOT/kernel_meta"
            echo "已删除 kernel_meta/"
        else
            echo "没有 kernel_meta/"
        fi
    fi

    echo "results/ 未改动（$(ls -1 "$RESULT_DIR"/*.jsonl 2>/dev/null | wc -l) 个结果文件）"
}

case "${1:-help}" in
    serve)    shift; cmd_serve "$@" ;;
    bench)    shift; bash "$SCRIPTS/bench.sh" "$@" ;;
    stop)     bash "$SCRIPTS/stop.sh" ;;
    status)   cmd_status ;;
    campaign) bash "$SCRIPTS/campaign.sh" ;;
    compare)  shift; python3 "$SCRIPTS/compare.py" "$@" ;;
    clean)    shift; cmd_clean "$@" ;;
    help|-h|--help) usage ;;
    *) echo "未知命令: $1" >&2; echo; usage; exit 1 ;;
esac
