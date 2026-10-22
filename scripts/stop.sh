#!/usr/bin/env bash
# 停掉当前服务并等显存释放干净，再跑下一个配置。
# 不等干净就启下一个，会因为显存不足失败，或者让对照组的 KV cache 预算不一致。

source "$(dirname "$0")/common.sh"

pid_file="$LOG_DIR/server.pid"

# 无论走哪条路径都清掉状态文件，否则残留会让下一次 bench 拿到过期信息
cleanup_state() { rm -f "$pid_file" "$LOG_DIR/server.logpath"; }

if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    pid="$(cat "$pid_file")"
    log "停止 PID $pid"
    kill "$pid" 2>/dev/null || true
else
    # 不用 pkill -f sglang.launch_server：它会匹配到任何命令行含该字符串的进程，
    # 包括监控命令自身，有误杀风险。这里只列出候选，由人工确认。
    log "没有记录在案的进程。候选 sglang 进程如下，请人工确认后处理："
    ps -eo pid,etime,args | awk '/sglang\.launch_server/ && !/awk/ {print "  " $0}' || true
    cleanup_state
    exit 0
fi

log "等待显存释放 ..."
for i in $(seq 1 60); do
    sleep 2
    max_used="$(npu-smi info | grep -oE '[0-9]+ / 65536' | awk '{if ($1>m) m=$1} END {print m+0}')"
    if (( max_used <= IDLE_HBM_MB )); then
        log "显存已释放（峰值 ${max_used}MB）"
        cleanup_state
        exit 0
    fi
done

log "等待 120s 后仍有显存占用（${max_used}MB）。检查残留进程："
log "  ps -ef | grep sglang"
cleanup_state
exit 1
