#!/usr/bin/env bash
# 端到端跑完整个对比实验，不需要人工介入。
#
# 设计成一个脚本而不是由对话逐步驱动，是因为全程有 2 次服务重启、9 轮压测，
# 衔接点太多；ssh 一断就会卡在衔接处。用 nohup 或 tmux 起这个脚本即可脱机运行。
#
# 用法:
#   nohup ./main.sh campaign > logs/campaign.out 2>&1 &
#   ROUNDS=1 CONCURRENCY="1" ./main.sh campaign    # 干跑验证编排

source "$(dirname "$0")/common.sh"

ROUNDS="${ROUNDS:-3}"
SCRIPTS="$SCRIPT_DIR"
CAMPAIGN_LOG="$LOG_DIR/campaign-$(date +%m%d-%H%M%S).log"

# 阶段级日志，和各服务自己的日志分开
clog() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$CAMPAIGN_LOG"; }

# 某阶段失败不应该让整个 campaign 中止：后面的配置仍然值得测
FAILED_STAGES=()

# 统计某 tag 已有多少轮结果，用于复用已跑过的数据
existing_rounds() {
    ls "$RESULT_DIR/$1"-round*.jsonl 2>/dev/null | wc -l
}

# $1=tag  $2=目标总轮数
run_rounds() {
    local tag="$1" target="$2"
    local have; have="$(existing_rounds "$tag")"

    if (( have >= target )); then
        clog "  $tag 已有 $have 轮，达到目标 $target，跳过"
        return 0
    fi

    for (( r = have + 1; r <= target; r++ )); do
        clog "  $tag 第 $r/$target 轮压测开始"
        if ROUND="$r" bash "$SCRIPTS/bench.sh" "$tag" >> "$CAMPAIGN_LOG" 2>&1; then
            clog "  $tag 第 $r 轮完成"
        else
            clog "  !! $tag 第 $r 轮失败，详见 $CAMPAIGN_LOG"
            return 1
        fi
    done
}

# $1=阶段名  $2=tag  $3...=启动命令
run_stage() {
    local name="$1" tag="$2"; shift 2
    clog "=============================================="
    clog "阶段: $name  (tag=$tag)"
    clog "=============================================="

    # 已有足够轮次就连服务都不用起
    if (( $(existing_rounds "$tag") >= ROUNDS )); then
        clog "  已有 $ROUNDS 轮结果，整个阶段跳过"
        return 0
    fi

    # 服务没在跑就启动；已在跑且 tag 相符则直接复用（省一次约 6 分钟的加载）
    local current=""
    curl -fsS --max-time 5 "http://$HOST:$PORT/health_generate" >/dev/null 2>&1 \
        && current="$(server_tag_from_api)"

    if [[ "$current" == "$tag" ]]; then
        clog "  复用已在运行的 $tag 服务"
    else
        if [[ -n "$current" ]]; then
            clog "  当前跑的是 $current，先停掉"
            bash "$SCRIPTS/stop.sh" >> "$CAMPAIGN_LOG" 2>&1 || true
        fi
        clog "  启动 $tag ..."
        if ! "$@" >> "$CAMPAIGN_LOG" 2>&1; then
            clog "  !! $tag 启动失败"
            FAILED_STAGES+=("$name(启动失败)")
            return 1
        fi
        clog "  $tag 已就绪"
    fi

    run_rounds "$tag" "$ROUNDS" || FAILED_STAGES+=("$name(压测失败)")
}

main() {
    clog "campaign 开始: ROUNDS=$ROUNDS CONCURRENCY=[${CONCURRENCY:-默认}]"
    clog "日志: $CAMPAIGN_LOG"

    # 阶段 1：dflash-b8。放第一个是为了复用当前可能已在运行的服务
    run_stage "1/3 DFlash block=8" "dflash-b8" \
        env BLOCK_SIZE=8 bash "$SCRIPTS/serve_dflash.sh" || true

    # 阶段 2：dflash-b16。NPU attention 算子每轮 decode 最多 16 token
    # （num_speculative_tokens + 1 <= 16），block=16 可能越界，失败则降级试 12。
    clog "=============================================="
    clog "阶段: 2/3 DFlash block=16"
    clog "=============================================="
    if (( $(existing_rounds "dflash-b16") >= ROUNDS )); then
        clog "  已有 $ROUNDS 轮结果，跳过"
    else
        bash "$SCRIPTS/stop.sh" >> "$CAMPAIGN_LOG" 2>&1 || true
        if BLOCK_SIZE=16 bash "$SCRIPTS/serve_dflash.sh" >> "$CAMPAIGN_LOG" 2>&1; then
            clog "  dflash-b16 已就绪"
            run_rounds "dflash-b16" "$ROUNDS" || FAILED_STAGES+=("2/3 b16(压测失败)")
        else
            clog "  !! block=16 启动失败（可能踩到 NPU 每轮 16 token 上限），降级试 block=12"
            bash "$SCRIPTS/stop.sh" >> "$CAMPAIGN_LOG" 2>&1 || true
            if BLOCK_SIZE=12 bash "$SCRIPTS/serve_dflash.sh" >> "$CAMPAIGN_LOG" 2>&1; then
                clog "  >>> 注意：实际使用 block=12，不是 16 <<<"
                run_rounds "dflash-b12" "$ROUNDS" || FAILED_STAGES+=("2/3 b12(压测失败)")
            else
                clog "  !! block=12 也失败，跳过本阶段"
                FAILED_STAGES+=("2/3 b16/b12(均启动失败)")
            fi
        fi
    fi

    # 阶段 3：baseline
    run_stage "3/3 baseline" "baseline" bash "$SCRIPTS/serve_baseline.sh" || true

    clog "=============================================="
    if (( ${#FAILED_STAGES[@]} == 0 )); then
        clog "campaign 全部完成"
    else
        clog "campaign 结束，但有阶段失败:"
        for s in "${FAILED_STAGES[@]}"; do clog "  - $s"; done
    fi
    clog "结果文件:"
    ls -1 "$RESULT_DIR"/*.jsonl 2>/dev/null | while read -r f; do clog "  $(basename "$f")"; done
    clog "对比: ./main.sh compare 'results/baseline-*' 'results/dflash-b8-*' 'results/dflash-b16-*'"
}

main
