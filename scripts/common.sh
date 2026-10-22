#!/usr/bin/env bash
# 共享配置与工具函数。由 serve_baseline.sh / serve_dflash.sh / bench.sh 通过 source 引入。
# 不要直接执行本文件。

set -euo pipefail

# ---------- 路径与环境 ----------
CONDA_ENV="${CONDA_ENV:-sglang_npu}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/data/models}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/Qwen3.5-122B-A10B}"
DRAFT_PATH="${DRAFT_PATH:-$MODEL_ROOT/Qwen3.5-122B-A10B-DFlash}"

# 本文件在 scripts/ 下，logs/ 和 results/ 在项目根（scripts/ 的上一层）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
RESULT_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

# ---------- 服务参数（对照组之间必须保持一致，只有 speculative 相关参数可以不同）----------
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"
MEM_FRACTION="${MEM_FRACTION:-0.8}"
MAX_RUNNING="${MAX_RUNNING:-32}"

# 模型 config 里写的是 float32，但昇腾算子 chunk_gated_delta_rule 的 FP32 initial_state
# 只在 arch35（Ascend950）上支持，910B3 会直接抛
#   RuntimeError: FP32 initial_state is only supported on Ascend950 (arch35)
# 这里必须显式覆盖成 bfloat16。sglang 的 NPU 后端目前不会自动降级。
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"

# ---------- NPU 检查阈值 ----------
EXPECTED_NPUS="${EXPECTED_NPUS:-8}"
# 空闲时每卡 CANN runtime 本底占用约 3.4GB，高于此值说明有残留进程占着卡
IDLE_HBM_MB="${IDLE_HBM_MB:-8000}"

# 模型加载 235GB 权重较慢，给足超时
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }

check_paths() {
    [[ -f "$MODEL_PATH/config.json" ]] || die "target 模型不存在: $MODEL_PATH"
    log "target = $MODEL_PATH"
}

# 机器上装了多套 conda（/home/zkw/data/miniconda3 和 /opt/miniconda3），
# 只有前者有 sglang_npu。直接用 PATH 里的 conda 会撞上后者，报 EnvironmentNameNotFound。
# 所以这里按环境名去各个 conda 根目录里找真实 prefix，再用全路径激活。
activate_env() {
    local root prefix=""

    if [[ -n "${ENV_PREFIX:-}" ]]; then
        prefix="$ENV_PREFIX"                      # 允许外部直接指定
    else
        local conda_exe="${CONDA_EXE:-}"
        for root in "$HOME/data/miniconda3" "$HOME/miniconda3" "$HOME/anaconda3" \
                    "${conda_exe%/bin/conda}" /opt/miniconda3; do
            if [[ -n "$root" && -x "$root/envs/$CONDA_ENV/bin/python" ]]; then
                prefix="$root/envs/$CONDA_ENV"
                break
            fi
        done
    fi

    [[ -n "$prefix" ]] || die "找不到 conda 环境 '$CONDA_ENV'。用 conda info --envs 确认，或显式指定 ENV_PREFIX=/path/to/env"

    # conda 的 hook 在非交互 shell 里不会自动加载
    root="${prefix%/envs/*}"
    # shellcheck disable=SC1091
    source "$root/etc/profile.d/conda.sh" || die "无法加载 $root/etc/profile.d/conda.sh"
    conda activate "$prefix" || die "无法激活: $prefix"

    # CANN 的 set_env.sh 由 ~/.bashrc 加载，交互式 shell 里直接继承。
    # 从 cron / 最小环境启动时不会有，这里显式兜一层，否则 import sglang 会失败。
    if [[ -z "${ASCEND_TOOLKIT_HOME:-}" ]]; then
        log "未检测到 CANN 环境，尝试加载 set_env.sh"
        # set_env.sh 里有 LD_LIBRARY_PATH=$LD_LIBRARY_PATH:... 这类写法，
        # 变量未定义时会被 set -u 直接打断，所以临时关掉 -u
        set +u
        # shellcheck disable=SC1091
        source /data/zkw/Ascend/cann/set_env.sh 2>/dev/null || true
        # shellcheck disable=SC1091
        source /data/zkw/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
        set -u
    fi

    log "conda env = $prefix"
    local ver
    ver="$(python -c 'import sglang; print(sglang.__version__)' 2>/dev/null)" \
        || die "无法 import sglang。确认 CANN 环境已加载（source /data/zkw/Ascend/cann/set_env.sh）"
    log "sglang = $ver"
}

# 确认 8 张卡健康、且没有残留进程占显存
check_npu() {
    command -v npu-smi >/dev/null || die "npu-smi 不可用"

    local info health_bad=0 busy=0 n=0
    info="$(npu-smi info)"

    # Health 列：统计非 OK 的芯片
    health_bad="$(awk -F'|' '/910B/ {gsub(/ /,"",$3); if ($3 != "OK") print $3}' <<< "$info" | wc -l)"
    (( health_bad == 0 )) || die "有 $health_bad 张卡 Health 异常，先执行 npu-smi info 排查"

    # HBM 列形如 "3451 / 65536"，逐卡取已用值
    while read -r used; do
        n=$((n + 1))
        if (( used > IDLE_HBM_MB )); then
            log "  NPU $((n - 1)): HBM 已用 ${used}MB，超过空闲阈值 ${IDLE_HBM_MB}MB"
            busy=$((busy + 1))
        fi
    done < <(grep -oE '[0-9]+ / 65536' <<< "$info" | awk '{print $1}')

    (( n == EXPECTED_NPUS )) || die "检测到 $n 张卡，期望 $EXPECTED_NPUS 张"

    if (( busy > 0 )); then
        log "有 $busy 张卡显存未释放，可能有残留服务。确认后可用下面的命令清理："
        log "  pkill -f sglang.launch_server"
        die "NPU 未处于空闲状态，中止启动"
    fi

    log "NPU 检查通过：$n 张 910B3 健康且空闲"
}

# 端口已被占用时直接报错，避免启动两个服务互相干扰
check_port_free() {
    if curl -fsS --max-time 2 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
        die "端口 $PORT 上已有服务在跑。先停掉它，或换一个 PORT。"
    fi
}

# 轮询直到 /health_generate 真正能跑通一次前向，而不只是端口通了
wait_for_server() {
    local log_file="$1" pid="$2" waited=0
    log "等待服务就绪（超时 ${STARTUP_TIMEOUT}s，加载 235GB 权重通常需要数分钟）..."

    while (( waited < STARTUP_TIMEOUT )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            log "---- 日志末尾 ----"; tail -n 40 "$log_file"
            die "服务进程已退出，完整日志见 $log_file"
        fi
        if curl -fsS --max-time 5 "http://$HOST:$PORT/health_generate" >/dev/null 2>&1; then
            log "服务就绪，用时 ${waited}s"
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
        (( waited % 60 == 0 )) && log "  已等待 ${waited}s ..."
    done

    log "---- 日志末尾 ----"; tail -n 40 "$log_file"
    die "启动超时，完整日志见 $log_file"
}

# 冒烟测试：发一个真实请求，确认输出不是空的
smoke_test() {
    log "冒烟测试 ..."
    local resp
    resp="$(curl -fsS --max-time 120 "http://$HOST:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"default","messages":[{"role":"user","content":"1+1=? 只回答数字。"}],
             "max_tokens":32,"temperature":0}')" || die "冒烟测试请求失败"

    python - "$resp" <<'PY'
import json, sys
msg = json.loads(sys.argv[1])["choices"][0]["message"]
text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
assert text.strip(), "模型返回了空内容"
print("  模型输出:", text.strip()[:120].replace("\n", " "))
PY
    log "冒烟测试通过"
}

# $1 = TAG（baseline / dflash-b8 等），其余参数为该配置独有的 server 参数
start_server() {
    local tag="$1"; shift
    local log_file="$LOG_DIR/server-${tag}-$(date +%m%d-%H%M%S).log"

    check_paths
    check_npu
    check_port_free

    log "启动配置: $tag"
    log "日志: $log_file"

    # 所有对照组共用的参数，改动请同时影响两个配置，否则对比不成立
    nohup python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --trust-remote-code \
        --device npu \
        --attention-backend ascend \
        --tp-size "$TP_SIZE" \
        --mem-fraction-static "$MEM_FRACTION" \
        --max-running-requests "$MAX_RUNNING" \
        --mamba-radix-cache-strategy extra_buffer \
        --mamba-ssm-dtype "$MAMBA_SSM_DTYPE" \
        --host 0.0.0.0 --port "$PORT" \
        "$@" > "$log_file" 2>&1 &

    local pid=$!
    log "PID = $pid"

    wait_for_server "$log_file" "$pid"
    smoke_test

    # 状态文件在服务确认就绪之后才写。写早了的话，启动失败会留下残留文件，
    # 后续 bench 会拿着过期的 logpath 去捞 accept length，而且不报错。
    echo "$pid"      > "$LOG_DIR/server.pid"
    echo "$log_file" > "$LOG_DIR/server.logpath"

    echo
    log "===== $tag 已就绪，监听 $HOST:$PORT ====="
    log "压测:   ./main.sh bench"
    log "停止:   ./main.sh stop"
    log "看日志: tail -f $log_file"
}

# 从活着的服务推导 tag，而不是读之前写下的文件 —— 标签不可能贴错。
# 输出形如 baseline / dflash-b8 / dflash-b16
server_tag_from_api() {
    curl -fsS --max-time 10 "http://$HOST:$PORT/get_server_info" 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
algo = (d.get('speculative_algorithm') or '').upper()
if algo == 'DFLASH':
    print(f\"dflash-b{d.get('speculative_dflash_block_size')}\")
elif algo:
    print(algo.lower())
else:
    print('baseline' if not d.get('disable_overlap_schedule') else 'baseline-nooverlap')
" 2>/dev/null
}

# 压测前校验共享参数与本脚本的预期一致。两组配置若在这些参数上不同，
# 对比就不成立，而这种差异不会报错，只会静默污染结论。
assert_shared_args() {
    local info
    info="$(curl -fsS --max-time 10 "http://$HOST:$PORT/get_server_info" 2>/dev/null)" \
        || die "无法读取 /get_server_info"

    python3 - "$info" "$TP_SIZE" "$MEM_FRACTION" "$MAX_RUNNING" "$MAMBA_SSM_DTYPE" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
want = {
    "tp_size":              int(sys.argv[2]),
    "mem_fraction_static":  float(sys.argv[3]),
    "max_running_requests": int(sys.argv[4]),
    "mamba_ssm_dtype":      sys.argv[5],
}
bad = []
for k, v in want.items():
    got = d.get(k)
    if got != v:
        bad.append(f"  {k}: 服务={got!r} 期望={v!r}")
if bad:
    sys.exit("共享参数与预期不一致，对比不成立：\n" + "\n".join(bad))
print("  共享参数校验通过: " + ", ".join(f"{k}={v}" for k, v in want.items()))
PY
}
