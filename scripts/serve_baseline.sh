#!/usr/bin/env bash
# 对照组：不带 DFlash 的 Qwen3.5-122B-A10B。
#
# 用法:
#   ./serve_baseline.sh              # 标准对照组
#   NO_OVERLAP=1 ./serve_baseline.sh # 可选：关掉 overlap，单独量化 overlap 本身的收益
#
# 注意：sglang 0.5.22 在 NPU 上**不会**因为开启投机解码而关闭 overlap scheduler
# （只有 device=cpu / mps 才强制关，见 speculative_hook.py:65-73、platform_hook.py:48）。
# 所以 baseline 和 DFlash 两组的 overlap 状态本来就一致，常规对比直接用默认即可，
# NO_OVERLAP 只在你想单独测 overlap 贡献时才需要。

source "$(dirname "$0")/common.sh"

EXTRA_ARGS=()
TAG="baseline"

if [[ "${NO_OVERLAP:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--disable-overlap-schedule)
    TAG="baseline-nooverlap"
fi

activate_env
start_server "$TAG" "${EXTRA_ARGS[@]}"
