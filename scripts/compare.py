#!/usr/bin/env python3
"""聚合多轮压测结果，按并发点对比各配置。

用法:
    ./compare.py 'results/baseline-*.jsonl' 'results/dflash-b8-*.jsonl' ...
    ./compare.py results/baseline-round1-*.jsonl results/dflash-b8-round*.jsonl

每个参数是一个配置（可以是 glob，也可以是已展开的多个文件 —— 同一 tag 的文件
会自动归并）。同一 (配置, 并发) 的多轮数据取 median，并给出 min-max 范围。
第一个配置作为 baseline，其余与它对比。
"""
import glob
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# 结果文件名形如 <tag>-round<N>-<MMDD-HHMMSS>.jsonl
NAME_RE = re.compile(r"^(?P<tag>.+?)-round(?P<round>\d+)-\d{4}-\d{6}\.jsonl$")


def tag_of(path):
    m = NAME_RE.match(Path(path).name)
    if m:
        return m.group("tag")
    # 兼容旧命名 <tag>-<MMDD-HHMMSS>.jsonl
    return re.sub(r"-\d{4}-\d{6}\.jsonl$", "", Path(path).name)


def load_group(patterns):
    """patterns: 一个或多个 glob/路径。返回 (tag, {并发: [每轮指标dict]})。"""
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)) or ([p] if Path(p).exists() else []))
    if not files:
        sys.exit(f"没有匹配到文件: {patterns}")

    tags = {tag_of(f) for f in files}
    if len(tags) > 1:
        sys.exit(f"一个配置里混了多个 tag {sorted(tags)}，请分开传: {files}")

    by_conc = defaultdict(list)
    for f in files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                by_conc[r.get("max_concurrency")].append(r)
    return tags.pop(), dict(by_conc)


def stat(rows, key):
    """返回 (median, min, max, 轮次数)。"""
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return None, None, None, 0
    return statistics.median(vals), min(vals), max(vals), len(vals)


def fmt_range(med, lo, hi, n, width=22):
    if med is None:
        return f"{'-':>{width}}"
    s = f"{med:.2f}" if n == 1 else f"{med:.2f} [{lo:.2f}-{hi:.2f}]"
    return f"{s:>{width}}"


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    groups = [load_group([a]) for a in sys.argv[1:]]
    base_tag, base = groups[0]

    # 轮次数不一致要明说，否则 median 的可信度不同却看不出来
    print("配置与轮次:")
    for tag, data in groups:
        counts = {c: len(v) for c, v in sorted(data.items())}
        uniq = set(counts.values())
        note = f"每并发 {uniq.pop()} 轮" if len(uniq) == 1 else f"轮次不一致 {counts}"
        print(f"  {tag:<20} {note}")
    print()

    for key, label, higher_better in [
        ("median_tpot_ms", "TPOT p50 (ms)", False),
        ("output_throughput", "输出吞吐 (tok/s)", True),
        ("p99_ttft_ms", "TTFT p99 (ms)", False),
    ]:
        print(f"=== {label} ===")
        head = f"{'conc':>6}" + "".join(f"{t:>22}" for t, _ in groups)
        head += "".join(f"{'vs base':>12}" for _ in groups[1:])
        print(head)

        for c in sorted(k for k in base if k is not None):
            line = f"{c:>6}"
            meds = []
            for _, data in groups:
                rows = data.get(c, [])
                med, lo, hi, n = stat(rows, key)
                meds.append(med)
                line += fmt_range(med, lo, hi, n)
            for med in meds[1:]:
                if med and meds[0]:
                    ratio = (med / meds[0]) if higher_better else (meds[0] / med)
                    line += f"{ratio:>11.2f}x"
                else:
                    line += f"{'-':>12}"
            print(line)
        print()

    print("vs 列 > 1.00x 表示该配置优于首个配置（已按指标方向归一）。")
    print("方括号内为多轮的 min-max 范围；范围跨越 1.00x 对应的区间说明差异可能是噪声。")


if __name__ == "__main__":
    main()
