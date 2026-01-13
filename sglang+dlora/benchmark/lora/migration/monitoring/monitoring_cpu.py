import json
from collections import defaultdict
from datetime import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 两个监控文件
FILES = {
    "sglang": "/workspace/sglang/benchmark/lora/migration/monitoring/monitoring_sglang_20260113_011002.jsonl",
    "dlora": "/workspace/sglang/benchmark/lora/migration/monitoring/monitoring_dlora_20260113_013236.jsonl",
}

def parse_timestamp(ts):
    try:
        return pd.to_datetime(ts)
    except Exception:
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return None

def load_runs_by_phase(path):
    """
    返回：
      - runs_by_phase: dict mapping phase -> list of runs (each run is list of cpu_memory_util in percent)
      - earliest_phase_ts: dict mapping phase -> earliest timestamp (pd.Timestamp) seen in file
    """
    runs_by_phase = defaultdict(list)
    earliest_phase_ts = {}
    with open(path, "r") as f:
        current_phase = None
        current_run = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            phase = obj.get("phase")
            ts = parse_timestamp(obj.get("timestamp")) if obj.get("timestamp") else None

            # 计算 cpu memory util %
            if "cpu_memory_util" in obj:
                # 已经是百分比或小数？ 从数据看是百分比（15.4）或小数（0.16）混用，做容错：
                val = obj["cpu_memory_util"]
                if 0 <= val <= 1:
                    cpu_mem_pct = float(val) * 100.0
                else:
                    cpu_mem_pct = float(val)
            elif "cpu_memory_used_mb" in obj and "cpu_memory_total_mb" in obj:
                used = obj.get("cpu_memory_used_mb", 0.0)
                total = obj.get("cpu_memory_total_mb", 1.0)
                cpu_mem_pct = float(used) / float(total) * 100.0 if total else 0.0
            else:
                # 兜底：尝试使用 cpu_utilization 字段（不是内存，但防止空）
                cpu_mem_pct = float(obj.get("cpu_utilization", math.nan))

            # 记录 earliest timestamp per phase
            if phase:
                if phase not in earliest_phase_ts or (ts is not None and ts < earliest_phase_ts[phase]):
                    earliest_phase_ts[phase] = ts

            # 按连续段切分 runs
            if phase == current_phase:
                current_run.append(cpu_mem_pct)
            else:
                if current_phase is not None and current_run:
                    runs_by_phase[current_phase].append(current_run)
                # start new
                current_phase = phase
                current_run = [cpu_mem_pct] if phase is not None else []

        # 结束时把最后一段加入
        if current_phase is not None and current_run:
            runs_by_phase[current_phase].append(current_run)

    return runs_by_phase, earliest_phase_ts

# 读取两个文件
data_runs = {}
data_earliest = {}
for name, path in FILES.items():
    runs, earliest = load_runs_by_phase(path)
    data_runs[name] = runs
    data_earliest[name] = earliest

# 全量阶段集合，按全局最早出现时间排序（若无时间则放后面）
all_phases = set()
for runs in data_runs.values():
    all_phases.update(runs.keys())

phase_first_ts = {}
for phase in all_phases:
    cand = []
    for name in FILES.keys():
        ts = data_earliest.get(name, {}).get(phase)
        if ts is not None:
            cand.append(ts)
    phase_first_ts[phase] = min(cand) if cand else pd.NaT

# 排序
sorted_phases = sorted(all_phases, key=lambda p: (pd.NaT if phase_first_ts[p] is pd.NaT else phase_first_ts[p]))

# 为每个文件每个阶段挑选最长连续段
selected = {name: {} for name in FILES.keys()}
for name in FILES.keys():
    runs_by_phase = data_runs[name]
    for phase in sorted_phases:
        runs = runs_by_phase.get(phase, [])
        if not runs:
            selected[name][phase] = []
        else:
            # 选择最长 run
            longest = max(runs, key=len)
            selected[name][phase] = longest

# 绘图：每个阶段一个 subplot，按最长区间对齐（用 NaN 填充短序列）
num_phases = len(sorted_phases)
if num_phases == 0:
    print("未发现任何 phase，退出。")
else:
    fig, axes = plt.subplots(num_phases, 1, figsize=(12, 3 * max(1, num_phases)))
    if num_phases == 1:
        axes = [axes]

    colors = {"sglang": "tab:blue", "dlora": "tab:orange"}
    for ax, phase in zip(axes, sorted_phases):
        seqs = {}
        max_len = 0
        for name in FILES.keys():
            seq = selected[name].get(phase, []) or []
            seqs[name] = seq
            if len(seq) > max_len:
                max_len = len(seq)

        x = np.arange(max_len)
        for name, seq in seqs.items():
            if len(seq) == 0:
                continue
            y = np.array(seq, dtype=float)
            if len(y) < max_len:
                y = np.concatenate([y, np.full(max_len - len(y), np.nan)])
            ax.plot(x, y, label=name, color=colors.get(name))
        ax.set_title(f"Phase: {phase}")
        ax.set_ylabel("CPU memory util (%)")
        ax.set_ylim(0, 105)
        ax.grid(True)
        ax.legend()
        # x 轴为区间索引（相对时间，单位为采样步）
        ax.set_xlabel("Relative sample index within phase")

    plt.tight_layout()
    out = "cpu_memory_by_phase_compare.png"
    plt.savefig(out, dpi=150)
    print(f"已保存对比图：{out}")