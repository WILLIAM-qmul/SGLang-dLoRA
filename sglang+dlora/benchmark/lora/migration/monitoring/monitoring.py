import json
import pandas as pd
import matplotlib.pyplot as plt

# ----------- 数据读取 -----------
data = []
monitoring_file = "/workspace/sglang/benchmark/lora/migration/monitoring/monitoring_dlora_20260113_013236.jsonl"
with open(monitoring_file) as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# ----------- 阶段统计与展示 -----------
def print_phase_stats(df, gpu_ids):
    for phase in df['phase'].unique():
        phase_data = df[df['phase'] == phase]
        print(f"\n{phase.upper()} Phase:")
        print(f"  Duration: {len(phase_data)} seconds")
        print(f"  Avg CPU: {phase_data['cpu_memory_util'].mean():.2f}%")
        for gpu_id in gpu_ids:
            gpu_key = f"gpu_{gpu_id}"
            gpu_utils = [
                row['gpus'][gpu_key]['memory_util']
                for _, row in phase_data.iterrows()
                if gpu_key in row['gpus']
            ]
            if gpu_utils:
                print(f"  Avg GPU {gpu_id}: {sum(gpu_utils)/len(gpu_utils):.2f}%")

gpu_ids = sorted(
    {int(k.split('_')[1]) for row in df['gpus'] for k in row.keys()}
)
print_phase_stats(df, gpu_ids)

# ----------- 绘图 -----------
fig, axes = plt.subplots(1 + len(gpu_ids), 1, figsize=(12, 4 + 3 * len(gpu_ids)))

# CPU 利用率
axes[0].plot(df.index, df['cpu_memory_util'], label='CPU')
axes[0].set_title('CPU Memory Utilization')
axes[0].set_ylabel('CPU %')
axes[0].set_ylim(0, 105)  # 固定纵坐标范围
axes[0].legend()

# GPU 利用率
for i, gpu_id in enumerate(gpu_ids):
    gpu_key = f"gpu_{gpu_id}"
    gpu_utils = [
        row['gpus'][gpu_key]['memory_util'] if gpu_key in row['gpus'] else 0
        for _, row in df.iterrows()
    ]
    axes[i+1].plot(df.index, gpu_utils, label=f'GPU {gpu_id}')
    axes[i+1].set_title(f'GPU {gpu_id} Memory Utilization')
    axes[i+1].set_ylabel('GPU %')
    axes[i+1].set_ylim(0, 105)  # 固定纵坐标范围
    axes[i+1].legend()

# 阶段标记
phase_colors = {
    'warmup': 'yellow',
    'launching_instances': 'orange',
    'benchmarking': 'green',
    'idle': 'gray'
}
for ax in axes:
    for phase, color in phase_colors.items():
        phase_indices = df[df['phase'] == phase].index
        if len(phase_indices) > 0:
            ax.axvspan(phase_indices[0], phase_indices[-1], alpha=0.2, color=color, label=phase)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

axes[-1].set_xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig('monitoring_analysis.png')
print("监控分析图已保存为 monitoring_analysis.png")