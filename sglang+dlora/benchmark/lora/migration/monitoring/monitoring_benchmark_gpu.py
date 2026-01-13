import json
import pandas as pd
import matplotlib.pyplot as plt

def load_df(path):
    with open(path) as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)

def get_gpu_ids(df):
    return sorted({int(k.split('_')[1]) for row in df['gpus'] for k in row.keys()})

def get_benchmark_diff(df, gpu_ids):
    bench_idx = df[df['phase'] == 'benchmarking'].index
    diffs = {gpu_id: [] for gpu_id in gpu_ids}
    times = []
    for idx in bench_idx:
        prev_df = df.iloc[:idx]
        prev_idle = prev_df.loc[prev_df['phase'] == 'idle']
        if prev_idle.empty:
            continue
        prev_idle_row = prev_idle.iloc[-1]
        bench_row = df.loc[idx]
        times.append(df.loc[idx, 'timestamp'])
        for gpu_id in gpu_ids:
            gpu_key = f'gpu_{gpu_id}'
            bench_mb = bench_row['gpus'][gpu_key]['memory_used_mb']
            idle_mb = prev_idle_row['gpus'][gpu_key]['memory_used_mb']
            diffs[gpu_id].append(bench_mb - idle_mb)
    return times, diffs

def plot_compare(sglang_path, dlora_path):
    df_sg = load_df(sglang_path)
    df_dl = load_df(dlora_path)
    gpu_ids = sorted(set(get_gpu_ids(df_sg)) | set(get_gpu_ids(df_dl)))
    fig, axes = plt.subplots(2, len(gpu_ids), figsize=(6*len(gpu_ids), 10), sharey='row')
    # sglang
    times_sg, diffs_sg = get_benchmark_diff(df_sg, gpu_ids)
    for i, gpu_id in enumerate(gpu_ids):
        ax = axes[0, i] if len(gpu_ids) > 1 else axes[0]
        ax.plot(range(len(diffs_sg[gpu_id])), diffs_sg[gpu_id])  # 去掉 marker
        ax.set_title(f'sglang GPU {gpu_id}')
        ax.set_ylabel('Δ Mem (MB)')
        ax.set_xlabel('Benchmark Step')
    # dlora
    times_dl, diffs_dl = get_benchmark_diff(df_dl, gpu_ids)
    for i, gpu_id in enumerate(gpu_ids):
        ax = axes[1, i] if len(gpu_ids) > 1 else axes[1]
        ax.plot(range(len(diffs_dl[gpu_id])), diffs_dl[gpu_id], color='orange')  # 去掉 marker
        ax.set_title(f'dlora GPU {gpu_id}')
        ax.set_ylabel('Δ Mem (MB)')
        ax.set_xlabel('Benchmark Step')
    plt.tight_layout()
    plt.savefig('compare_benchmark_gpu_mem_mb.png')
    print('对比图已保存为 compare_benchmark_gpu_mem_mb.png')
    
    fig2, axes2 = plt.subplots(1, len(gpu_ids), figsize=(6*len(gpu_ids), 5), sharey=True)
    if len(gpu_ids) == 1:
        axes2 = [axes2]
    for i, gpu_id in enumerate(gpu_ids):
        ax = axes2[i]
        ax.plot(range(len(diffs_sg[gpu_id])), diffs_sg[gpu_id], label='sglang')
        ax.plot(range(len(diffs_dl[gpu_id])), diffs_dl[gpu_id], label='dlora', color='orange')
        ax.set_title(f'GPU {gpu_id}')
        ax.set_ylabel('Δ Mem (MB)')
        ax.set_xlabel('Benchmark Step')
        ax.legend()
    plt.tight_layout()
    plt.savefig('compare_benchmark_gpu_mem_mb_pergpu.png')
    print('每GPU对比图已保存为 compare_benchmark_gpu_mem_mb_pergpu.png')

if __name__ == '__main__':
    plot_compare(
        "/workspace/sglang/benchmark/lora/migration/monitoring/monitoring_sglang_20260113_011002.jsonl",
        "/workspace/sglang/benchmark/lora/migration/monitoring/monitoring_dlora_20260113_013236.jsonl"
    )