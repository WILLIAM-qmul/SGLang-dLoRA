import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_latency_comparison():
    """
    读取 sglang 和 dlora 的基准测试结果，
    并绘制不同 rate 下 mean_e2e_latency_ms 的对比柱状图。
    """
    base_dir = '/workspace/sglang/benchmark/lora/load_balance'
    sglang_dir = os.path.join(base_dir, 'sglang')
    dlora_dir = os.path.join(base_dir, 'dlora')
    
    rates = [4, 8, 16, 32, 64]
    strides = sorted([4, 8, 16, 32, 64, 128])

    for rate in rates:
        architectures = {
            "sglang": {"dir": sglang_dir, "data": {}},
            "dlora": {"dir": dlora_dir, "data": {}}
        }

        # 提取数据
        for arch, info in architectures.items():
            for stride in strides:
                # 文件名格式如: sglang_rate64_stride128.jsonl
                filename = f"{arch}_rate{rate}_stride{stride}.jsonl"
                filepath = os.path.join(info["dir"], filename)
                
                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            latency = data.get("mean_e2e_latency_ms")
                            if latency is not None:
                                info["data"][stride] = latency
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"无法读取或解析文件 {filepath}: {e}")
                else:
                    print(f"警告: 文件未找到 {filepath}")

        sglang_latencies = [architectures["sglang"]["data"].get(s, 0) for s in strides]
        dlora_latencies = [architectures["dlora"]["data"].get(s, 0) for s in strides]

        # 绘图
        x = np.arange(len(strides))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(12, 7))
        rects1 = ax.bar(x - width/2, sglang_latencies, width, label='sglang', color='skyblue')
        rects2 = ax.bar(x + width/2, dlora_latencies, width, label='dlora', color='sandybrown')

        ax.set_ylabel('Mean End-to-End Latency (ms)')
        ax.set_xlabel('Stride')
        ax.set_title(f'Comparison of Mean E2E Latency (rate={rate})')
        ax.set_xticks(x)
        ax.set_xticklabels(strides)
        ax.legend()

        ax.bar_label(rects1, padding=3, fmt='%.0f')
        ax.bar_label(rects2, padding=3, fmt='%.0f')

        fig.tight_layout()
        
        # 保存图表到文件
        output_filename = f'/workspace/sglang/benchmark/lora/test-1/latency_comparison_rate{rate}.png'
        plt.savefig(output_filename)
        print(f"图表已保存为 {output_filename}")
        
        plt.show()

        plt.close(fig)  # 关闭当前图，防止内存泄漏

if __name__ == '__main__':
    plot_latency_comparison()