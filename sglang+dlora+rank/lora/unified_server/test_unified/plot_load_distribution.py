import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_load_distribution():
    """
    对比 sglang 和 dlora 在 rate=32, 不同 stride 下的实例负载分布。
    """
    base_dir = '/workspace/sglang/benchmark/lora/load_balance'
    sglang_dir = os.path.join(base_dir, 'sglang')
    dlora_dir = os.path.join(base_dir, 'dlora')
    
    # --- 配置 ---
    rate = 32
    strides = sorted([4, 8, 16, 32, 64, 128])
    
    # dlora stride=4 需要减去的初始值
    dlora_base_counts = {
        "instance_0": 10233,
        "instance_1": 9787
    }

    # --- 数据存储 ---
    sglang_data = {}
    dlora_raw_data = {}

    # --- 1. 读取原始数据 ---
    # 读取 sglang 数据
    for stride in strides:
        filepath = os.path.join(sglang_dir, f"sglang_rate{rate}_stride{stride}.jsonl")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                sglang_data[stride] = data.get("instance_load_distribution", {})
        else:
            print(f"警告: sglang 文件未找到 {filepath}")

    # 读取 dlora 数据
    for stride in strides:
        filepath = os.path.join(dlora_dir, f"dlora_rate{rate}_stride{stride}.jsonl")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                dlora_raw_data[stride] = data.get("instance_load_distribution", {})
        else:
            print(f"警告: dlora 文件未找到 {filepath}")

    # --- 2. 处理数据 ---
    # sglang: 直接提取数量
    sglang_counts = {
        "instance_0": [sglang_data.get(s, {}).get("instance_0", {}).get("request_count", 0) for s in strides],
        "instance_1": [sglang_data.get(s, {}).get("instance_1", {}).get("request_count", 0) for s in strides]
    }

    # dlora: 计算差值
    dlora_counts = {"instance_0": [], "instance_1": []}
    prev_counts = dlora_base_counts
    for stride in strides:
        current_raw = dlora_raw_data.get(stride, {})
        
        # 获取当前原始计数值
        count0_raw = current_raw.get("instance_0", {}).get("request_count", 0)
        count1_raw = current_raw.get("instance_1", {}).get("request_count", 0)
        
        # 计算差值
        diff0 = count0_raw - prev_counts["instance_0"]
        diff1 = count1_raw - prev_counts["instance_1"]
        
        dlora_counts["instance_0"].append(diff0)
        dlora_counts["instance_1"].append(diff1)
        
        # 更新 "前一个" 计数值为当前原始值
        prev_counts = {"instance_0": count0_raw, "instance_1": count1_raw}

    # --- 3. 计算百分比 ---
    def calculate_percentages(counts_dict):
        i0 = np.array(counts_dict["instance_0"])
        i1 = np.array(counts_dict["instance_1"])
        total = i0 + i1
        # 防止除以零
        total[total == 0] = 1 
        
        p0 = (i0 / total) * 100
        p1 = (i1 / total) * 100
        return p0, p1

    sglang_p0, sglang_p1 = calculate_percentages(sglang_counts)
    dlora_p0, dlora_p1 = calculate_percentages(dlora_counts)

    # --- 4. 绘图 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    x = np.arange(len(strides))
    width = 0.5

    # 辅助函数：在柱子上添加标签
    def add_labels(ax, x_pos, p0, p1, counts0, counts1):
        for i in x_pos:
            # 标签 for Instance 0 (bottom)
            if counts0[i] > 0:
                ax.text(i, p0[i] / 2, f'{counts0[i]}\n({p0[i]:.1f}%)',
                        ha='center', va='center', color='white', fontsize=9, weight='bold')
            # 标签 for Instance 1 (top)
            if counts1[i] > 0:
                ax.text(i, p0[i] + p1[i] / 2, f'{counts1[i]}\n({p1[i]:.1f}%)',
                        ha='center', va='center', color='white', fontsize=9, weight='bold')

    # 绘制 sglang 图
    ax1.bar(x, sglang_p0, width, label='Instance 0', color='cornflowerblue')
    ax1.bar(x, sglang_p1, width, bottom=sglang_p0, label='Instance 1', color='lightcoral')
    add_labels(ax1, x, sglang_p0, sglang_p1, sglang_counts["instance_0"], sglang_counts["instance_1"])
    ax1.set_title('sglang Load Distribution (rate=32)')
    ax1.set_xlabel('Stride')
    ax1.set_ylabel('Request Distribution (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strides)
    ax1.legend()
    ax1.set_ylim(0, 100)

    # 绘制 dlora 图
    ax2.bar(x, dlora_p0, width, label='Instance 0', color='cornflowerblue')
    ax2.bar(x, dlora_p1, width, bottom=dlora_p0, label='Instance 1', color='lightcoral')
    add_labels(ax2, x, dlora_p0, dlora_p1, dlora_counts["instance_0"], dlora_counts["instance_1"])
    ax2.set_title('dlora Load Distribution (rate=32)')
    ax2.set_xlabel('Stride')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strides)
    ax2.legend()

    fig.suptitle('Comparison of Instance Load Distribution', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存图表
    # output_filename = 'load_distribution_comparison_with_labels.png'
    output_filename = f'/workspace/sglang/benchmark/lora/test-1/load_distribution_comparison_with_labels.png'
    plt.savefig(output_filename)
    print(f"图表已保存为 {output_filename}")
    plt.show()

if __name__ == '__main__':
    # 假设 rate=32 的文件存在于您的目录中
    # 如果文件不存在，脚本会打印警告但仍会尝试绘图
    plot_load_distribution()