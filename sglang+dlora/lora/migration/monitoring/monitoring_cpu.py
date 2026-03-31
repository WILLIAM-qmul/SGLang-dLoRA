# File: sglang+dlora/benchmark/lora/migration/monitoring/monitoring_cpu. py
"""
Compare SGLang and dLoRA CPU memory utilization - Phase-Aligned Comparison
阶段对齐对比：每个阶段按开始时间间隔对齐，以最大时间段填充背景，独立线条
支持按时间顺序区分同名阶段（如多个idle阶段）
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

def load_monitoring_data(file_path):
    """Load monitoring data"""
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return pd.DataFrame()
    
    data = []
    with open(file_path) as f:
        for line in f:  
            try:
                data.append(json.loads(line.strip()))
            except json. JSONDecodeError:
                continue
    
    if not data:   
        print(f"Warning: File is empty or invalid {file_path}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Parse timestamp
    df['datetime'] = df['timestamp'].apply(lambda x: datetime.fromisoformat(x. replace('Z', '+00:00')))
    
    # Compute relative time (seconds from first data point)
    start_time = df['datetime'].min()
    df['relative_time'] = (df['datetime'] - start_time).dt.total_seconds()
    
    return df

def extract_sequential_phase_data(df, system_name):
    """Extract phase data with sequential numbering for same phase names"""
    phase_data = {}
    
    if df.empty:
        return phase_data
    
    if 'phase' in df.columns:
        # System with phase information (dLoRA)
        df_sorted = df.sort_values('relative_time').reset_index(drop=True)
        
        # Track phase transitions to identify sequential phases
        phase_counter = {}
        current_phase = None
        phase_segments = []
        segment_start_idx = 0
        
        for idx, row in df_sorted.iterrows():
            phase = row['phase']
            if pd.isna(phase):
                continue
                
            # Detect phase change
            if phase != current_phase:
                # Save previous phase segment if exists
                if current_phase is not None:
                    segment_data = df_sorted.iloc[segment_start_idx:idx]
                    if not segment_data.empty:
                        phase_segments.append({
                            'phase': current_phase,
                            'sequence': phase_counter.get(current_phase, 0),
                            'data': segment_data
                        })
                
                # Start new phase segment
                current_phase = phase
                phase_counter[phase] = phase_counter.get(phase, 0) + 1
                segment_start_idx = idx
        
        # Don't forget the last segment
        if current_phase is not None:
            segment_data = df_sorted.iloc[segment_start_idx:]
            if not segment_data. empty:
                phase_segments. append({
                    'phase':  current_phase,
                    'sequence': phase_counter.get(current_phase, 0),
                    'data': segment_data
                })
        
        # Convert segments to phase_data with sequential naming
        for segment in phase_segments:
            phase = segment['phase']
            sequence = segment['sequence']
            data = segment['data']
            
            # Create unique phase name with sequence number
            if sequence > 1:
                phase_key = f"{phase}_{sequence}"
                phase_label = f"{phase. replace('_', ' ').title()} #{sequence}"
            else:
                phase_key = phase
                phase_label = phase.replace('_', ' ').title()
            
            if not data.empty and 'cpu_memory_util' in data.columns:
                phase_data[phase_key] = {
                    'original_phase': phase,
                    'sequence': sequence,
                    'phase_label': phase_label,
                    'start_time': data['relative_time'].iloc[0],
                    'end_time': data['relative_time'].iloc[-1],
                    'duration': data['relative_time'].iloc[-1] - data['relative_time'].iloc[0],
                    'relative_times': data['relative_time']. values,
                    'cpu_values': data['cpu_memory_util'].values,
                    'system':  system_name
                }
    else:
        # System without phase information (SGLang) - treat as continuous overall phase
        if 'cpu_memory_util' in df. columns:
            df_sorted = df.sort_values('relative_time')
            phase_data['overall'] = {
                'original_phase': 'overall',
                'sequence': 1,
                'phase_label': 'Overall',
                'start_time': df_sorted['relative_time'].iloc[0],
                'end_time': df_sorted['relative_time'].iloc[-1],
                'duration':  df_sorted['relative_time']. iloc[-1] - df_sorted['relative_time'].iloc[0],
                'relative_times':  df_sorted['relative_time']. values,
                'cpu_values': df_sorted['cpu_memory_util'].values,
                'system': system_name
            }
    
    return phase_data

def create_aligned_phase_mapping(sglang_phases, dlora_phases):
    """Create phase alignment mapping with maximum duration and aligned start times"""
    
    # Collect all unique phase keys from both systems
    all_phase_keys = set(list(sglang_phases.keys()) + list(dlora_phases.keys()))
    
    # Define colors for different phase types
    phase_type_colors = {
        'warmup': ('yellow', 'Warmup'),
        'launching_instances': ('orange', 'Launching Instances'),
        'idle': ('lightgray', 'Idle'),
        'benchmarking': ('lightgreen', 'Benchmarking'),
        'overall': ('lightblue', 'Overall')
    }
    
    # Sort phase keys by their chronological order (based on start times)
    def get_phase_start_time(phase_key):
        sglang_start = sglang_phases. get(phase_key, {}).get('start_time', float('inf'))
        dlora_start = dlora_phases.get(phase_key, {}).get('start_time', float('inf'))
        return min(sglang_start, dlora_start)
    
    sorted_phase_keys = sorted(all_phase_keys, key=get_phase_start_time)
    
    aligned_phases = {}
    aligned_offset = 0  # Starting position in aligned timeline
    
    print("\n" + "="*70)
    print("Sequential Phase Alignment Analysis")
    print("="*70)
    
    for phase_key in sorted_phase_keys: 
        sglang_phase = sglang_phases.get(phase_key, {})
        dlora_phase = dlora_phases.get(phase_key, {})
        
        # Get original phase name for color selection
        original_phase = (sglang_phase.get('original_phase') or 
                         dlora_phase.get('original_phase', 'unknown'))
        
        # Get phase label
        phase_label = (sglang_phase.get('phase_label') or 
                      dlora_phase.get('phase_label', phase_key. replace('_', ' ').title()))
        
        # Calculate phase start times (time intervals from beginning)
        sglang_start = sglang_phase. get('start_time', 0) if sglang_phase else None
        dlora_start = dlora_phase.get('start_time', 0) if dlora_phase else None
        
        # Calculate maximum duration for this phase
        durations = []
        if sglang_phase:
            durations.append(sglang_phase['duration'])
        if dlora_phase:
            durations.append(dlora_phase['duration'])
        
        max_duration = max(durations) if durations else 0
        # Ensure minimum duration for visibility
        if max_duration < 5:  
            max_duration = 5
            
        # Get color for this phase type
        color, default_label = phase_type_colors.get(original_phase, ('lightblue', 'Unknown'))
        
        aligned_phases[phase_key] = {
            'aligned_start': aligned_offset,
            'aligned_end': aligned_offset + max_duration,
            'max_duration': max_duration,
            'sglang_data': sglang_phase,
            'dlora_data': dlora_phase,
            'color': color,
            'label':  phase_label,
            'original_phase': original_phase
        }
        
        print(f"\n📊 Phase: {phase_label. upper()}")
        if sglang_phase:
            print(f"   SGLang: Start={sglang_start:.1f}s, Duration={sglang_phase['duration']:.1f}s")
        else:
            print(f"   SGLang: No data")
        
        if dlora_phase:  
            print(f"   dLoRA:  Start={dlora_start:.1f}s, Duration={dlora_phase['duration']:.1f}s")
        else:
            print(f"   dLoRA:  No data")
            
        print(f"   → Aligned:  {aligned_offset:.1f}s - {aligned_offset + max_duration:.1f}s (Max Duration: {max_duration:.1f}s)")
        
        aligned_offset += max_duration + 3  # 3s gap between phases
    
    return aligned_phases

def plot_phase_aligned_comparison(sglang_file, dlora_file, output_file="cpu_phase_aligned_comparison. png"):
    """Plot phase-aligned CPU utilization comparison with separate line segments"""
    
    print("Loading monitoring data...")
    sglang_df = load_monitoring_data(sglang_file)
    dlora_df = load_monitoring_data(dlora_file)
    
    if sglang_df.empty and dlora_df.empty:
        print("Error: No valid data to plot")
        return
    
    # Extract sequential phase data
    sglang_phases = extract_sequential_phase_data(sglang_df, "SGLang")
    dlora_phases = extract_sequential_phase_data(dlora_df, "dLoRA")
    
    # Create aligned phases
    aligned_phases = create_aligned_phase_mapping(sglang_phases, dlora_phases)
    
    if not aligned_phases:
        print("Error: No phases to align")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
    
    # ==================== Phase-Aligned Time Series Plot ====================
    
    # Plot phase backgrounds first
    for phase_key, phase_info in aligned_phases.items():
        aligned_start = phase_info['aligned_start']
        aligned_end = phase_info['aligned_end']
        color = phase_info['color']
        label = phase_info['label']
        
        # Add phase background
        ax1.axvspan(aligned_start, aligned_end, alpha=0.2, color=color)
        
        # Add phase label at the top
        mid_time = (aligned_start + aligned_end) / 2
        ax1.annotate(label, xy=(mid_time, 1), xycoords=('data', 'axes fraction'),
                     xytext=(0, -5), textcoords='offset points',
                     ha='center', va='top', fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    # Set up for legend tracking
    sglang_label_added = False
    dlora_label_added = False
    
    # Plot aligned data for each phase
    phase_stats = {'sglang': {}, 'dlora': {}}
    
    for phase_key, phase_info in aligned_phases.items():
        aligned_start = phase_info['aligned_start']
        max_duration = phase_info['max_duration']
        
        # Plot SGLang data for this phase
        sglang_data = phase_info['sglang_data']
        if sglang_data and sglang_data['cpu_values']. size > 0:
            # Normalize original times to start from 0
            original_times = sglang_data['relative_times'] - sglang_data['relative_times'][0]
            cpu_values = sglang_data['cpu_values']
            
            # Create aligned times for this phase
            aligned_times = aligned_start + original_times
            
            # Only plot data within max_duration
            valid_mask = original_times <= max_duration
            if valid_mask.any():
                ax1.plot(aligned_times[valid_mask], cpu_values[valid_mask],
                        color='blue', linewidth=2.5, alpha=0.8,
                        label='SGLang' if not sglang_label_added else "")
                if not sglang_label_added:
                    sglang_label_added = True
            
            # Store phase statistics
            phase_stats['sglang'][phase_key] = {
                'mean': np.mean(cpu_values),
                'min': np. min(cpu_values),
                'max': np. max(cpu_values),
                'std': np.std(cpu_values),
                'count':  len(cpu_values),
                'label': phase_info['label']
            }
        
        # Plot dLoRA data for this phase
        dlora_data = phase_info['dlora_data']
        if dlora_data and dlora_data['cpu_values'].size > 0:
            # Normalize original times to start from 0
            original_times = dlora_data['relative_times'] - dlora_data['relative_times'][0]
            cpu_values = dlora_data['cpu_values']
            
            # Create aligned times for this phase
            aligned_times = aligned_start + original_times
            
            # Only plot data within max_duration
            valid_mask = original_times <= max_duration
            if valid_mask.any():
                ax1.plot(aligned_times[valid_mask], cpu_values[valid_mask],
                        color='red', linewidth=2.5, alpha=0.8,
                        label='dLoRA' if not dlora_label_added else "")
                if not dlora_label_added:
                    dlora_label_added = True
            
            # Store phase statistics
            phase_stats['dlora'][phase_key] = {
                'mean': np.mean(cpu_values),
                'min': np.min(cpu_values),
                'max': np.max(cpu_values),
                'std': np.std(cpu_values),
                'count': len(cpu_values),
                'label': phase_info['label']
            }
    
    # Set up the time series plot
    ax1.set_xlabel('Aligned Time (seconds)', fontsize=12)
    ax1.set_ylabel('CPU Memory Utilization (%)', fontsize=12)
    ax1.set_title('Phase-Aligned CPU Memory Utilization Comparison\n(Sequential phases aligned by chronological order)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Set reasonable Y-axis limits
    all_cpu_values = []
    for system_stats in phase_stats.values():
        for phase_stat in system_stats. values():
            all_cpu_values.extend([phase_stat['min'], phase_stat['max']])
    
    if all_cpu_values:
        min_cpu = min(all_cpu_values)
        max_cpu = max(all_cpu_values)
        margin = max(2, (max_cpu - min_cpu) * 0.1)
        ax1.set_ylim(max(0, min_cpu - margin), min(100, max_cpu + margin))
    
    # ==================== Phase Comparison Bar Chart ====================
    
    phases_with_data = list(aligned_phases.keys())
    phase_labels = [aligned_phases[p]['label'] for p in phases_with_data]
    
    sglang_means = [phase_stats['sglang']. get(p, {'mean': 0})['mean'] for p in phases_with_data]
    dlora_means = [phase_stats['dlora'].get(p, {'mean': 0})['mean'] for p in phases_with_data]
    
    x = np.arange(len(phase_labels))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, sglang_means, width, label='SGLang', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, dlora_means, width, label='dLoRA', color='red', alpha=0.7)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:  
        for bar in bars:  
            height = bar.get_height()
            if height > 0:
                ax2.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel('Phase (Chronological Order)', fontsize=12)
    ax2.set_ylabel('Average CPU Memory Utilization (%)', fontsize=12)
    ax2.set_title('Average CPU Utilization by Sequential Phase', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(phase_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.show()
    
    print(f"\nPhase-aligned comparison plot saved as: {output_file}")
    
    # Print detailed statistics
    print_aligned_phase_statistics(phase_stats, aligned_phases)

def print_aligned_phase_statistics(phase_stats, aligned_phases):
    """Print detailed phase statistics"""
    
    print("\n" + "="*70)
    print("Sequential Phase Statistics")
    print("="*70)
    
    for phase_key in aligned_phases.keys():
        phase_label = aligned_phases[phase_key]['label']
        print(f"\n📊 {phase_label} Phase:")
        
        sglang_stat = phase_stats['sglang']. get(phase_key)
        dlora_stat = phase_stats['dlora'].get(phase_key)
        
        if sglang_stat:  
            print(f"   SGLang: {sglang_stat['mean']:.2f}% ± {sglang_stat['std']:.2f}% "
                  f"(Range: {sglang_stat['min']:.1f}%-{sglang_stat['max']:.1f}%, "
                  f"{sglang_stat['count']} points)")
        else:
            print(f"   SGLang: No data")
            
        if dlora_stat: 
            print(f"   dLoRA:   {dlora_stat['mean']:.2f}% ± {dlora_stat['std']:.2f}% "
                  f"(Range: {dlora_stat['min']:.1f}%-{dlora_stat['max']:.1f}%, "
                  f"{dlora_stat['count']} points)")
        else:
            print(f"   dLoRA:  No data")
            
        # Compare if both have data
        if sglang_stat and dlora_stat:  
            diff = dlora_stat['mean'] - sglang_stat['mean']
            diff_percent = (diff / sglang_stat['mean']) * 100 if sglang_stat['mean'] != 0 else 0
            print(f"   → Difference: {diff:+.2f}% ({diff_percent:+.1f}%)")  # Fixed formatting
            
            if abs(diff) < 1:  
                comparison = "Similar performance"
            elif diff > 0:
                comparison = "dLoRA higher"
            else:
                comparison = "SGLang higher"
            print(f"   → {comparison}")

if __name__ == "__main__":  
    # Monitoring file paths
    monitoring_dir = "/workspace/sglang/benchmark/lora/migration/monitoring"
    sglang_file = os.path.join(monitoring_dir, "monitoring_sglang_20260113_011002.jsonl")
    dlora_file = os.path. join(monitoring_dir, "monitoring_dlora_20260113_013236.jsonl")
    
    # Check file existence
    files_exist = True
    if not os.path.exists(sglang_file):
        print(f"Warning: SGLang monitoring file not found: {sglang_file}")
        files_exist = False
    if not os. path.exists(dlora_file):
        print(f"Warning: dLoRA monitoring file not found: {dlora_file}")
        files_exist = False
    
    if files_exist:  
        # Plot phase-aligned comparison
        plot_phase_aligned_comparison(sglang_file, dlora_file, "cpu_phase_aligned_comparison.png")
    else:
        print("Please check file paths and ensure files exist.")