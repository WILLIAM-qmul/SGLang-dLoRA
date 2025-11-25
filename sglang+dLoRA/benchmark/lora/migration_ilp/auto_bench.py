import os
import subprocess

BENCH_SCRIPT = "/workspace/sglang/benchmark/lora/lora_bench_unified.py"
RESULT_ROOT = "/workspace/sglang/benchmark/lora/migration_ilp"
SG_LANG_DIR = os.path.join(RESULT_ROOT, "sglang")
DLORA_DIR = os.path.join(RESULT_ROOT, "dlora")

# 可根据需要调整
# REQUEST_RATES = [2, 4, 8, 16, 32, 64]
REQUEST_RATES = [10]
TRACE_MAP_STRIDES = [100]
ARCHS = ["sglang", "dlora"]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_bench(arch, req_rate, trace_map_stride, out_dir):
    ensure_dir(out_dir)
    output_file = os.path.join(out_dir, f"{arch}_rate{req_rate}_stride{trace_map_stride}.jsonl")
    cmd = [
        "python3", BENCH_SCRIPT,
        "--num-prompts", "1000",
        "--trace-total-rate", str(req_rate),
        "--inference-architecture", arch,
        "--output-file", output_file,
        "--use-trace",
        "--trace-total-rate", str(req_rate),
        "--trace-map-stride", str(trace_map_stride)
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    for arch, out_dir in zip(ARCHS, [SG_LANG_DIR, DLORA_DIR]):
        for req_rate in REQUEST_RATES:
            for trace_map_stride in TRACE_MAP_STRIDES:
                run_bench(arch, req_rate, trace_map_stride, out_dir)

if __name__ == "__main__":
    main()