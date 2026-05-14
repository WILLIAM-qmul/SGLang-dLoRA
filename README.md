# DynaRank: Dynamic LoRA Serving with Rank-Aware Scheduling on SGLang

DynaRank is a dynamic LoRA serving system built on top of SGLang, featuring **rank-aware scheduling** for efficient multi-LoRA inference under dynamic workloads.

This repository contains:
- **`DynaRank/`**: the final implementation of the system.
- **`versions/`**: historical intermediate versions showing the step-by-step evolution of the project.
- **Static baseline workflow**: the baseline serving setup.
- **Rank-aware workflow**: the final DynaRank system.

---

## Overview

Serving multiple LoRA adapters efficiently is challenging when request patterns are dynamic and adapters have different computational costs.  
DynaRank addresses this with:

- **Dynamic LoRA serving**
- **Rank-aware scheduling**
- **Decode credit scheduling**
- **Performance modeling support**
- **Trace-driven benchmarking**

In this project:

- **Static** = baseline implementation / baseline serving method
- **Rank-aware** = **DynaRank**, the final optimized system

---

## Repository Structure

```text
.
├── DynaRank/                 # Final implementation
│   ├── benchmark_lora/
│   ├── lora/
│   └── python/
└── versions/                 # Historical intermediate development versions
    ├── benchmark_lora/
    ├── lora/
    ├── python/
    ├── python_credit/
    ├── python_credit_init/
    ├── python_migration/
    ├── python_rank/
    └── un&merge/
```

### Directory Description

#### `DynaRank/`
This is the **final version** of the project and represents the completed DynaRank system.

#### `versions/`
This folder contains the **incremental development history** of the project.  
It is useful for:
- understanding how the system was built step by step,
- comparing different implementation stages,
- reproducing ablation or historical experiments.

---

## Environment

This project is designed to run in an environment integrated with **SGLang** and LoRA benchmarking scripts.

Typical working directory used in experiments:

```bash
/workspace/sglang/benchmark/lora/final/csgmv_2/
```

Please make sure:
- Python dependencies are installed,
- SGLang is properly configured,
- datasets and trace files are available at the expected paths.

---

## Running the Static Baseline

The **Static** setting is the baseline system.

### Launch the static server

```bash
cd /workspace/sglang/benchmark/lora/final/csgmv_2/sglang
python launch_sglang_server.py
```

---

## Running DynaRank

The **Rank-aware** setting corresponds to the final **DynaRank** system.

### Launch the DynaRank server

```bash
cd /workspace/sglang/benchmark/lora/final/csgmv_2/dynarank
python launch_rank_server.py --rank-aware-scheduling --enable-decode-credit-schedule
```

---

## Benchmarking

The repository supports trace-driven benchmarking with different workload traces such as:

- `azure_v1`
- `azure_v2`

Below are the representative commands used in experiments.

---

# Static Baseline Experiments

## Static + azure_v1

```bash
cd /workspace/sglang/benchmark/lora/final/csgmv_2/sglang

python lora_bench_static.py \
  --backend sglang \
  --inference-architecture sglang \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 800 \
  --request-rate 8 \
  --num-instances 2 \
  --sglang-port 30001 \
  --unified-server-url http://127.0.0.1:8000 \
  --trace-total-rate 8.0 \
  --use-trace \
  --trace-name azure_v1 \
  --trace-path /workspace/datasets/maf1/
```

Example scaling commands:

```bash
python lora_bench_static.py \
  --backend sglang \
  --inference-architecture sglang \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 1600 \
  --request-rate 16 \
  --num-instances 2 \
  --sglang-port 30001 \
  --unified-server-url http://127.0.0.1:8000 \
  --trace-total-rate 16.0 \
  --use-trace \
  --trace-name azure_v1 \
  --trace-path /workspace/datasets/maf1/

python lora_bench_static.py \
  --backend sglang \
  --inference-architecture sglang \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 2400 \
  --request-rate 24 \
  --num-instances 2 \
  --sglang-port 30001 \
  --unified-server-url http://127.0.0.1:8000 \
  --trace-total-rate 24.0 \
  --use-trace \
  --trace-name azure_v1 \
  --trace-path /workspace/datasets/maf1/

python lora_bench_static.py \
  --backend sglang \
  --inference-architecture sglang \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 3200 \
  --request-rate 32 \
  --num-instances 2 \
  --sglang-port 30001 \
  --unified-server-url http://127.0.0.1:8000 \
  --trace-total-rate 32.0 \
  --use-trace \
  --trace-name azure_v1 \
  --trace-path /workspace/datasets/maf1/

python lora_bench_static.py \
  --backend sglang \
  --inference-architecture sglang \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 4000 \
  --request-rate 40 \
  --num-instances 2 \
  --sglang-port 30001 \
  --unified-server-url http://127.0.0.1:8000 \
  --trace-total-rate 40.0 \
  --use-trace \
  --trace-name azure_v1 \
  --trace-path /workspace/datasets/maf1/

python lora_bench_static.py \
  --backend sglang \
  --inference-architecture sglang \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 4800 \
  --request-rate 48 \
  --num-instances 2 \
  --sglang-port 30001 \
  --unified-server-url http://127.0.0.1:8000 \
  --trace-total-rate 48.0 \
  --use-trace \
  --trace-name azure_v1 \
  --trace-path /workspace/datasets/maf1/
```

---

## Static + azure_v2

```bash
cd /workspace/sglang/benchmark/lora/final/csgmv_2/sglang

python lora_bench_static.py \
  --backend sglang \
  --inference-architecture sglang \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 800 \
  --request-rate 8 \
  --num-instances 2 \
  --sglang-port 30001 \
  --unified-server-url http://127.0.0.1:8000 \
  --trace-total-rate 8.0 \
  --use-trace \
  --trace-name azure_v2
```

Example scaling commands follow the same pattern for request rates:
- 16
- 24
- 32
- 40
- 48

---

## Static + map_stride sensitivity

```bash
cd /workspace/sglang/benchmark/lora/final/csgmv_2/sglang

python lora_bench_static.py \
  --backend sglang \
  --inference-architecture sglang \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 800 \
  --request-rate 8 \
  --num-instances 2 \
  --sglang-port 30001 \
  --unified-server-url http://127.0.0.1:8000 \
  --trace-total-rate 8.0 \
  --use-trace \
  --trace-name azure_v2 \
  --trace-need-sort \
  --trace-map-stride 4
```

You can vary `--trace-map-stride` across:
- `4`
- `8`
- `16`
- `32`
- `64`

---

# DynaRank Experiments

## DynaRank + azure_v1

```bash
cd /workspace/sglang/benchmark/lora/final/csgmv_2/dynarank

python lora_bench_rank.py \
  --backend sglang \
  --inference-architecture dlora \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 800 \
  --request-rate 8 \
  --num-instances 2 \
  --sglang-port 30001 \
  --unified-server-url http://127.0.0.1:8000 \
  --trace-total-rate 8.0 \
  --use-trace \
  --trace-name azure_v1 \
  --trace-path /workspace/datasets/maf1/
```

Example scaling commands:

```bash
python lora_bench_rank.py \
  --backend sglang \
  --inference-architecture dlora \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 1600 \
  --request-rate 16 \
  --num-instances 2 \
  --sglang-port 30001 \
  --unified-server-url http://127.0.0.1:8000 \
  --trace-total-rate 16.0 \
  --use-trace \
  --trace-name azure_v1 \
  --trace-path /workspace/datasets/maf1/

python lora_bench_rank.py \
  --backend sglang \
  --inference-architecture dlora \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 2400 \
  --request-rate 24 \
  --num-instances 2 \
  --sglang-port 30001 \
  --unified-server-url http://127.0.0.1:8000 \
  --trace-total-rate 24.0 \
  --use-trace \
  --trace-name azure_v1 \
  --trace-path /workspace/datasets/maf1/

python lora_bench_rank.py \
  --backend sglang \
  --inference-architecture dlora \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 3200 \
  --request-rate 32 \
  --num-instances 2 \
  --sglang-port 30001 \
  --unified-server-url http://127.0.0.1:8000 \
  --trace-total-rate 32.0 \
  --use-trace \
  --trace-name azure_v1 \
  --trace-path /workspace/datasets/maf1/

python lora_bench_rank.py \
  --backend sglang \
  --inference-architecture dlora \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 4000 \
  --request-rate 40 \
  --num-instances 2 \
  --sglang-port 30001 \
  --unified-server-url http://127.0.0.1:8000 \
  --trace-total-rate 40.0 \
  --use-trace \
  --trace-name azure_v1 \
  --trace-path /workspace/datasets/maf1/

python lora_bench_rank.py \
  --backend sglang \
  --inference-architecture dlora \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 4800 \
  --request-rate 48 \
  --num-instances 2 \
  --sglang-port 30001 \
  --unified-server-url http://127.0.0.1:8000 \
  --trace-total-rate 48.0 \
  --use-trace \
  --trace-name azure_v1 \
  --trace-path /workspace/datasets/maf1/
```

---

## DynaRank + azure_v2

```bash
cd /workspace/sglang/benchmark/lora/final/csgmv_2/dynarank

python lora_bench_rank.py \
  --backend sglang \
  --inference-architecture dlora \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 800 \
  --request-rate 8 \
  --num-instances 2 \
  --sglang-port 30001 \
  --unified-server-url http://127.0.0.1:8000 \
  --trace-total-rate 8.0 \
  --use-trace \
  --trace-name azure_v2
```

Example scaling commands follow the same pattern for request rates:
- 16
- 24
- 32
- 40
- 48

---

## DynaRank + map_stride sensitivity

```bash
cd /workspace/sglang/benchmark/lora/final/csgmv_2/dynarank

python lora_bench_rank.py \
  --backend sglang \
  --inference-architecture dlora \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 800 \
  --request-rate 8 \
  --num-instances 2 \
  --sglang-port 30001 \
  --unified-server-url http://127.0.0.1:8000 \
  --trace-total-rate 8.0 \
  --use-trace \
  --trace-name azure_v2 \
  --trace-need-sort \
  --trace-map-stride 4
```

You can vary `--trace-map-stride` across:
- `4`
- `8`
- `16`
- `32`
- `64`

---

## Performance Modeling

The repository also includes scripts for kernel-level performance modeling.

```bash
cd /workspace/sglang/benchmark/lora/rank/kernel/real

python bench_kernel_perf_llama2_dual.py \
  --backend csgmv \
  --fit \
  --decode-only \
  --save-dec-csv csgmv_dec.csv \
  --save-fit-csv csgmv_fit_results.csv

python bench_kernel_perf_llama2_dual.py \
  --backend triton \
  --fit \
  --decode-only \
  --save-dec-csv triton_dec.csv \
  --save-fit-csv triton_fit_results.csv

python bench_kernel_perf_llama2_dual.py \
  --backend both \
  --fit \
  --prefill-seq-lens 64 128 256 \
  --save-dec-csv dec.csv \
  --save-pre-csv pre.csv \
  --save-fit-csv fit_results.csv
```

---

## Additional map_stride Evaluation with vLLM-style Benchmarking

The following commands were also used for `map_stride` evaluation:

```bash
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --tokenizer /home/lsl/wwg/models/Llama-2-7b-hf \
  --dataset /home/lsl/wwg/datasets/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json \
  --request-rate 4 \
  --num-models 8 \
  --num-prompts 400 \
  --trace_name azure_v2 \
  --trace_path /home/lsl/wwg/datasets/maf2/ \
  --start_time 0.0.0 \
  --end_time 0.6.0 \
  --interval 60 \
  --host $1 \
  --policy $2 \
  --output $output \
  --need_sort \
  --map_stride 4
```

Similarly, `--map_stride` can be set to:
- `8`
- `16`
- `32`
- `64`

---

## Key Idea

The core idea of this repository is:

1. Start from a **static baseline** for LoRA serving.
2. Introduce **rank-aware scheduling** for dynamic requests.
3. Improve scheduling behavior with mechanisms such as:
   - rank-aware request handling,
   - decode credit scheduling,
   - trace-aware evaluation,
   - performance modeling.
4. Arrive at the final system: **DynaRank**.

---

## Notes

- `versions/` is intended for development history and ablation-style understanding.
- `DynaRank/` is the final and recommended implementation.
- `Static` is the baseline.
- `Rank-aware` corresponds to **DynaRank**.

---

## Citation

If you use this repository in your research, please cite the corresponding project/paper if available.

```bibtex
@misc{dynarank,
  title={DynaRank: Dynamic LoRA Serving with Rank-Aware Scheduling on SGLang},
  author={Your Name},
  year={2026}
}
```

---

## Acknowledgement

This project is built on top of the SGLang-based LoRA serving and benchmarking workflow, and extends it with dynamic rank-aware scheduling for efficient multi-LoRA inference.
