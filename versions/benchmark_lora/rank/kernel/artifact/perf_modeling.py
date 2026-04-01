"""
LoRA Kernel Performance Modeling Benchmark（真实权重版）
=========================================================
使用真实磁盘上的 LoRA adapter 权重，在 Llama-2-7B（32层）的真实维度下，
分别对 decode 和 prefill 两种模式进行 kernel 延迟扫描，然后对 4 个候选
自变量做线性回归，比较 R²。

真实维度 (Llama-2-7B, o_proj / down_proj 为例):
  hidden_dim  = 4096
  num_layers  = 32
  每次 forward = 对 32 层分别调用一次 (A sgemm + B sgemm)，模拟真实推理路径

候选自变量:
  X1 = batch_size * max_rank   (TOPPINGS BGMV / Triton 模型)
  X2 = sum_of_ranks            (TOPPINGS MBGMV / csgmv 模型)
  X3 = batch_size              (对照)
  X4 = max_rank                (对照)

同构: 批内所有请求用相同 rank 的 adapter
异构: 批内请求按 rank=8/16/32/64 混合分配

用法:
  python perf_model_benchmark.py
  python perf_model_benchmark.py --backends triton --device cuda:1
"""

import argparse
import csv
import os
import sys
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

# ── 在任何 sglang 导入之前先注入全局 ServerArgs stub ─────────────────────────
import dataclasses

@dataclasses.dataclass
class _MinimalServerArgs:
    """只提供 loader.py 所需的字段，不触发 ServerArgs.__post_init__ 的 GPU 检测。"""
    weight_loader_disable_mmap: bool = False

import sglang.srt.server_args as _sargs_mod
_sargs_mod._global_server_args = _MinimalServerArgs()   # type: ignore[assignment]

# ── 现在可以安全导入 sglang LoRA 模块 ────────────────────────────────────────
from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend

# safetensors 直接读取
try:
    from safetensors.torch import load_file as safetensors_load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

# ── 真实 adapter 路径 ─────────────────────────────────────────────────────────
DEFAULT_BASE_MODEL = "/workspace/models/Llama-2-7b-hf"

REAL_ADAPTERS: List[Tuple[str, int]] = [
    # rank=8
    ("/workspace/models/adapters/8/Llama2-7B-LoRA-Adapter",  8),
    ("/workspace/models/adapters/8/llama2-7b-lora-genwiki",  8),
    ("/workspace/models/adapters/8/llama2-7b-lora-rebel",    8),
    ("/workspace/models/adapters/8/MUFFIN-Llama2-lora-7B",   8),
    # rank=16
    ("/workspace/models/adapters/16/llama-2-7b-hf-lora-alpaca-json", 16),
    ("/workspace/models/adapters/16/llama-2-7b-LORA-data-analyst",   16),
    ("/workspace/models/adapters/16/llama-2-7b-lora-v1",             16),
    ("/workspace/models/adapters/16/llama2-7B-init-dolly-lora",      16),
    # rank=32
    ("/workspace/models/adapters/32/Llama-2-7b-hf-lora-r32-wnli-loraxs", 32),
    ("/workspace/models/adapters/32/llama-2-7b-sft-lora",                 32),
    ("/workspace/models/adapters/32/llama2-7b-recipe-lora",               32),
    ("/workspace/models/adapters/32/ola_llama2_7B_lora1",                 32),
    # rank=64
    ("/workspace/models/adapters/64/azma-llama2-7b-hf-lora-adapter", 64),
    ("/workspace/models/adapters/64/llama-2-7b-chat-lora-adaptor",   64),
    ("/workspace/models/adapters/64/llama2-7b-airos-lora",           64),
    ("/workspace/models/adapters/64/llama2-stable-7b-lora",          64),
]

ADAPTERS_BY_RANK: Dict[int, List[str]] = {}
for _path, _rank in REAL_ADAPTERS:
    ADAPTERS_BY_RANK.setdefault(_rank, []).append(_path)

# ── Llama-2-7B 参数 ───────────────────────────────────────────────────────────
NUM_LAYERS  = 32
HIDDEN_DIM  = 4096
OUTPUT_DIM  = 4096
DTYPE       = torch.float16

DECODE_SEQ_LEN  = 1
PREFILL_SEQ_LEN = 128

WARMUP_ITERS = 20
BENCH_ITERS  = 100

BATCH_SIZES     = [4, 8, 16, 32, 64]
RANK_CANDIDATES = [8, 16, 32, 64]

HETERO_PATTERNS_BS16 = [
    ([8]  * 16,                                         "homo_8"),
    ([16] * 16,                                         "homo_16"),
    ([32] * 16,                                         "homo_32"),
    ([64] * 16,                                         "homo_64"),
    ([8]  * 8  + [64] * 8,                              "mixed_8_64"),
    ([16] * 8  + [64] * 8,                              "mixed_16_64"),
    ([8]  * 4  + [16] * 4 + [32] * 4 + [64] * 4,       "four_rank"),
    ([8]  * 12 + [64] * 4,                              "skewed_8_64"),
    ([32] * 12 + [64] * 4,                              "skewed_32_64"),
]


# ── ForwardBatch 桩 ───────────────────────────────────────────────────────────

class _ForwardMode:
    def __init__(self, is_extend: bool):
        self._extend = is_extend
    def is_extend(self) -> bool:
        return self._extend


class _ForwardBatch:
    def __init__(self, batch_size: int, seq_len: int, is_extend: bool, device: str):
        self.batch_size    = batch_size
        self.forward_mode  = _ForwardMode(is_extend)
        self.extend_num_tokens = batch_size * seq_len
        if is_extend:
            self.extend_seq_lens_cpu = [seq_len] * batch_size
            self.extend_seq_lens = torch.full(
                (batch_size,), seq_len, dtype=torch.int32, device=device
            )
        else:
            self.extend_seq_lens_cpu = None
            self.extend_seq_lens     = None


class _ServerArgsCsgmv:
    def __init__(self, max_lora_chunk_size: int = 128):
        self.max_lora_chunk_size = max_lora_chunk_size


# ── 权重加载：直接读文件，不经过 SGLang loader ────────────────────────────────

def _find_weight_files(adapter_path: str) -> List[str]:
    """返回 adapter 目录下所有权重文件（.safetensors 优先）。"""
    st_files  = sorted(f for f in os.listdir(adapter_path) if f.endswith(".safetensors"))
    bin_files = sorted(f for f in os.listdir(adapter_path) if f.endswith(".bin"))
    chosen = st_files if st_files else bin_files
    return [os.path.join(adapter_path, f) for f in chosen if
            os.path.isfile(os.path.join(adapter_path, f))]


def _load_single_file(fpath: str) -> Dict[str, torch.Tensor]:
    """加载单个权重文件，返回 {name: tensor}（均在 CPU 上）。"""
    if fpath.endswith(".safetensors"):
        if not HAS_SAFETENSORS:
            raise ImportError("pip install safetensors")
        return safetensors_load_file(fpath, device="cpu")

    # .bin 文件：先用 weights_only=True，失败则回退 weights_only=False
    try:
        data = torch.load(fpath, map_location="cpu", weights_only=True)
    except Exception:
        data = torch.load(fpath, map_location="cpu", weights_only=False)

    # 兼容含 state_dict 包装的 checkpoint
    if isinstance(data, dict):
        if "state_dict" in data:
            data = data["state_dict"]
        # 过滤掉非 Tensor 项（例如 TrainingArguments 对象）
        data = {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}
    return data


def _load_raw_weights(adapter_path: str) -> Dict[str, torch.Tensor]:
    """合并 adapter 目录下所有权重文件。"""
    files = _find_weight_files(adapter_path)
    if not files:
        raise FileNotFoundError(f"No weight files (.safetensors / .bin) in {adapter_path}")
    merged: Dict[str, torch.Tensor] = {}
    for fpath in files:
        merged.update(_load_single_file(fpath))
    return merged


def _parse_layer_id(name: str) -> Optional[int]:
    m = re.search(r"\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def load_adapter_layer_weights(
    adapter_path: str,
    rank: int,
    device: str,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    加载 adapter 每层的 lora_A / lora_B，规范化为:
      A: (rank, HIDDEN_DIM)
      B: (OUTPUT_DIM, rank)
    缺失的层用零张量填充。
    """
    raw = _load_raw_weights(adapter_path)

    # 按 layer_id 分组
    layer_A: Dict[int, Dict[str, torch.Tensor]] = {}
    layer_B: Dict[int, Dict[str, torch.Tensor]] = {}

    for name, tensor in raw.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        lid = _parse_layer_id(name)
        if lid is None:
            continue
        if "lora_A" in name:
            mod_m = re.search(r"(\w+)\.lora_A", name)
            mod   = mod_m.group(1) if mod_m else "unknown"
            layer_A.setdefault(lid, {})[mod] = tensor.cpu()
        elif "lora_B" in name:
            mod_m = re.search(r"(\w+)\.lora_B", name)
            mod   = mod_m.group(1) if mod_m else "unknown"
            layer_B.setdefault(lid, {})[mod] = tensor.cpu()

    # 目标模块优先级
    PREFERRED = ["o_proj", "down_proj", "q_proj", "v_proj",
                 "k_proj", "gate_proj", "up_proj"]

    result: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    for lid in range(NUM_LAYERS):
        a_mods = layer_A.get(lid, {})
        b_mods = layer_B.get(lid, {})

        chosen = None
        for pref in PREFERRED:
            if pref in a_mods and pref in b_mods:
                chosen = pref
                break
        if chosen is None:
            common = set(a_mods.keys()) & set(b_mods.keys())
            chosen = next(iter(common)) if common else None

        A_raw = a_mods.get(chosen) if chosen else None
        B_raw = b_mods.get(chosen) if chosen else None

        # 规范化 A → (rank, HIDDEN_DIM)
        if A_raw is not None:
            if A_raw.dim() > 2:
                A_raw = A_raw.reshape(A_raw.shape[0], -1)
            r_a   = min(A_raw.shape[0], rank)
            in_d  = min(A_raw.shape[1], HIDDEN_DIM)
            A_out = torch.zeros(rank, HIDDEN_DIM, dtype=DTYPE, device=device)
            A_out[:r_a, :in_d] = A_raw[:r_a, :in_d].to(dtype=DTYPE)
        else:
            A_out = torch.zeros(rank, HIDDEN_DIM, dtype=DTYPE, device=device)

        # 规范化 B → (OUTPUT_DIM, rank)
        if B_raw is not None:
            if B_raw.dim() > 2:
                B_raw = B_raw.reshape(B_raw.shape[0], -1)
            out_d = min(B_raw.shape[0], OUTPUT_DIM)
            r_b   = min(B_raw.shape[1], rank)
            B_out = torch.zeros(OUTPUT_DIM, rank, dtype=DTYPE, device=device)
            B_out[:out_d, :r_b] = B_raw[:out_d, :r_b].to(dtype=DTYPE)
        else:
            B_out = torch.zeros(OUTPUT_DIM, rank, dtype=DTYPE, device=device)

        result[lid] = (A_out.contiguous(), B_out.contiguous())

    return result


# ── 堆叠成 batch buffer ───────────────────────────────────────────────────────

def build_batched_buffers(
    per_adapter_lw: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    max_rank: int,
    device: str,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    返回 {layer_id: (A_buf, B_buf)}
      A_buf: (num_adapters, max_rank, HIDDEN_DIM)
      B_buf: (num_adapters, OUTPUT_DIM, max_rank)
    """
    num_adapters = len(per_adapter_lw)
    result: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for lid in range(NUM_LAYERS):
        A_buf = torch.zeros(num_adapters, max_rank, HIDDEN_DIM, dtype=DTYPE, device=device)
        B_buf = torch.zeros(num_adapters, OUTPUT_DIM, max_rank,  dtype=DTYPE, device=device)
        for i, lw in enumerate(per_adapter_lw):
            A, B = lw[lid]
            r = A.shape[0]
            A_buf[i, :r, :]  = A
            B_buf[i, :, :r]  = B
        result[lid] = (A_buf.contiguous(), B_buf.contiguous())
    return result


# ── Backend 工厂 & prepare ────────────────────────────────────────────────────

def make_backend(backend_name: str, num_loras: int, device: str):
    d = torch.device(device)
    if backend_name == "triton":
        return TritonLoRABackend(max_loras_per_batch=num_loras, device=d)
    else:
        return ChunkedSgmvLoRABackend(
            max_loras_per_batch=num_loras,
            device=d,
            server_args=_ServerArgsCsgmv(max_lora_chunk_size=128),
        )


def prepare_backend(
    backend,
    batch_size: int,
    seq_len: int,
    is_extend: bool,
    weight_indices: List[int],
    lora_ranks: List[int],
    device: str,
):
    fb = _ForwardBatch(batch_size, seq_len, is_extend, device)
    backend.prepare_lora_batch(
        forward_batch=fb,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=[1.0] * len(lora_ranks),
        batch_info=None,
    )


# ── 完整 32 层前向 ────────────────────────────────────────────────────────────

def run_32layer_forward(
    backend,
    x: torch.Tensor,
    layer_buffers: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> torch.Tensor:
    out_offset = torch.tensor([0, OUTPUT_DIM], dtype=torch.int32, device=device)
    h = x
    for lid in range(NUM_LAYERS):
        A, B = layer_buffers[lid]
        la = backend.run_lora_a_sgemm(h, A)
        h  = backend.run_lora_b_sgemm(x=la, weights=B,
                                       output_offset=out_offset, base_output=None)
    return h


# ── CUDA Event 计时 ───────────────────────────────────────────────────────────

def measure(
    backend,
    x: torch.Tensor,
    layer_buffers: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    device: str,
    warmup: int = WARMUP_ITERS,
    bench: int  = BENCH_ITERS,
) -> Tuple[float, float]:
    for _ in range(warmup):
        run_32layer_forward(backend, x, layer_buffers, device)
    torch.cuda.synchronize()

    t_start = [torch.cuda.Event(enable_timing=True) for _ in range(bench)]
    t_end   = [torch.cuda.Event(enable_timing=True) for _ in range(bench)]
    for i in range(bench):
        t_start[i].record()
        run_32layer_forward(backend, x, layer_buffers, device)
        t_end[i].record()
    torch.cuda.synchronize()

    lats = [s.elapsed_time(e) for s, e in zip(t_start, t_end)]
    return float(np.mean(lats)), float(np.std(lats))


# ── 线性回归 ──────────────────────────────────────────────────────────────────

def fit(X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    Xm = np.stack([X, np.ones_like(X)], axis=1)
    (alpha, beta), *_ = np.linalg.lstsq(Xm, y, rcond=None)
    y_hat  = alpha * X + beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(alpha), float(beta), float(r2)


# ── Sweep 1: 同构 ─────────────────────────────────────────────────────────────

def sweep_homogeneous(
    backend_name: str,
    mode: str,
    device: str,
    warmup: int,
    bench: int,
) -> List[dict]:
    is_extend = (mode == "prefill")
    seq_len   = PREFILL_SEQ_LEN if is_extend else DECODE_SEQ_LEN
    records: List[dict] = []
    total = len(BATCH_SIZES) * len(RANK_CANDIDATES)
    done  = 0

    print(f"\n[{backend_name}][{mode}][homo]  sweeping {total} combos ...")
    print(f"  {'bs':>4}  {'rank':>4}  {'bs×rank':>7}  {'mean_ms':>10}  {'std_ms':>7}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*7}  {'-'*10}  {'-'*7}")

    for bs in BATCH_SIZES:
        for rank in RANK_CANDIDATES:
            avail = ADAPTERS_BY_RANK.get(rank, [])
            if not avail:
                print(f"  [SKIP] no adapters for rank={rank}")
                continue

            selected_paths  = [avail[i % len(avail)] for i in range(bs)]
            unique_paths    = list(dict.fromkeys(selected_paths))
            path_to_idx     = {p: i for i, p in enumerate(unique_paths)}
            weight_indices   = [path_to_idx[p] for p in selected_paths]
            num_unique       = len(unique_paths)
            lora_ranks_list  = [rank] * num_unique

            per_adapter_lw = [
                load_adapter_layer_weights(path, rank, device)
                for path in unique_paths
            ]
            layer_buffers = build_batched_buffers(per_adapter_lw, rank, device)

            total_tokens = bs * seq_len
            x = torch.randn(total_tokens, HIDDEN_DIM, dtype=DTYPE,
                            device=device).contiguous()

            backend = make_backend(backend_name, num_unique, device)
            prepare_backend(backend, bs, seq_len, is_extend,
                            weight_indices, lora_ranks_list, device)

            mean_ms, std_ms = measure(backend, x, layer_buffers,
                                      device, warmup, bench)

            rec = dict(
                backend       = backend_name,
                mode          = mode,
                dist          = "homo",
                batch_size    = bs,
                rank          = rank,
                max_rank      = rank,
                sum_rank      = bs * rank,
                bs_x_max_rank = bs * rank,
                total_tokens  = total_tokens,
                num_unique    = num_unique,
                mean_ms       = mean_ms,
                std_ms        = std_ms,
            )
            records.append(rec)
            done += 1
            print(f"  {bs:>4}  {rank:>4}  {bs*rank:>7}  "
                  f"{mean_ms:>10.4f}  {std_ms:>7.4f}  [{done}/{total}]")

            del backend, layer_buffers, per_adapter_lw, x
            torch.cuda.empty_cache()

    return records


# ── Sweep 2: 异构，固定 bs=16 ────────────────────────────────────────────────

def sweep_heterogeneous(
    backend_name: str,
    mode: str,
    device: str,
    warmup: int,
    bench: int,
) -> List[dict]:
    is_extend = (mode == "prefill")
    seq_len   = PREFILL_SEQ_LEN if is_extend else DECODE_SEQ_LEN
    bs        = 16
    records: List[dict] = []

    path_to_rank = {p: r for p, r in REAL_ADAPTERS}

    print(f"\n[{backend_name}][{mode}][hetero bs=16]  "
          f"{len(HETERO_PATTERNS_BS16)} patterns ...")
    print(f"  {'pattern':<20}  {'sum_rank':>8}  {'max_rank':>8}  "
          f"{'bs×max':>7}  {'mean_ms':>10}  {'std_ms':>7}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*7}")

    for ranks_per_req, label in HETERO_PATTERNS_BS16:
        assert len(ranks_per_req) == bs

        rank_counters  = {r: 0 for r in set(ranks_per_req)}
        selected_paths = []
        for r in ranks_per_req:
            avail = ADAPTERS_BY_RANK.get(r, [])
            if avail:
                selected_paths.append(avail[rank_counters[r] % len(avail)])
                rank_counters[r] += 1
            else:
                fallback = next(iter(ADAPTERS_BY_RANK.values()))[0]
                selected_paths.append(fallback)

        unique_paths  = list(dict.fromkeys(selected_paths))
        path_to_idx   = {p: i for i, p in enumerate(unique_paths)}
        weight_indices = [path_to_idx[p] for p in selected_paths]
        num_unique     = len(unique_paths)
        unique_ranks   = [path_to_rank.get(p, ranks_per_req[0]) for p in unique_paths]
        max_rank_val   = max(ranks_per_req)
        sum_rank_val   = sum(ranks_per_req)

        per_adapter_lw = [
            load_adapter_layer_weights(
                path, path_to_rank.get(path, max_rank_val), device
            )
            for path in unique_paths
        ]
        layer_buffers = build_batched_buffers(per_adapter_lw, max_rank_val, device)

        total_tokens = bs * seq_len
        x = torch.randn(total_tokens, HIDDEN_DIM, dtype=DTYPE,
                        device=device).contiguous()

        backend = make_backend(backend_name, num_unique, device)
        prepare_backend(backend, bs, seq_len, is_extend,
                        weight_indices, unique_ranks, device)

        mean_ms, std_ms = measure(backend, x, layer_buffers, device, warmup, bench)

        rec = dict(
            backend       = backend_name,
            mode          = mode,
            dist          = "hetero",
            batch_size    = bs,
            rank_pattern  = label,
            max_rank      = max_rank_val,
            sum_rank      = sum_rank_val,
            bs_x_max_rank = bs * max_rank_val,
            total_tokens  = total_tokens,
            num_unique    = num_unique,
            mean_ms       = mean_ms,
            std_ms        = std_ms,
        )
        records.append(rec)
        print(f"  {label:<20}  {sum_rank_val:>8}  {max_rank_val:>8}  "
              f"{bs*max_rank_val:>7}  {mean_ms:>10.4f}  {std_ms:>7.4f}")

        del backend, layer_buffers, per_adapter_lw, x
        torch.cuda.empty_cache()

    return records


# ── 拟合 & 打印 ───────────────────────────────────────────────────────────────

def analyze(records: List[dict]):
    combos = sorted({(r["backend"], r["mode"], r["dist"]) for r in records})

    print("\n" + "=" * 76)
    print("Performance Model Fitting   latency(32 layers) = α·X + β")
    print("=" * 76)

    for backend_name, mode, dist in combos:
        sub = [r for r in records
               if r["backend"] == backend_name
               and r["mode"]    == mode
               and r["dist"]    == dist]
        if len(sub) < 3:
            continue

        y            = np.array([r["mean_ms"]       for r in sub])
        bs_x_maxrank = np.array([r["bs_x_max_rank"] for r in sub])
        sum_ranks    = np.array([r["sum_rank"]       for r in sub])
        batch_sizes  = np.array([r["batch_size"]     for r in sub])
        max_ranks    = np.array([r["max_rank"]       for r in sub])

        print(f"\n[backend={backend_name}]  [mode={mode}]  [dist={dist}]  (n={len(sub)})")
        print(f"  {'X':<32}  {'α':>12}  {'β':>10}  {'R²':>7}")
        print(f"  {'-'*32}  {'-'*12}  {'-'*10}  {'-'*7}")

        for lbl, X in [
            ("bs × max_rank  [BGMV/Triton]",  bs_x_maxrank),
            ("sum_of_ranks   [MBGMV/csgmv]",  sum_ranks),
            ("batch_size     [���照]",           batch_sizes),
            ("max_rank       [对照]",            max_ranks),
        ]:
            alpha, beta, r2 = fit(X, y)
            star = "  ★" if r2 >= 0.90 else ""
            print(f"  {lbl:<32}  {alpha:>12.6f}  {beta:>10.4f}  {r2:>7.4f}{star}")


# ── CSV ───────────────────────────────────────────────────────────────────────

def save_csv(records: List[dict], path: str):
    if not records:
        return
    all_keys = list(dict.fromkeys(k for r in records for k in r))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in all_keys})
    print(f"\nResults saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends",     nargs="+", default=["triton", "csgmv"],
                        choices=["triton", "csgmv"])
    parser.add_argument("--modes",        nargs="+", default=["decode", "prefill"],
                        choices=["decode", "prefill"])
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--warmup-iters", type=int, default=WARMUP_ITERS)
    parser.add_argument("--bench-iters",  type=int, default=BENCH_ITERS)
    parser.add_argument("--output-csv",   default="./perf_model_results/results.csv")
    parser.add_argument("--skip-hetero",  action="store_true",
                        help="只跑同构 sweep，跳过异构 sweep")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    print("LoRA Kernel Performance Modeling  (真实 adapter 权重 + 32层)")
    print(f"  Base model : {DEFAULT_BASE_MODEL}")
    print(f"  num_layers : {NUM_LAYERS}   hidden_dim : {HIDDEN_DIM}")
    print(f"  dtype      : {DTYPE}   device : {args.device}")
    print(f"  warmup     : {args.warmup_iters}   bench : {args.bench_iters}")
    print(f"  batch_sizes: {BATCH_SIZES}")
    print(f"  ranks      : {RANK_CANDIDATES}")

    all_records: List[dict] = []
    for backend_name in args.backends:
        for mode in args.modes:
            all_records.extend(
                sweep_homogeneous(backend_name, mode, args.device,
                                  args.warmup_iters, args.bench_iters)
            )
            if not args.skip_hetero:
                all_records.extend(
                    sweep_heterogeneous(backend_name, mode, args.device,
                                        args.warmup_iters, args.bench_iters)
                )

    analyze(all_records)
    save_csv(all_records, args.output_csv)


if __name__ == "__main__":
    main()