"""
bench_kernel_perf_llama2_dual.py  ── 真实推理路径版（修复版）
=======================================================
完全按照 Llama-2-7B + SGLang LoRA 的实际推理路径模拟 32 层 forward：

  每层 = 4 次 kernel 调用（与 layers.py 对应）:
    1. qkv_proj    → backend.run_qkv_lora
                     A: (nu, 3r, H), B: (nu, Q+2KV, r)
                     output_offset: [0, Q, Q+KV, Q+2KV]  ← 4 元素
    2. o_proj      → backend.run_lora_a_sgemm + run_lora_b_sgemm
                     A: (nu, r, Q), B: (nu, H, r)
    3. gate_up_proj→ backend.run_gate_up_lora
                     A: (nu, 2r, H), B: (nu, 2*FFN, r)
    4. down_proj   → backend.run_lora_a_sgemm + run_lora_b_sgemm
                     A: (nu, r, FFN), B: (nu, H, r)

TOPPINGS 模型拟合（decode / prefill 分别）：
  BGMV:  latency = α · bs·max_rank + β
  MBGMV: latency = α · Σrank_i    + β

用法:
    python bench_kernel_perf_llama2_dual.py --backend both --fit \\
        --save-dec-csv dec.csv --save-fit-csv fit.csv
"""

# ─── 标准库 ───────────────────────────────────────────────────────────────────
import argparse
import csv
import itertools
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

# ─── 第三方库 ─────────────────────────────────────────────────────────────────
import numpy as np
import torch

# ─── SGLang 后端（直接 import，无需 stub） ───────────────────────────────────
from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend
from sglang.srt.lora.utils import LoRABatchInfo


# ═════════════════════════════════════════════════════════════════════════════
# 全局常量 — Llama-2-7B
# ═════════════════════════════════════════════════════════════════════════════

DEVICE = "cuda"
DTYPE  = torch.float16

HIDDEN_DIM   = 4096
NUM_HEADS    = 32
NUM_KV_HEADS = 32
HEAD_DIM     = 128          # HIDDEN_DIM // NUM_HEADS
FFN_DIM      = 11008
NUM_LAYERS   = 32

Q_OUT_DIM    = NUM_HEADS    * HEAD_DIM    # 4096
KV_OUT_DIM   = NUM_KV_HEADS * HEAD_DIM   # 4096
QKV_OUT_DIM  = Q_OUT_DIM + 2 * KV_OUT_DIM  # 12288

# qkv output_offset 必须是 4 元素: [0, Q, Q+KV, Q+2KV]
QKV_OUTPUT_OFFSET = [0, Q_OUT_DIM, Q_OUT_DIM + KV_OUT_DIM, Q_OUT_DIM + 2 * KV_OUT_DIM]
MAX_QKV_OUT_DIM   = max(Q_OUT_DIM, KV_OUT_DIM)   # 4096

# gate_up output_offset: [0, FFN, 2*FFN]
GU_OUTPUT_OFFSET  = [0, FFN_DIM, 2 * FFN_DIM]

# (in_dim, out_dim) for lora A/B shapes
MODULE_DIMS: Dict[str, Tuple[int, int]] = {
    "q_proj":    (HIDDEN_DIM, Q_OUT_DIM),
    "k_proj":    (HIDDEN_DIM, KV_OUT_DIM),
    "v_proj":    (HIDDEN_DIM, KV_OUT_DIM),
    "o_proj":    (Q_OUT_DIM,  HIDDEN_DIM),
    "gate_proj": (HIDDEN_DIM, FFN_DIM),
    "up_proj":   (HIDDEN_DIM, FFN_DIM),
    "down_proj": (FFN_DIM,    HIDDEN_DIM),
}

TARGET_RANKS      = [8, 16, 32, 64]
ADAPTERS_PER_RANK = 4

ADAPTER_REGISTRY: List[Tuple[str, int]] = [
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
    ("/workspace/models/adapters/32/Final_llama2-7B-lora_r_32", 32),
    ("/workspace/models/adapters/32/llama-2-7b-sft-lora",                 32),
    ("/workspace/models/adapters/32/llama2-7b-recipe-lora",               32),
    ("/workspace/models/adapters/32/ola_llama2_7B_lora1",                 32),
    # rank=64
    ("/workspace/models/adapters/64/azma-llama2-7b-hf-lora-adapter", 64),
    ("/workspace/models/adapters/64/llama-2-7b-chat-lora-adaptor",   64),
    ("/workspace/models/adapters/64/llama2-7b-airos-lora",           64),
    ("/workspace/models/adapters/64/llama2-stable-7b-lora",          64),
]

DEFAULT_BASE_MODEL = "/workspace/models/Llama-2-7b-hf"


# ═════════════════════════════════════════════════════════════════════════════
# 后端桩（ForwardBatch / ServerArgs 最小实现）
# ═════════════════════════════════════════════════════════════════════════════

class _CsgmvArgs:
    def __init__(self, max_lora_chunk_size: int = 128):
        self.max_lora_chunk_size = max_lora_chunk_size

class _ForwardMode:
    def __init__(self, is_extend: bool): self._e = is_extend
    def is_extend(self) -> bool: return self._e

class _ForwardBatch:
    def __init__(self, batch_size: int, seq_len: int, is_extend: bool, device: str):
        self.batch_size    = batch_size
        self.forward_mode  = _ForwardMode(is_extend)
        self.extend_num_tokens = batch_size * seq_len
        if is_extend:
            self.extend_seq_lens_cpu = [seq_len] * batch_size
            self.extend_seq_lens = torch.full(
                (batch_size,), seq_len, dtype=torch.int32, device=device)
        else:
            self.extend_seq_lens_cpu = None
            self.extend_seq_lens     = None

def make_backend(name: str, num_loras: int, device: str):
    d = torch.device(device)
    if name == "triton":
        return TritonLoRABackend(max_loras_per_batch=num_loras, device=d)
    else:
        return ChunkedSgmvLoRABackend(
            max_loras_per_batch=num_loras, device=d,
            server_args=_CsgmvArgs(128))

def prepare_backend(backend, bs: int, seq_len: int, is_extend: bool,
                    weight_indices: List[int], lora_ranks: List[int], device: str):
    fb = _ForwardBatch(bs, seq_len, is_extend, device)
    backend.prepare_lora_batch(
        forward_batch=fb, weight_indices=weight_indices,
        lora_ranks=lora_ranks, scalings=[1.0]*len(lora_ranks), batch_info=None)


# ═════════════════════════════════════════════════════════════════════════════
# 基础模型权重加载（GPU）
# ═════════════════════════════════════════════════════════════════════════════

def load_base_model(model_dir: str) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    返回 {layer_idx: {module_name: Tensor(out_dim, in_dim)}} on GPU.
    module_name ∈ {q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj}
    """
    import glob
    from safetensors.torch import load_file

    shards = sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors")))
    if not shards:
        s = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(s): shards = [s]
    if not shards:
        raise FileNotFoundError(f"No safetensors in {model_dir}")

    KEEP = set(MODULE_DIMS.keys())
    raw: Dict[str, torch.Tensor] = {}
    for p in shards:
        print(f"  [BASE] loading {os.path.basename(p)} ...", end=" ", flush=True)
        d = load_file(p, device="cuda")
        n = 0
        for k, t in d.items():
            if "model.layers." not in k: continue
            if not any(k.endswith(f".{m}.weight") for m in KEEP): continue
            raw[k] = t.to(torch.float16); n += 1
        print(f"{n} tensors")

    result: Dict[int, Dict[str, torch.Tensor]] = {i: {} for i in range(NUM_LAYERS)}
    for k, t in raw.items():
        m = re.search(r"model\.layers\.(\d+)\.", k)
        if not m: continue
        li = int(m.group(1))
        if li >= NUM_LAYERS: continue
        for mod in KEEP:
            if k.endswith(f".{mod}.weight"):
                result[li][mod] = t; break

    total = NUM_LAYERS * len(MODULE_DIMS)
    loaded = sum(1 for li in range(NUM_LAYERS) for mod in MODULE_DIMS if result[li].get(mod) is not None)
    print(f"  [BASE] loaded {loaded}/{total} tensors")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Adapter 权重加载（CPU）
# ═════════════════════════════════════════════════════════════════════════════

def _load_adapter_raw(path: str) -> Dict[str, torch.Tensor]:
    st = os.path.join(path, "adapter_model.safetensors")
    bn = os.path.join(path, "adapter_model.bin")
    if os.path.exists(st):
        from safetensors.torch import load_file
        return load_file(st, device="cpu")
    if os.path.exists(bn):
        try:
            data = torch.load(bn, map_location="cpu", weights_only=True)
        except Exception:
            data = torch.load(bn, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            if "state_dict" in data: data = data["state_dict"]
            data = {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}
        return data
    raise FileNotFoundError(f"No weights in {path}")

def _read_adapter_config(path: str) -> dict:
    f = os.path.join(path, "adapter_config.json")
    return json.load(open(f)) if os.path.exists(f) else {}


def parse_adapter(
    raw: Dict[str, torch.Tensor],
    rank: int,
) -> Tuple[Dict[str, List[Optional[torch.Tensor]]], Dict[str, List[Optional[torch.Tensor]]], List[str]]:
    """
    解析 PEFT state_dict → (lora_A_store, lora_B_store, covered_modules)
      lora_A_store[module][layer_idx] = Tensor(rank, in_dim) on CPU or None
      lora_B_store[module][layer_idx] = Tensor(out_dim, rank) on CPU or None
    """
    lora_A: Dict[str, List[Optional[torch.Tensor]]] = {m: [None]*NUM_LAYERS for m in MODULE_DIMS}
    lora_B: Dict[str, List[Optional[torch.Tensor]]] = {m: [None]*NUM_LAYERS for m in MODULE_DIMS}
    covered: set = set()

    for key, tensor in raw.items():
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2: continue
        if   ".lora_A.weight" in key: ab = "A"
        elif ".lora_B.weight" in key: ab = "B"
        else: continue

        m_li = re.search(r"layers\.(\d+)\.", key)
        if not m_li: continue
        li = int(m_li.group(1))
        if li >= NUM_LAYERS: continue

        mod = None
        for candidate in MODULE_DIMS:
            if f".{candidate}.lora_{ab}" in key:
                mod = candidate; break
        if mod is None: continue
        covered.add(mod)

        in_dim, out_dim = MODULE_DIMS[mod]
        t = tensor.to(torch.float16)

        if ab == "A":
            # expected: (rank, in_dim)
            r_src = t.shape[0]; c_src = t.shape[1]
            out_t = torch.zeros(rank, in_dim, dtype=torch.float16)
            out_t[:min(r_src, rank), :min(c_src, in_dim)] = t[:min(r_src, rank), :min(c_src, in_dim)]
            lora_A[mod][li] = out_t.contiguous()
        else:
            # expected: (out_dim, rank)
            r_src = t.shape[0]; c_src = t.shape[1]
            out_t = torch.zeros(out_dim, rank, dtype=torch.float16)
            out_t[:min(r_src, out_dim), :min(c_src, rank)] = t[:min(r_src, out_dim), :min(c_src, rank)]
            lora_B[mod][li] = out_t.contiguous()

    return lora_A, lora_B, sorted(covered)


# ═════════════════════════════════════════════════════════════════════════════
# WeightPool — 每个 rank 选 1 个 adapter
# ═════════════════════════════════════════════════════════════════════════════

class WeightPool:
    def __init__(self, base_model_dir: str, registry: List[Tuple[str, int]], seed: int = 42):
        self.seed = seed
        print("\n─── Loading Llama-2-7B base weights ───────────────────────")
        self.base = load_base_model(base_model_dir)

        by_rank: Dict[int, List[Tuple[str, int]]] = {r: [] for r in TARGET_RANKS}
        for path, hint in registry:
            bucket = min(TARGET_RANKS, key=lambda r: abs(r - hint))
            by_rank[bucket].append((path, hint))

        self.adapters: Dict[int, Tuple[dict, dict, List[str], str]] = {}
        print("\n─── Loading LoRA adapters (1 per rank) ─────────────────────")
        for rank in TARGET_RANKS:
            candidates = by_rank[rank]
            if not candidates:
                print(f"  [WARN] rank={rank}: no adapters available"); continue
            rng = random.Random(seed ^ rank)
            path, hint = rng.choice(candidates)
            if not os.path.isdir(path):
                for p2, h2 in candidates:
                    if os.path.isdir(p2): path, hint = p2, h2; break
            if not os.path.isdir(path):
                print(f"  [WARN] rank={rank}: no valid path found"); continue
            try:
                raw = _load_adapter_raw(path)
                cfg = _read_adapter_config(path)
                actual_rank = int(cfg.get("r", hint))
                lA, lB, covered = parse_adapter(raw, rank)
                self.adapters[rank] = (lA, lB, covered, path)
                print(f"  [OK  ] rank={rank:3d}  actual_r={actual_rank:3d}  "
                      f"modules={covered}  path={os.path.basename(path)}")
            except Exception as e:
                print(f"  [ERR ] rank={rank}: {e}")
        print()

    def get_adapter(self, rank: int):
        if rank not in self.adapters:
            nearest = min(self.adapters.keys(), key=lambda r: abs(r - rank), default=None)
            if nearest is None: return {}, {}, []
            rank = nearest
        lA, lB, covered, _ = self.adapters[rank]
        return lA, lB, covered


# ═════════════════════════════════════════════════════════════════════════════
# LayerBuffers — 32 层全部 stacked GPU 张量
# ═════════════════════════════════════════════════════════════════════════════

class LayerBuffers:
    """
    构建 32 层所有 kernel 所需的 stacked GPU 张量：

      Layer kernel 1: run_qkv_lora
        qkv_A[li]: (nu, 3*max_r, H)       lora A for q+k+v
        qkv_B[li]: (nu, Q+2KV, max_r)     lora B for q+k+v (flattened along out_dim)

      Layer kernel 2: run_lora_a_sgemm + run_lora_b_sgemm  (o_proj)
        o_A[li]:   (nu, max_r, Q)          lora A: input=Q, output=r
        o_B[li]:   (nu, H, max_r)          lora B: input=r, output=H

      Layer kernel 3: run_gate_up_lora
        gu_A[li]:  (nu, 2*max_r, H)        lora A for gate+up
        gu_B[li]:  (nu, 2*FFN, max_r)      lora B for gate+up

      Layer kernel 4: run_lora_a_sgemm + run_lora_b_sgemm  (down_proj)
        dw_A[li]:  (nu, max_r, FFN)        lora A: input=FFN
        dw_B[li]:  (nu, H, max_r)          lora B: output=H

    ★ NOTE on weight tensor conventions (must match sgemm_lora_a / sgemm_lora_b):
      sgemm_lora_a_fwd   weights shape: (num_lora, stack_num*r, K)
                                        i.e.  (nu, r, in_dim)   for stack_num=1
      sgemm_lora_b_fwd   weights shape: (num_lora, out_dim, r)
      qkv_lora_b_fwd     weights shape: (num_lora, Q+2KV, r)    — lora_A already applied inside run_qkv_lora
      gate_up_lora_b_fwd weights shape: (num_lora, 2*FFN, r)    — same, lora_A applied inside
    """

    def __init__(self, pool: WeightPool, ranks: List[int], device: str):
        bs       = len(ranks)
        max_rank = max(ranks)

        # Build unique-adapter mapping: same rank → same adapter
        seen: Dict[int, int] = {}
        unique_ranks: List[int] = []
        self.weight_indices: List[int] = []
        for r in ranks:
            if r not in seen:
                seen[r] = len(seen)
                unique_ranks.append(r)
            self.weight_indices.append(seen[r])

        self.unique_ranks = unique_ranks
        self.num_unique   = len(unique_ranks)
        self.bs           = bs
        self.max_rank     = max_rank
        self.device       = device
        nu                = self.num_unique

        # ── Allocate GPU buffers ─────────────────────────────────────────
        def _z(*shape): return torch.zeros(*shape, dtype=DTYPE, device=device)

        self.qkv_A = [_z(nu, 3*max_rank, HIDDEN_DIM)   for _ in range(NUM_LAYERS)]
        self.qkv_B = [_z(nu, QKV_OUT_DIM,  max_rank)   for _ in range(NUM_LAYERS)]
        self.o_A   = [_z(nu, max_rank,   Q_OUT_DIM)    for _ in range(NUM_LAYERS)]
        self.o_B   = [_z(nu, HIDDEN_DIM, max_rank)     for _ in range(NUM_LAYERS)]
        self.gu_A  = [_z(nu, 2*max_rank, HIDDEN_DIM)   for _ in range(NUM_LAYERS)]
        self.gu_B  = [_z(nu, 2*FFN_DIM,  max_rank)     for _ in range(NUM_LAYERS)]
        self.dw_A  = [_z(nu, max_rank,   FFN_DIM)      for _ in range(NUM_LAYERS)]
        self.dw_B  = [_z(nu, HIDDEN_DIM, max_rank)     for _ in range(NUM_LAYERS)]

        # Fixed offset tensors (tp=1, single GPU)
        # output_offset for run_qkv_lora: MUST be length-4
        self.qkv_output_offset = torch.tensor(
            QKV_OUTPUT_OFFSET, dtype=torch.int32, device=device)  # len=4 ✓
        self.max_qkv_out_dim   = MAX_QKV_OUT_DIM

        # output_offset for run_gate_up_lora: length-3
        self.gu_output_offset = torch.tensor(
            GU_OUTPUT_OFFSET, dtype=torch.int32, device=device)   # len=3

        # output_offset for run_lora_b_sgemm (o_proj / down_proj): length-2
        self.o_output_offset  = torch.tensor([0, HIDDEN_DIM], dtype=torch.int32, device=device)
        self.dw_output_offset = torch.tensor([0, HIDDEN_DIM], dtype=torch.int32, device=device)

        # ── Fill real adapter weights ─────────────────────────────────────
        for ui, rank in enumerate(unique_ranks):
            lA, lB, covered = pool.get_adapter(rank)
            r = rank

            for li in range(NUM_LAYERS):
                # ── qkv_A: (nu, 3r, H) = [q_A; k_A; v_A] along dim-1 ──
                q_a = lA.get("q_proj", [None]*NUM_LAYERS)[li]
                k_a = lA.get("k_proj", [None]*NUM_LAYERS)[li]
                v_a = lA.get("v_proj", [None]*NUM_LAYERS)[li]
                if q_a is not None:
                    self.qkv_A[li][ui, :r,      :] = q_a.to(device, DTYPE)
                if k_a is not None:
                    self.qkv_A[li][ui, r:2*r,   :] = k_a.to(device, DTYPE)
                if v_a is not None:
                    self.qkv_A[li][ui, 2*r:3*r, :] = v_a.to(device, DTYPE)

                # ── qkv_B: (nu, Q+2KV, r) = [q_B; k_B; v_B] along dim-1 ──
                q_b = lB.get("q_proj", [None]*NUM_LAYERS)[li]
                k_b = lB.get("k_proj", [None]*NUM_LAYERS)[li]
                v_b = lB.get("v_proj", [None]*NUM_LAYERS)[li]
                if q_b is not None:
                    self.qkv_B[li][ui, :Q_OUT_DIM, :r]                        = q_b[:Q_OUT_DIM,  :r].to(device, DTYPE)
                if k_b is not None:
                    self.qkv_B[li][ui, Q_OUT_DIM:Q_OUT_DIM+KV_OUT_DIM, :r]   = k_b[:KV_OUT_DIM, :r].to(device, DTYPE)
                if v_b is not None:
                    self.qkv_B[li][ui, Q_OUT_DIM+KV_OUT_DIM:, :r]            = v_b[:KV_OUT_DIM, :r].to(device, DTYPE)

                # ── o_proj: A(nu,r,Q) B(nu,H,r) ──
                oa = lA.get("o_proj", [None]*NUM_LAYERS)[li]
                ob = lB.get("o_proj", [None]*NUM_LAYERS)[li]
                if oa is not None:
                    self.o_A[li][ui, :r, :] = oa[:r, :Q_OUT_DIM].to(device, DTYPE)
                if ob is not None:
                    self.o_B[li][ui, :, :r] = ob[:HIDDEN_DIM, :r].to(device, DTYPE)

                # ── gate_up: A(nu,2r,H) B(nu,2FFN,r) ──
                ga = lA.get("gate_proj", [None]*NUM_LAYERS)[li]
                ua = lA.get("up_proj",   [None]*NUM_LAYERS)[li]
                gb = lB.get("gate_proj", [None]*NUM_LAYERS)[li]
                ub = lB.get("up_proj",   [None]*NUM_LAYERS)[li]
                if ga is not None:
                    self.gu_A[li][ui, :r,    :] = ga[:r, :HIDDEN_DIM].to(device, DTYPE)
                if ua is not None:
                    self.gu_A[li][ui, r:2*r, :] = ua[:r, :HIDDEN_DIM].to(device, DTYPE)
                if gb is not None:
                    self.gu_B[li][ui, :FFN_DIM,          :r] = gb[:FFN_DIM, :r].to(device, DTYPE)
                if ub is not None:
                    self.gu_B[li][ui, FFN_DIM:2*FFN_DIM, :r] = ub[:FFN_DIM, :r].to(device, DTYPE)

                # ── down_proj: A(nu,r,FFN) B(nu,H,r) ──
                da = lA.get("down_proj", [None]*NUM_LAYERS)[li]
                db = lB.get("down_proj", [None]*NUM_LAYERS)[li]
                if da is not None:
                    self.dw_A[li][ui, :r, :] = da[:r, :FFN_DIM].to(device, DTYPE)
                if db is not None:
                    self.dw_B[li][ui, :, :r] = db[:HIDDEN_DIM, :r].to(device, DTYPE)

        # Make contiguous for kernel efficiency
        for li in range(NUM_LAYERS):
            self.qkv_A[li] = self.qkv_A[li].contiguous()
            self.qkv_B[li] = self.qkv_B[li].contiguous()
            self.o_A[li]   = self.o_A[li].contiguous()
            self.o_B[li]   = self.o_B[li].contiguous()
            self.gu_A[li]  = self.gu_A[li].contiguous()
            self.gu_B[li]  = self.gu_B[li].contiguous()
            self.dw_A[li]  = self.dw_A[li].contiguous()
            self.dw_B[li]  = self.dw_B[li].contiguous()

    def release(self):
        for attr in ("qkv_A","qkv_B","o_A","o_B","gu_A","gu_B","dw_A","dw_B"):
            lst = getattr(self, attr)
            for t in lst: del t
            setattr(self, attr, [])
        torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════
# 32 层真实推理 forward
# ═════════════════════════════════════════════════════════════════════════════

def _run_lora_b_sgemm(backend, x, weights, output_offset, base_output):
    """o_proj / down_proj 的 lora B，兼容 triton 和 csgmv。"""
    if backend.name == "csgmv":
        return backend.run_lora_b_sgemm(
            x=x, weights=weights,
            output_offset=output_offset,
            base_output=base_output,
        )
    else:  # triton
        return backend.run_lora_b_sgemm(
            x=x, weights=weights,
            base_output=base_output,
        )


def _run_qkv_lora(backend, x, qkv_lora_a, qkv_lora_b,
                  output_offset, max_qkv_out_dim, base_output):
    """QKV lora，兼容 triton 和 csgmv。"""
    if backend.name == "csgmv":
        return backend.run_qkv_lora(
            x=x, qkv_lora_a=qkv_lora_a, qkv_lora_b=qkv_lora_b,
            output_offset=output_offset,
            max_qkv_out_dim=max_qkv_out_dim,
            base_output=base_output,
        )
    else:  # triton
        return backend.run_qkv_lora(
            x=x, qkv_lora_a=qkv_lora_a, qkv_lora_b=qkv_lora_b,
            output_offset=output_offset,
            max_qkv_out_dim=max_qkv_out_dim,
            base_output=base_output,
        )
    # NOTE: qkv 两个后端签名相同（都需要 output_offset），保留包装统一风格


def _run_gate_up_lora(backend, x, gate_up_lora_a, gate_up_lora_b,
                      output_offset, base_output):
    """gate_up_proj lora，兼容 triton 和 csgmv。"""
    if backend.name == "csgmv":
        return backend.run_gate_up_lora(
            x=x, gate_up_lora_a=gate_up_lora_a, gate_up_lora_b=gate_up_lora_b,
            output_offset=output_offset,
            base_output=base_output,
        )
    else:  # triton — output_offset 不在签名中，通过 output_dim 内部推导
        return backend.run_gate_up_lora(
            x=x, gate_up_lora_a=gate_up_lora_a, gate_up_lora_b=gate_up_lora_b,
            base_output=base_output,
        )


def run_32layer_forward(
    backend,
    pool:    WeightPool,
    bufs:    LayerBuffers,
    seq_len: int,
    device:  str,
) -> torch.Tensor:
    """
    模拟 Llama-2-7B 32 层 decoder forward，每层调用 6 次 LoRA kernel
    （与 layers.py 实际调用路径完全对齐）：

      1. run_qkv_lora         (qkv_proj)
      2. run_lora_a_sgemm     (o_proj, part-A)
         run_lora_b_sgemm     (o_proj, part-B)
      3. run_gate_up_lora     (gate_up_proj)
      4. run_lora_a_sgemm     (down_proj, part-A)
         run_lora_b_sgemm     (down_proj, part-B)
    """
    bs        = bufs.bs
    total_tok = bs * seq_len
    h = torch.randn(total_tok, HIDDEN_DIM, dtype=DTYPE, device=device) * 0.02

    qkv_off  = bufs.qkv_output_offset  # 4-element: [0, Q, Q+KV, Q+2KV]
    gu_off   = bufs.gu_output_offset   # 3-element: [0, FFN, 2*FFN]
    o_off    = torch.tensor([0, HIDDEN_DIM], dtype=torch.int32, device=device)
    dw_off   = torch.tensor([0, HIDDEN_DIM], dtype=torch.int32, device=device)
    max_qkv_dim = bufs.max_qkv_out_dim

    for li in range(NUM_LAYERS):
        W_q  = pool.base[li]["q_proj"]      # (Q_OUT, H)
        W_k  = pool.base[li]["k_proj"]      # (KV_OUT, H)
        W_v  = pool.base[li]["v_proj"]      # (KV_OUT, H)
        W_o  = pool.base[li]["o_proj"]      # (H, Q_OUT)
        W_g  = pool.base[li]["gate_proj"]   # (FFN, H)
        W_u  = pool.base[li]["up_proj"]     # (FFN, H)
        W_d  = pool.base[li]["down_proj"]   # (H, FFN)

        # ── 1. QKV projection ─────────────────────────────────────────────
        base_qkv = torch.cat([h @ W_q.T, h @ W_k.T, h @ W_v.T], dim=-1)
        qkv_out  = _run_qkv_lora(
            backend, h, bufs.qkv_A[li], bufs.qkv_B[li],
            output_offset=qkv_off, max_qkv_out_dim=max_qkv_dim,
            base_output=base_qkv,
        )
        attn_out = qkv_out[:, :Q_OUT_DIM].contiguous()

        # ── 2. o_proj ─────────────────────────────────────────────────────
        base_o     = attn_out @ W_o.T
        lora_mid_o = backend.run_lora_a_sgemm(attn_out, bufs.o_A[li])
        h          = _run_lora_b_sgemm(
            backend, lora_mid_o, bufs.o_B[li],
            output_offset=o_off, base_output=base_o)

        # ── 3. gate_up_proj ───────────────────────────────────────────────
        base_gu = torch.cat([h @ W_g.T, h @ W_u.T], dim=-1)
        ffn_out = _run_gate_up_lora(
            backend, h, bufs.gu_A[li], bufs.gu_B[li],
            output_offset=gu_off, base_output=base_gu,
        )
        gate    = ffn_out[:, :FFN_DIM].contiguous()
        up      = ffn_out[:, FFN_DIM:].contiguous()
        ffn_act = (torch.sigmoid(gate) * gate * up).contiguous()

        # ── 4. down_proj ──────────────────────────────────────────────────
        base_d     = ffn_act @ W_d.T
        lora_mid_d = backend.run_lora_a_sgemm(ffn_act, bufs.dw_A[li])
        h          = _run_lora_b_sgemm(
            backend, lora_mid_d, bufs.dw_B[li],
            output_offset=dw_off, base_output=base_d)

    return h

# ═════════════════════════════════════════════════════════════════════════════
# 计时
# ═════════════════════════════════════════════════════════════════════════════

def time_fn(fn, warmup: int, repeat: int) -> Tuple[float, float, float]:
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    lats = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return float(np.median(lats)), float(lats.mean()), float(lats.std())


# ═════════════════════════════════════════════════════════════════════════════
# 特征工程
# ═════════════════════════════════════════════════════════════════════════════

def compute_features(ranks: List[int], seq_len: int) -> Dict[str, float]:
    bs        = len(ranks)
    max_rank  = max(ranks); min_rank = min(ranks); sum_rank = sum(ranks)
    mean_rank = sum_rank / bs
    var_rank  = float(np.var(ranks)); std_rank = float(np.std(ranks))
    rank_ratio = max_rank / min_rank if min_rank > 0 else float(max_rank)
    total_tok = bs * seq_len
    return {
        "bs_x_max_rank":       float(bs * max_rank),
        "sum_rank":            float(sum_rank),
        "tok_x_max_rank":      float(total_tok * max_rank),
        "tok_x_sum_rank":      float(sum_rank * seq_len),
        "bs":                  float(bs),
        "max_rank":            float(max_rank),
        "min_rank":            float(min_rank),
        "mean_rank":           float(mean_rank),
        "seq_len":             float(seq_len),
        "total_tokens":        float(total_tok),
        "var_rank":            var_rank,
        "std_rank":            std_rank,
        "rank_ratio":          float(rank_ratio),
        "max_minus_min":       float(max_rank - min_rank),
        "bs_x_sum_rank":       float(bs * sum_rank),
        "bs_x_mean_rank":      float(bs * mean_rank),
        "bs_x_min_rank":       float(bs * min_rank),
        "sum_x_max_rank":      float(sum_rank * max_rank),
        "tok_x_mean_rank":     float(total_tok * mean_rank),
        "sqrt_bs_x_max_rank":  float(np.sqrt(bs * max_rank)),
        "sqrt_sum_rank":       float(np.sqrt(sum_rank)),
        "log1p_bs_x_max_rank": float(np.log1p(bs * max_rank)),
        "log1p_sum_rank":      float(np.log1p(sum_rank)),
        "sum_rank_sq":         float(sum_rank ** 2),
        "max_rank_sq":         float(max_rank ** 2),
    }

CANDIDATE_PREDICTORS = [
    "bs_x_max_rank", "sum_rank", "tok_x_max_rank", "tok_x_sum_rank",
    "bs", "max_rank", "min_rank", "mean_rank", "seq_len", "total_tokens",
    "var_rank", "std_rank", "rank_ratio", "max_minus_min",
    "bs_x_sum_rank", "bs_x_mean_rank", "bs_x_min_rank",
    "sum_x_max_rank", "tok_x_mean_rank",
    "sqrt_bs_x_max_rank", "sqrt_sum_rank",
    "log1p_bs_x_max_rank", "log1p_sum_rank",
    "sum_rank_sq", "max_rank_sq",
]


# ═════════════════════════════════════════════════════════════════════════════
# 核心 benchmark
# ═════════════════════════════════════════════════════════════════════════════

def benchmark_one(
    backend_name: str,
    ranks:        List[int],
    pool:         WeightPool,
    seq_len:      int,
    warmup:       int,
    repeat:       int,
) -> Dict:
    bs        = len(ranks)
    is_extend = (seq_len > 1)

    bufs = LayerBuffers(pool, ranks, DEVICE)
    backend = make_backend(backend_name, bufs.num_unique, DEVICE)
    prepare_backend(backend, bs, seq_len, is_extend,
                    bufs.weight_indices, bufs.unique_ranks, DEVICE)

    def fn():
        run_32layer_forward(backend, pool, bufs, seq_len, DEVICE)

    median_ms, mean_ms, std_ms = time_fn(fn, warmup, repeat)

    bufs.release()
    del backend
    torch.cuda.empty_cache()

    feats = compute_features(ranks, seq_len)
    return {
        "backend":      backend_name,
        "ranks_str":    str(ranks),
        "seq_len":      seq_len,
        "batch_size":   bs,
        "max_rank_int": max(ranks),
        "sum_rank_int": sum(ranks),
        "num_unique":   bufs.num_unique,
        "median_ms":    median_ms,
        "mean_ms":      mean_ms,
        "std_ms":       std_ms,
        **feats,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Rank 配置生成
# ═════════════════════════════════════════════════════════════════════════════

def build_rank_configs(batch_sizes: List[int], rank_values: List[int], mode: str = "all") -> List[List[int]]:
    configs = []
    if mode in ("homogeneous", "all"):
        for bs in batch_sizes:
            for r in rank_values:
                configs.append([r] * bs)
    if mode in ("heterogeneous", "all"):
        for bs in batch_sizes:
            if bs < 2: continue
            for r_lo, r_hi in itertools.combinations(rank_values, 2):
                half    = bs // 2
                quarter = max(1, bs // 4)
                configs.append([r_lo]*half         + [r_hi]*(bs-half))
                configs.append([r_lo]*quarter      + [r_hi]*(bs-quarter))
                configs.append([r_lo]*(bs-quarter) + [r_hi]*quarter)
    seen, unique = set(), []
    for c in configs:
        k = tuple(sorted(c))
        if k not in seen: seen.add(k); unique.append(c)
    return unique


def run_all_configs(backend_name, configs, pool, seq_len, warmup, repeat, verbose):
    results = []
    total   = len(configs)
    tag     = "decode" if seq_len == 1 else f"prefill(seq={seq_len})"
    if verbose:
        print(f"\n  [{backend_name:6s}] {tag}  {total} configs")
        print(f"  {'#':>4}  {'bs':>3}  {'ranks':<28}  {'max_r':>5}  "
              f"{'sum_r':>5}  {'median_ms':>10}  {'std':>7}")
        print(f"  {'─'*4}  {'─'*3}  {'─'*28}  {'─'*5}  {'─'*5}  {'─'*10}  {'─'*7}")
    for done, ranks in enumerate(configs, 1):
        row = benchmark_one(backend_name, ranks, pool, seq_len, warmup, repeat)
        results.append(row)
        if verbose:
            rs = str(ranks)[:28]
            print(f"  {done:>4}  {row['batch_size']:>3}  {rs:<28}  "
                  f"{row['max_rank_int']:>5}  {row['sum_rank_int']:>5}  "
                  f"{row['median_ms']:>10.4f}  {row['std_ms']:>7.4f}")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 拟合分析
# ═════════════════════════════════════════════════════════════════════════════

def ols_fit_1d(X, Y):
    from scipy.stats import linregress
    sl, ic, r, *_ = linregress(X, Y)
    return float(sl), float(ic), float(r**2)

def ols_fit_nd(Xm, Y):
    coef, *_ = np.linalg.lstsq(Xm, Y, rcond=None)
    Yp = Xm @ coef
    ss_res = float(np.sum((Y-Yp)**2))
    ss_tot = float(np.sum((Y-Y.mean())**2))
    return coef, 1.0-ss_res/ss_tot if ss_tot>1e-12 else 0.0

def _rmse(Yt, Yp): return float(np.sqrt(np.mean((Yt-Yp)**2)))
def _mae(Yt, Yp):  return float(np.mean(np.abs(Yt-Yp)))


def fit_and_explore(results: List[Dict], backend: str, mode: str, top_k: int = 5):
    try:
        from scipy.stats import shapiro; HAS_SCI = True
    except ImportError:
        HAS_SCI = False

    Y  = np.array([r["median_ms"] for r in results], dtype=float)
    n  = len(Y)
    Ym = Y.mean()
    is_dec = mode.startswith("decode")

    if is_dec:
        bp, mp = "bs_x_max_rank",  "sum_rank"
        bl, ml = "bs×max_rank",    "Σrank_i"
    else:
        bp, mp = "tok_x_max_rank", "tok_x_sum_rank"
        bl, ml = "tok×max_rank",   "Σrank·seqlen"

    print(f"\n{'='*80}")
    print(f"  FIT — {backend.upper()} [{mode}]  n={n}  "
          f"latency∈[{Y.min():.3f},{Y.max():.3f}] ms")
    print(f"{'='*80}")

    def _pfit(label, Yp_arr, formula):
        res  = Y - Yp_arr
        r2   = 1 - np.sum(res**2)/max(np.sum((Y-Ym)**2), 1e-12)
        rms  = _rmse(Y, Yp_arr); ma = _mae(Y, Yp_arr)
        mxe  = float(np.abs(res).max())
        mape = float(np.abs(res/np.where(Y!=0, Y, 1e-9)).mean())*100
        print(f"\n  ┌─ {label}")
        print(f"  │  {formula}")
        print(f"  │  R²={r2:.6f}  RMSE={rms:.5f}ms  MAE={ma:.5f}ms  "
              f"MaxErr={mxe:.5f}ms  MAPE={mape:.2f}%")
        if HAS_SCI and n >= 8:
            W, p = shapiro(res)
            print(f"  │  Shapiro-Wilk W={W:.4f} p={p:.4f} {'✓' if p>0.05 else '✗'}")
        print(f"  └{'─'*62}")
        return float(r2), rms, ma

    Xb = np.array([r[bp] for r in results])
    ab, bb, _ = ols_fit_1d(Xb, Y)
    r2b, rb, _ = _pfit(f"[BGMV]  latency ~ α·({bl}) + β",
                        ab*Xb+bb, f"latency = {ab:+.8f}·({bl}) + {bb:+.5f}")

    Xm = np.array([r[mp] for r in results])
    am, bm, _ = ols_fit_1d(Xm, Y)
    r2m, rm, _ = _pfit(f"[MBGMV] latency ~ α·({ml}) + β",
                        am*Xm+bm, f"latency = {am:+.8f}·({ml}) + {bm:+.5f}")

    # Single predictor sweep
    fit_rows = []
    for pred in CANDIDATE_PREDICTORS:
        Xp = np.array([r.get(pred, np.nan) for r in results])
        if np.any(np.isnan(Xp)) or np.nanstd(Xp) < 1e-10: continue
        ap, bp2, r2p = ols_fit_1d(Xp, Y)
        Yp2 = ap*Xp + bp2
        fit_rows.append({"backend":backend,"mode":mode,"tag":"single",
            "predictor":pred,"alpha":ap,"beta":bp2,"r2":r2p,
            "rmse_ms":_rmse(Y,Yp2),"mae_ms":_mae(Y,Yp2),
            "max_err_ms":float(np.abs(Y-Yp2).max()),"n_points":n,
            "formula":f"{ap:+.8f}·{pred}+{bp2:+.5f}"})
    fit_rows.sort(key=lambda x: x["r2"], reverse=True)

    print(f"\n  ── Top-15 single predictors [{mode}] ──")
    print(f"  {'#':>3}  {'Predictor':<26}  {'α':>14}  {'β':>11}  {'R²':>8}  {'RMSE':>8}  Notes")
    for i, fr in enumerate(fit_rows[:15]):
        notes = []
        if fr["predictor"] == bp: notes.append("BGMV")
        if fr["predictor"] == mp: notes.append("MBGMV")
        if i == 0:                notes.append("★BEST")
        print(f"  {i+1:>3}  {fr['predictor']:<26}  {fr['alpha']:>+14.8f}  "
              f"{fr['beta']:>+11.5f}  {fr['r2']:>8.6f}  {fr['rmse_ms']:>8.5f}  "
              f"{', '.join(notes)}")

    top_preds = [fr["predictor"] for fr in fit_rows[:top_k]]
    mv_rows   = []
    bpr2, bprow = -1., None
    btr2, btrow = -1., None

    print(f"\n  ── Pairs ──")
    for p1, p2 in itertools.combinations(top_preds, 2):
        X1  = np.array([r.get(p1, 0.) for r in results])
        X2  = np.array([r.get(p2, 0.) for r in results])
        Xm2 = np.column_stack([X1, X2, np.ones(n)])
        coef, r2mv = ols_fit_nd(Xm2, Y); Yp = Xm2@coef; rms = _rmse(Y, Yp)
        row = {"backend":backend,"mode":mode,"tag":"pair","predictor":f"{p1}+{p2}",
               "formula":f"{coef[0]:+.6f}·{p1}+{coef[1]:+.6f}·{p2}+{coef[2]:+.6f}",
               "alpha":coef[0],"beta":coef[2],"r2":r2mv,"rmse_ms":rms,
               "mae_ms":_mae(Y,Yp),"max_err_ms":float(np.abs(Y-Yp).max()),"n_points":n}
        mv_rows.append(row)
        if r2mv > bpr2: bpr2, bprow = r2mv, row
        print(f"  {'★' if r2mv==bpr2 else ' '} {p1}+{p2:<40}  R²={r2mv:.6f}  RMSE={rms:.5f}")

    print(f"\n  ── Triples ──")
    for p1, p2, p3 in itertools.combinations(top_preds, 3):
        X1  = np.array([r.get(p1, 0.) for r in results])
        X2  = np.array([r.get(p2, 0.) for r in results])
        X3  = np.array([r.get(p3, 0.) for r in results])
        Xm3 = np.column_stack([X1, X2, X3, np.ones(n)])
        coef, r2mv = ols_fit_nd(Xm3, Y); Yp = Xm3@coef; rms = _rmse(Y, Yp)
        row = {"backend":backend,"mode":mode,"tag":"triple",
               "predictor":f"{p1}+{p2}+{p3}",
               "formula":f"{coef[0]:+.6f}·{p1}+{coef[1]:+.6f}·{p2}+{coef[2]:+.6f}·{p3}+{coef[3]:+.6f}",
               "alpha":coef[0],"beta":coef[3],"r2":r2mv,"rmse_ms":rms,
               "mae_ms":_mae(Y,Yp),"max_err_ms":float(np.abs(Y-Yp).max()),"n_points":n}
        mv_rows.append(row)
        if r2mv > btr2: btr2, btrow = r2mv, row
        print(f"  {'★' if r2mv==btr2 else ' '} {p1}+{p2}+{p3:<30}  R²={r2mv:.6f}  RMSE={rms:.5f}")

    best = fit_rows[0] if fit_rows else {"predictor":"N/A","alpha":0.,"beta":0.,"r2":0.,"rmse_ms":0.}
    delta   = r2b - r2m
    verdict = ("BGMV≈MBGMV" if abs(delta)<0.02 else ("BGMV better" if delta>0 else "MBGMV better"))

    print(f"\n  {'─'*78}\n  SUMMARY — {backend.upper()} [{mode}]")
    print(f"  {'Model':<16}  {'X':<28}  {'α':>12}  {'β':>10}  {'R²':>8}  {'RMSE':>9}")
    print(f"  {'BGMV':<16}  {bl:<28}  {ab:>+12.6f}  {bb:>+10.5f}  {r2b:>8.6f}  {rb:>9.5f}")
    print(f"  {'MBGMV':<16}  {ml:<28}  {am:>+12.6f}  {bm:>+10.5f}  {r2m:>8.6f}  {rm:>9.5f}")
    print(f"  {'★ Best single':<16}  {best['predictor']:<28}  "
          f"{best['alpha']:>+12.6f}  {best['beta']:>+10.5f}  "
          f"{best['r2']:>8.6f}  {best['rmse_ms']:>9.5f}")
    if bprow: print(f"  {'★ Best pair':<16}  {bprow['predictor'][:28]:<28}  "
                    f"{'':>12}  {'':>10}  {bpr2:>8.6f}  {bprow['rmse_ms']:>9.5f}")
    if btrow: print(f"  {'★ Best triple':<16}  {btrow['predictor'][:28]:<28}  "
                    f"{'':>12}  {'':>10}  {btr2:>8.6f}  {btrow['rmse_ms']:>9.5f}")
    print(f"\n  Interpretation: ΔR²(BGMV-MBGMV)={delta:+.4f} → {verdict}")

    summary = {"backend":backend,"mode":mode,"n":n,
               "bgmv": {"pred":bp,"alpha":ab,"beta":bb,"r2":r2b,"rmse":rb},
               "mbgmv":{"pred":mp,"alpha":am,"beta":bm,"r2":r2m,"rmse":rm},
               "best_single":best,"best_pair":bprow,"best_triple":btrow}
    return summary, fit_rows + mv_rows


# ═════════════════════════════════════════════════════════════════════════════
# CSV 保存
# ═════════════════════════════════════════════════════════════════════════════

def save_csv(rows: List[Dict], path: str, label: str = ""):
    if not rows: return
    all_keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in all_keys})
    print(f"  [CSV] [{label}] → {path}  ({len(rows)} rows)")


# ═════════════════════════════════════════════════════════════════════════════
# 主函数
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Llama-2-7B 32层真实推理路径 LoRA Kernel 性能建模",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend",           default="both", choices=["triton","csgmv","both"])
    parser.add_argument("--base-model",        default=DEFAULT_BASE_MODEL)
    parser.add_argument("--batch-sizes",       type=int, nargs="+", default=[4,8,16,32])
    parser.add_argument("--rank-values",       type=int, nargs="+", default=[8,16,32,64])
    parser.add_argument("--prefill-seq-lens",  type=int, nargs="+", default=[64,128,256])
    parser.add_argument("--mode",              default="all",
                        choices=["homogeneous","heterogeneous","all"])
    parser.add_argument("--warmup",            type=int, default=10)
    parser.add_argument("--repeat",            type=int, default=100)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--fit",               action="store_true")
    parser.add_argument("--top-k-multivar",    type=int, default=5)
    parser.add_argument("--save-dec-csv",      default=None)
    parser.add_argument("--save-pre-csv",      default=None)
    parser.add_argument("--save-fit-csv",      default=None)
    parser.add_argument("--decode-only",       action="store_true")
    parser.add_argument("--prefill-only",      action="store_true")
    parser.add_argument("--quiet",             action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA GPU"
    assert not (args.decode_only and args.prefill_only)

    print("="*80)
    print("  Llama-2-7B 32层真实推理路径 LoRA Kernel 性能建模")
    print("  ★ 每层 6 次 kernel: qkv_lora | o_proj(A+B) | gate_up_lora | down_proj(A+B)")
    print("  ★ 每个 rank 桶选 1 个真实 adapter，按实际 target_modules 填充")
    print("  ★ 无真实权重的模块以零矩阵填充（rank=0 等效，kernel 直接 no-op）")
    print("="*80)
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  Backend(s)  : {args.backend}")
    print(f"  Base model  : {args.base_model}")
    print(f"  Batch sizes : {args.batch_sizes}")
    print(f"  Rank values : {args.rank_values}")
    print(f"  Warmup/rep  : {args.warmup}/{args.repeat}")

    pool    = WeightPool(args.base_model, ADAPTER_REGISTRY, seed=args.seed)
    configs = build_rank_configs(args.batch_sizes, args.rank_values, args.mode)
    backends = ["triton","csgmv"] if args.backend=="both" else [args.backend]
    print(f"\n  Rank configs: {len(configs)}")

    dec_res:      List[Dict] = []
    pre_res:      List[Dict] = []
    fit_rows_all: List[Dict] = []
    summaries:    List[Dict] = []

    if not args.prefill_only:
        print(f"\n{'─'*80}\n  PHASE 1 — DECODE (seq_len=1)\n{'─'*80}")
        for bk in backends:
            res = run_all_configs(bk, configs, pool, 1,
                                  args.warmup, args.repeat, not args.quiet)
            dec_res.extend(res)
            if args.fit:
                sm, rows = fit_and_explore(res, bk, "decode", args.top_k_multivar)
                fit_rows_all.extend(rows); summaries.append(sm)

    if not args.decode_only:
        print(f"\n{'─'*80}\n  PHASE 2 — PREFILL\n{'─'*80}")
        per_sl: Dict[int, List[Dict]] = {}
        for sl in args.prefill_seq_lens:
            per_sl[sl] = []
            for bk in backends:
                res = run_all_configs(bk, configs, pool, sl,
                                      args.warmup, args.repeat, not args.quiet)
                per_sl[sl].extend(res); pre_res.extend(res)
        if args.fit:
            for sl, res in per_sl.items():
                for bk in backends:
                    sub = [r for r in res if r["backend"]==bk]
                    if sub:
                        sm, rows = fit_and_explore(sub, bk, f"prefill_sl{sl}", args.top_k_multivar)
                        fit_rows_all.extend(rows); summaries.append(sm)
            print(f"\n{'─'*80}\n  POOLED PREFILL\n{'─'*80}")
            for bk in backends:
                sub = [r for r in pre_res if r["backend"]==bk]
                if sub:
                    sm, rows = fit_and_explore(sub, bk, "prefill_pooled", args.top_k_multivar)
                    fit_rows_all.extend(rows); summaries.append(sm)

    if summaries:
        print(f"\n{'='*80}\n  CONSOLIDATED RESULTS\n{'='*80}")
        print(f"  {'Backend':<8}  {'Mode':<22}  {'Model':<8}  {'Predictor':<28}  "
              f"{'α':>12}  {'β':>10}  {'R²':>8}  {'RMSE':>9}")
        for sm in summaries:
            for mn, md in [("BGMV",sm["bgmv"]),("MBGMV",sm["mbgmv"])]:
                print(f"  {sm['backend']:<8}  {sm['mode']:<22}  {mn:<8}  "
                      f"{md['pred']:<28}  {md['alpha']:>+12.6f}  "
                      f"{md['beta']:>+10.5f}  {md['r2']:>8.6f}  {md['rmse']:>9.5f}")

    if args.save_dec_csv and dec_res: save_csv(dec_res, args.save_dec_csv, "decode")
    if args.save_pre_csv and pre_res: save_csv(pre_res, args.save_pre_csv, "prefill")
    if args.fit and fit_rows_all:
        p = args.save_fit_csv or "fit_results.csv"
        save_csv(fit_rows_all, p, "fit")

    print(f"\n{'='*80}")
    print(f"  Done.  Dec={len(dec_res)}  Pre={len(pre_res)}  Fit={len(fit_rows_all)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()