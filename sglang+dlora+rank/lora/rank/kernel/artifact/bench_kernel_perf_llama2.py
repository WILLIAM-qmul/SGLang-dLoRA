"""
bench_kernel_perf_llama2.py
============================
Kernel-level microbenchmark: Triton (BGMV) vs CSGMV (MBGMV)
using REAL LoRA adapters with REAL rank diversity.

Experimental design:
  - Each benchmark call runs ALL LoRA-B modules in sequence
    (o_proj, qkv_proj, gate_up_proj, down_proj) — matching real inference
  - Total latency = sum over all modules (one per transformer layer pass)
  - Predictors: bs, ranks only (scheduler-observable, NO output_dim)
  - Both backends share IDENTICAL weight tensors for fair comparison

Fitting pipeline:
  1. BGMV baseline:  latency ~ α·(bs×max_rank) + β
  2. MBGMV baseline: latency ~ α·Σrank_i + β
  3. Exhaustive single-predictor OLS over 28 candidates
  4. Multi-variate OLS: best2 + best3 predictor combinations (optional)
  5. Residual diagnostics: plot predicted vs actual, residual histogram
  6. Save measurement CSV + fit-results CSV

Usage:
    python bench_kernel_perf_llama2.py --backend both --fit \\
        --save-csv results.csv --save-fit-csv fit.csv

    # With plots saved to disk:
    python bench_kernel_perf_llama2.py --backend both --fit \\
        --save-csv results.csv --save-fit-csv fit.csv --plot-dir ./plots
"""

import argparse
import csv
import itertools
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.lora.triton_ops import chunked_sgmv_lora_expand_forward, sgemm_lora_b_fwd
from sglang.srt.lora.utils import LoRABatchInfo

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda"
DTYPE  = torch.float16

CSGMV_MIN_BLOCK_M = 16
CSGMV_BLOCK_K     = 16
CSGMV_BLOCK_N     = 64

MODULES: Dict[str, Tuple[int, int]] = {
    "o_proj":       (4096,  4096),
    "qkv_proj":     (4096,  4096 * 3),
    "gate_up_proj": (4096,  11008 * 2),
    "down_proj":    (11008, 4096),
}

MODULE_LORA_B_SUFFIXES: Dict[str, List[str]] = {
    "o_proj":       ["self_attn.o_proj.lora_B.weight", "o_proj.lora_B.weight"],
    "qkv_proj":     ["self_attn.q_proj.lora_B.weight", "q_proj.lora_B.weight"],
    "gate_up_proj": ["mlp.gate_proj.lora_B.weight",    "gate_proj.lora_B.weight"],
    "down_proj":    ["mlp.down_proj.lora_B.weight",    "down_proj.lora_B.weight"],
}

TARGET_RANKS      = [8, 16, 32, 64]
ADAPTERS_PER_RANK = 1

ADAPTER_REGISTRY: List[Tuple[str, int]] = [
    ("/workspace/models/MUFFIN-Llama2-lora-7B",                        8),
    ("/workspace/models/CodeLlama-7b-hf-Text2VQL-LoRA",                8),
    ("/workspace/models/llama-2-7b-LORA-data-analyst",                16),
    ("/workspace/models/Llama-2-7b-hf-lora-r32-wnli-loraxs",         32),
    ("/workspace/models/llama-2-7b-chat-lora-adaptor",                64),
    ("/workspace/models/llama2-stable-7b-lora",                       64),
    ("/workspace/models/llava-llama-2-7b-chat-lightning-lora-preview", 64),
]


# ─────────────────────────────────────────────────────────────────────────────
# Adapter I/O
# ─────────────────────────────────────────────────────────────────────────────

def _read_config(path: str) -> Dict:
    f = os.path.join(path, "adapter_config.json")
    return json.load(open(f)) if os.path.exists(f) else {}


def _load_weights(path: str) -> Dict[str, torch.Tensor]:
    st  = os.path.join(path, "adapter_model.safetensors")
    bn  = os.path.join(path, "adapter_model.bin")
    if os.path.exists(st):
        from safetensors.torch import load_file
        return load_file(st, device="cpu")
    if os.path.exists(bn):
        return torch.load(bn, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No weights found in {path}")


def _find_lora_b(weights: Dict[str, torch.Tensor], module: str) -> Optional[torch.Tensor]:
    for key, t in weights.items():
        for suf in MODULE_LORA_B_SUFFIXES[module]:
            if key.endswith(suf) and t.dim() == 2:
                return t.to(torch.float16)
    return None


def _adapt_to_shape(
    lora_b: torch.Tensor,
    output_dim: int,
    target_rank: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    src_out, src_rank = lora_b.shape
    # align output dim
    if src_out != output_dim:
        reps = -(-output_dim // src_out)
        lora_b = lora_b.repeat(reps, 1)[:output_dim, :]
    # align rank dim
    if src_rank >= target_rank:
        lora_b = lora_b[:, :target_rank].contiguous()
    else:
        reps  = -(-target_rank // src_rank)
        lora_b = lora_b.repeat(1, reps)[:, :target_rank].contiguous()
        scale = float(lora_b.abs().float().mean()) * 0.01
        noise = torch.from_numpy(
            rng.standard_normal(lora_b.shape).astype(np.float16)
        ) * scale
        lora_b = lora_b + noise
    return lora_b.contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# AdapterPool
# ─────────────────────────────────────────────────────────────────────────────

class AdapterPool:
    def __init__(self, registry: List[Tuple[str, int]], seed: int = 42):
        self.seed = seed
        self._by_rank: Dict[int, List[Dict]] = {r: [] for r in TARGET_RANKS}
        self._w_cache: Dict[Tuple, torch.Tensor] = {}
        self._x_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        print("\n  ── Loading real LoRA adapters ──────────────────────────────")
        loaded = {r: 0 for r in TARGET_RANKS}
        for path, hint in registry:
            if not os.path.isdir(path):
                continue
            try:
                w    = _load_weights(path)
                cfg  = _read_config(path)
                rank = int(cfg.get("r", hint))
            except Exception as e:
                print(f"  [ERR ] {os.path.basename(path)}: {e}")
                continue
            bucket = min(TARGET_RANKS, key=lambda r: abs(r - rank))
            self._by_rank[bucket].append(w)
            loaded[bucket] += 1
            print(f"  [OK  ] {os.path.basename(path):50s} "
                  f"rank={rank:3d} → bucket r={bucket}")

        print(f"\n  Real adapters per rank: {loaded}")
        for r in TARGET_RANKS:
            if loaded[r] == 0:
                donor = self._find_donor(r)
                print(f"  [PLAN] rank={r:2d}: 0 real → derive from rank={donor} + noise")
            elif loaded[r] < ADAPTERS_PER_RANK:
                print(f"  [PLAN] rank={r:2d}: {loaded[r]} real → "
                      f"derive {ADAPTERS_PER_RANK - loaded[r]} more by perturbation")
        print()

    def _find_donor(self, target: int) -> int:
        for r in sorted(TARGET_RANKS, key=lambda r: abs(r - target)):
            if self._by_rank[r]:
                return r
        raise RuntimeError("All rank buckets empty.")

    def _det_rng(self, *tags) -> np.random.Generator:
        h = self.seed
        for t in tags:
            h ^= hash(str(t)) & 0xFFFF_FFFF
        return np.random.default_rng(h & 0xFFFF_FFFF_FFFF_FFFF)

    def _build_one(self, module: str, target_rank: int, output_dim: int, slot: int) -> torch.Tensor:
        rng = self._det_rng(module, target_rank, output_dim, slot)
        # try same-rank bucket first, then others by proximity
        for try_rank in sorted(TARGET_RANKS, key=lambda r: abs(r - target_rank)):
            bucket = self._by_rank[try_rank]
            if not bucket:
                continue
            for attempt in range(len(bucket)):
                raw = bucket[(slot + attempt) % len(bucket)]
                for try_mod in [module] + [m for m in MODULES if m != module]:
                    t = _find_lora_b(raw, try_mod)
                    if t is not None:
                        w = _adapt_to_shape(t, output_dim, target_rank, rng)
                        # add perturbation for derived slots
                        if try_rank != target_rank or slot >= len(bucket):
                            scale = float(w.float().std()) * 0.05 or 0.02
                            noise = torch.from_numpy(
                                rng.standard_normal(w.shape).astype(np.float16)
                            ) * scale
                            w = (w + noise).contiguous()
                        return w
        # pure synthetic fallback
        arr = self._det_rng(module, target_rank, output_dim, slot, "synth") \
                  .standard_normal((output_dim, target_rank)).astype(np.float16)
        return torch.from_numpy(arr * 0.02).contiguous()

    def get_weights(self, module: str, num_lora: int, target_rank: int, output_dim: int) -> torch.Tensor:
        key = (module, num_lora, target_rank, output_dim)
        if key not in self._w_cache:
            slices = [self._build_one(module, target_rank, output_dim, s) for s in range(num_lora)]
            self._w_cache[key] = torch.stack(slices).to(DEVICE, DTYPE).contiguous()
        return self._w_cache[key]

    def get_x(self, num_tokens: int, x_cols: int) -> torch.Tensor:
        key = (num_tokens, x_cols)
        if key not in self._x_cache:
            arr = self._det_rng("x", num_tokens, x_cols) \
                      .standard_normal((num_tokens, x_cols)).astype(np.float16) * 0.1
            self._x_cache[key] = torch.from_numpy(arr).to(DEVICE).contiguous()
        return self._x_cache[key]


# ─��───────────────────────────────────────────────────────────────────────────
# LoRABatchInfo builders
# ─────────────────────────────────────────────────────────────────────────────

def make_triton_bi(ranks: List[int], seq_len: int) -> LoRABatchInfo:
    bs = len(ranks)
    sl = torch.full((bs,), seq_len, dtype=torch.int32, device=DEVICE)
    sp = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
    sp[1:] = torch.cumsum(sl, 0)
    return LoRABatchInfo(
        bs=bs, num_segments=bs, max_len=max(seq_len, 16),
        use_cuda_graph=False, seg_lens=sl, seg_indptr=sp,
        weight_indices=torch.arange(bs, dtype=torch.int32, device=DEVICE),
        lora_ranks=torch.tensor(ranks, dtype=torch.int32, device=DEVICE),
        scalings=torch.ones(bs, dtype=torch.float32, device=DEVICE),
        permutation=None,
    )


def make_csgmv_bi(ranks: List[int], seq_len: int) -> LoRABatchInfo:
    bs  = len(ranks)
    tok = bs * seq_len
    sp  = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
    sp[1:] = torch.arange(1, bs + 1, dtype=torch.int32, device=DEVICE) * seq_len
    return LoRABatchInfo(
        bs=bs, num_segments=bs, max_len=max(seq_len, CSGMV_MIN_BLOCK_M),
        use_cuda_graph=False,
        seg_lens=torch.full((bs,), seq_len, dtype=torch.int32, device=DEVICE),
        seg_indptr=sp,
        weight_indices=torch.arange(bs, dtype=torch.int32, device=DEVICE),
        lora_ranks=torch.tensor(ranks, dtype=torch.int32, device=DEVICE),
        scalings=torch.ones(bs, dtype=torch.float32, device=DEVICE),
        permutation=torch.arange(tok, dtype=torch.int32, device=DEVICE),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Kernel wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _triton_module(x, weights, bi, output_dim):
    base = torch.zeros(x.shape[0], output_dim, dtype=DTYPE, device=DEVICE)
    return sgemm_lora_b_fwd(x=x, weights=weights, batch_info=bi, base_output=base)


def _csgmv_module(x, weights, bi, output_dim):
    base = torch.zeros(x.shape[0], output_dim, dtype=DTYPE, device=DEVICE)
    slc  = torch.tensor([0, output_dim], dtype=torch.int32, device=DEVICE)
    return chunked_sgmv_lora_expand_forward(
        x=x, weights=weights, batch_info=bi,
        slice_offsets=slc, max_slice_size=output_dim, base_output=base,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────

def time_fn(fn, warmup: int, repeat: int) -> Tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    lats = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return float(np.median(lats)), float(lats.mean()), float(lats.std())


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(ranks: List[int], seq_len: int) -> Dict[str, float]:
    bs        = len(ranks)
    max_rank  = max(ranks)
    min_rank  = min(ranks)
    sum_rank  = sum(ranks)
    mean_rank = sum_rank / bs
    var_rank  = float(np.var(ranks))
    std_rank  = float(np.std(ranks))
    rank_ratio = max_rank / min_rank if min_rank > 0 else float(max_rank)
    total_tok  = bs * seq_len

    return {
        "bs_x_max_rank":       float(bs * max_rank),
        "sum_rank":            float(sum_rank),
        "bs":                  float(bs),
        "max_rank":            float(max_rank),
        "min_rank":            float(min_rank),
        "mean_rank":           float(mean_rank),
        "total_tokens":        float(total_tok),
        "var_rank":            var_rank,
        "std_rank":            std_rank,
        "rank_ratio":          float(rank_ratio),
        "max_minus_min":       float(max_rank - min_rank),
        "max_x_std_rank":      float(max_rank * std_rank),
        "bs_x_sum_rank":       float(bs * sum_rank),
        "bs_x_mean_rank":      float(bs * mean_rank),
        "bs_x_min_rank":       float(bs * min_rank),
        "bs_x_var_rank":       float(bs * var_rank),
        "sum_x_max_rank":      float(sum_rank * max_rank),
        "bs_sq_x_max_rank":    float(bs ** 2 * max_rank),
        "bs_x_max_rank_sq":    float(bs * max_rank ** 2),
        "tok_x_max_rank":      float(total_tok * max_rank),
        "tok_x_sum_rank":      float(total_tok * sum_rank),
        "tok_x_mean_rank":     float(total_tok * mean_rank),
        "sqrt_bs_x_max_rank":  float(np.sqrt(bs * max_rank)),
        "sqrt_sum_rank":       float(np.sqrt(sum_rank)),
        "log1p_bs_x_max_rank": float(np.log1p(bs * max_rank)),
        "log1p_sum_rank":      float(np.log1p(sum_rank)),
        "sum_rank_sq":         float(sum_rank ** 2),
        "max_rank_sq":         float(max_rank ** 2),
        "bs_x_max_rank_sq_nl": float((bs * max_rank) ** 2),
    }


CANDIDATE_PREDICTORS = [
    "bs_x_max_rank", "sum_rank",
    "bs", "max_rank", "min_rank", "mean_rank", "total_tokens",
    "var_rank", "std_rank", "rank_ratio", "max_minus_min", "max_x_std_rank",
    "bs_x_sum_rank", "bs_x_mean_rank", "bs_x_min_rank", "bs_x_var_rank",
    "sum_x_max_rank", "bs_sq_x_max_rank", "bs_x_max_rank_sq",
    "tok_x_max_rank", "tok_x_sum_rank", "tok_x_mean_rank",
    "sqrt_bs_x_max_rank", "sqrt_sum_rank",
    "log1p_bs_x_max_rank", "log1p_sum_rank",
    "sum_rank_sq", "max_rank_sq", "bs_x_max_rank_sq_nl",
]


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_one(
    backend: str,
    ranks:   List[int],
    pool:    AdapterPool,
    seq_len: int,
    warmup:  int,
    repeat:  int,
) -> Dict:
    bs        = len(ranks)
    max_rank  = max(ranks)
    total_tok = bs * seq_len

    mod_data = {}
    for mod, (in_dim, out_dim) in MODULES.items():
        w = pool.get_weights(mod, bs, max_rank, out_dim)
        x = pool.get_x(total_tok, max_rank)
        mod_data[mod] = (w.contiguous(), x.contiguous(), out_dim)

    if backend == "triton":
        bi = make_triton_bi(ranks, seq_len)
        def fn():
            for mod, (w, x, out_dim) in mod_data.items():
                _triton_module(x, w, bi, out_dim)
    else:
        bi = make_csgmv_bi(ranks, seq_len)
        def fn():
            for mod, (w, x, out_dim) in mod_data.items():
                _csgmv_module(x, w, bi, out_dim)

    median_ms, mean_ms, std_ms = time_fn(fn, warmup=warmup, repeat=repeat)
    feats = compute_features(ranks, seq_len)

    return {
        "backend":      backend,
        "ranks_str":    str(ranks),
        "seq_len":      seq_len,
        "batch_size":   bs,
        "max_rank_int": max_rank,
        "sum_rank_int": sum(ranks),
        "median_ms":    median_ms,
        "mean_ms":      mean_ms,
        "std_ms":       std_ms,
        **feats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rank config builder
# ─────────────────────────────────────────────────────────────────────────────

def build_rank_configs(batch_sizes, rank_values, mode="all") -> List[List[int]]:
    configs = []
    if mode in ("homogeneous", "all"):
        for bs in batch_sizes:
            for r in rank_values:
                configs.append([r] * bs)
    if mode in ("heterogeneous", "all"):
        for bs in batch_sizes:
            if bs < 2:
                continue
            for r_lo, r_hi in itertools.combinations(rank_values, 2):
                half    = bs // 2
                quarter = max(1, bs // 4)
                configs.append([r_lo] * half    + [r_hi] * (bs - half))
                configs.append([r_lo] * quarter + [r_hi] * (bs - quarter))
                configs.append([r_lo] * (bs - quarter) + [r_hi] * quarter)
    seen, unique = set(), []
    for c in configs:
        k = tuple(sorted(c))
        if k not in seen:
            seen.add(k); unique.append(c)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# Full sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_all_configs(backend, rank_configs, pool, seq_len, warmup, repeat, verbose) -> List[Dict]:
    results = []
    total   = len(rank_configs)
    if verbose:
        print(f"\n  [{backend:6s}]  ALL modules: {' + '.join(MODULES.keys())}")
        print(f"  {total} configurations\n")
    for done, ranks in enumerate(rank_configs, 1):
        row = benchmark_one(backend, ranks, pool, seq_len, warmup, repeat)
        results.append(row)
        if verbose:
            bs = row["batch_size"]
            print(
                f"  {done:3d}/{total}  bs={bs:3d}  "
                f"ranks={row['ranks_str']:32s}  "
                f"max_r={row['max_rank_int']:3d}  sum_r={row['sum_rank_int']:5d}  "
                f"bs×max_r={bs * row['max_rank_int']:5d}  "
                f"lat={row['median_ms']:.4f}±{row['std_ms']:.4f} ms"
            )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# OLS helpers
# ─────────────────────────────────────────────────────────────────────────────

def ols_fit_1d(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float, float]:
    """Simple OLS: Y = α·X + β. Returns (α, β, R²)."""
    from scipy.stats import linregress
    sl, ic, r, *_ = linregress(X, Y)
    return float(sl), float(ic), float(r ** 2)


def ols_fit_nd(X_mat: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Multi-variate OLS via least squares: Y = X_mat @ coef.
    X_mat should include a column of ones for intercept.
    Returns (coef, R²).
    """
    coef, *_ = np.linalg.lstsq(X_mat, Y, rcond=None)
    Y_pred   = X_mat @ coef
    ss_res   = float(np.sum((Y - Y_pred) ** 2))
    ss_tot   = float(np.sum((Y - Y.mean()) ** 2))
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return coef, float(r2)


def rmse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((Y_true - Y_pred) ** 2)))


def mae(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(Y_true - Y_pred)))


# ─────────────────────────────────────────────────────────────────────────────
# Main fitting analysis
# ───────────────────────────────────────────��─────────────────────────────────

def fit_and_explore(
    results:  List[Dict],
    backend:  str,
    plot_dir: Optional[str] = None,
    top_k_multivar: int = 5,
) -> Tuple[Dict, List[Dict]]:
    """
    Complete fitting analysis for one backend.

    Steps:
      1. BGMV baseline:  latency ~ α·(bs×max_rank) + β
      2. MBGMV baseline: latency ~ α·Σrank_i + β
      3. Exhaustive single-predictor OLS (28 candidates)
      4. Top-K multi-variate OLS  (pair and triple combinations of best predictors)
      5. Residual diagnostics (RMSE, MAE, max error, normality)
      6. Optional: scatter + residual plots saved to plot_dir

    Returns:
      summary   : dict with key fit statistics
      fit_rows  : list of dicts for CSV export (one row per predictor)
    """
    try:
        from scipy.stats import shapiro
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    Y     = np.array([r["median_ms"] for r in results], dtype=float)
    n     = len(Y)
    title = backend.upper()
    Y_mean = Y.mean()
    Y_std  = Y.std()

    print(f"\n{'='*80}")
    print(f"  FIT ANALYSIS — {title}  (n={n}, all 4 modules per call)")
    print(f"  Y (latency_ms): mean={Y_mean:.4f}  std={Y_std:.4f}  "
          f"min={Y.min():.4f}  max={Y.max():.4f}")
    print(f"{'='*80}")

    # ── Helper: print one model's fit quality ──────────────────────────────
    def _print_fit(label: str, Y_pred: np.ndarray, formula: str):
        res  = Y - Y_pred
        r2   = 1 - np.sum(res**2) / np.sum((Y - Y_mean)**2)
        rms  = rmse(Y, Y_pred)
        ma   = mae(Y, Y_pred)
        mxe  = float(np.abs(res).max())
        pct  = float(np.abs(res / Y).mean()) * 100  # MAPE %
        print(f"\n  ┌─ {label}")
        print(f"  │  Formula : {formula}")
        print(f"  │  R²      = {r2:.6f}")
        print(f"  │  RMSE    = {rms:.6f} ms")
        print(f"  │  MAE     = {ma:.6f} ms")
        print(f"  │  MaxErr  = {mxe:.6f} ms")
        print(f"  │  MAPE    = {pct:.2f}%")
        if HAS_SCIPY and n >= 8:
            stat, p = shapiro(res)
            print(f"  │  Shapiro-Wilk (residuals): W={stat:.4f}  p={p:.4f}  "
                  f"{'normal ✓' if p > 0.05 else 'non-normal ✗'}")
        print(f"  └{'─'*60}")
        return float(r2), rms, ma, mxe

    # ── 1. BGMV baseline ──────────────────────────────────────────────────
    X_bgmv  = np.array([r["bs_x_max_rank"] for r in results])
    a_b, b_b, r2_b_raw = ols_fit_1d(X_bgmv, Y)
    Y_bgmv  = a_b * X_bgmv + b_b
    r2_b, rmse_b, mae_b, maxe_b = _print_fit(
        "BGMV baseline",
        Y_bgmv,
        f"latency = {a_b:+.8f}·(bs×max_rank) + {b_b:+.6f}",
    )

    # ── 2. MBGMV baseline ─────────────────────────────────────────────────
    X_mbgmv = np.array([r["sum_rank"] for r in results])
    a_m, b_m, r2_m_raw = ols_fit_1d(X_mbgmv, Y)
    Y_mbgmv = a_m * X_mbgmv + b_m
    r2_m, rmse_m, mae_m, maxe_m = _print_fit(
        "MBGMV baseline",
        Y_mbgmv,
        f"latency = {a_m:+.8f}·Σrank_i + {b_m:+.6f}",
    )

    # ── 3. Exhaustive single-predictor OLS ────────────────────────────────
    fit_rows: List[Dict] = []
    for pred in CANDIDATE_PREDICTORS:
        Xp = np.array([r.get(pred, np.nan) for r in results])
        if np.any(np.isnan(Xp)) or np.nanstd(Xp) < 1e-10:
            continue
        a_p, b_p, r2_p = ols_fit_1d(Xp, Y)
        Y_p = a_p * Xp + b_p
        fit_rows.append({
            "backend":    backend,
            "tag":        "single",
            "predictor":  pred,
            "formula":    f"{a_p:+.8f}·{pred} + {b_p:+.6f}",
            "alpha":      a_p,
            "beta":       b_p,
            "r2":         r2_p,
            "rmse_ms":    rmse(Y, Y_p),
            "mae_ms":     mae(Y, Y_p),
            "max_err_ms": float(np.abs(Y - Y_p).max()),
            "n_points":   n,
        })
    fit_rows.sort(key=lambda x: x["r2"], reverse=True)

    print(f"\n  ── Exhaustive single-predictor OLS (top 20) ──")
    print(f"  {'#':>3}  {'Predictor':<28}  {'α':>14}  {'β':>11}  "
          f"{'R²':>8}  {'RMSE':>8}  {'MAPE%':>6}  Notes")
    print(f"  {'─'*3}  {'─'*28}  {'─'*14}  {'─'*11}  "
          f"{'─'*8}  {'─'*8}  {'─'*6}  {'─'*12}")
    for i, fr in enumerate(fit_rows[:20]):
        Xp    = np.array([r.get(fr["predictor"], 0.0) for r in results])
        Y_p   = fr["alpha"] * Xp + fr["beta"]
        mape  = float(np.mean(np.abs((Y - Y_p) / Y))) * 100
        notes = []
        if fr["predictor"] == "bs_x_max_rank": notes.append("BGMV")
        if fr["predictor"] == "sum_rank":        notes.append("MBGMV")
        if i == 0:                               notes.append("★BEST")
        print(
            f"  {i+1:>3}  {fr['predictor']:<28}  "
            f"{fr['alpha']:>+14.8f}  {fr['beta']:>+11.6f}  "
            f"{fr['r2']:>8.6f}  {fr['rmse_ms']:>8.5f}  {mape:>6.2f}  "
            f"{', '.join(notes)}"
        )

    # ── 4. Multi-variate OLS (top-K pairs and triples) ────────────────────
    top_preds = [fr["predictor"] for fr in fit_rows[:top_k_multivar]]
    multivar_rows: List[Dict] = []

    print(f"\n  ── Multi-variate OLS (pairs from top-{top_k_multivar} single predictors) ──")
    print(f"  {'Predictors':<52}  {'R²':>8}  {'RMSE':>8}  Notes")
    print(f"  {'─'*52}  {'─'*8}  {'─'*8}  {'─'*12}")

    best_pair_r2   = -1.0
    best_pair_row  = None

    for p1, p2 in itertools.combinations(top_preds, 2):
        X1 = np.array([r.get(p1, 0.0) for r in results])
        X2 = np.array([r.get(p2, 0.0) for r in results])
        X_mat = np.column_stack([X1, X2, np.ones(n)])
        coef, r2_mv = ols_fit_nd(X_mat, Y)
        Y_p = X_mat @ coef
        rms = rmse(Y, Y_p)
        label = f"{p1} + {p2}"
        formula = (f"{coef[0]:+.6f}·{p1} + {coef[1]:+.6f}·{p2} "
                   f"+ {coef[2]:+.6f}")
        notes = "★ best pair" if r2_mv > best_pair_r2 else ""
        if r2_mv > best_pair_r2:
            best_pair_r2  = r2_mv
            best_pair_row = {
                "backend": backend, "tag": "pair",
                "predictor": label, "formula": formula,
                "alpha": coef[0], "beta": coef[2],
                "r2": r2_mv, "rmse_ms": rms,
                "mae_ms": mae(Y, Y_p),
                "max_err_ms": float(np.abs(Y - Y_p).max()),
                "n_points": n,
            }
        print(f"  {label:<52}  {r2_mv:>8.6f}  {rms:>8.5f}  {notes}")
        multivar_rows.append({
            "backend": backend, "tag": "pair",
            "predictor": label, "formula": formula,
            "alpha": coef[0], "beta": coef[2],
            "r2": r2_mv, "rmse_ms": rms,
            "mae_ms": mae(Y, Y_p),
            "max_err_ms": float(np.abs(Y - Y_p).max()),
            "n_points": n,
        })

    # triples
    print(f"\n  ── Multi-variate OLS (triples from top-{top_k_multivar}) ──")
    print(f"  {'Predictors':<60}  {'R²':>8}  {'RMSE':>8}")
    print(f"  {'─'*60}  {'─'*8}  {'─'*8}")
    best_triple_r2  = -1.0
    best_triple_row = None

    for p1, p2, p3 in itertools.combinations(top_preds, 3):
        X1 = np.array([r.get(p1, 0.0) for r in results])
        X2 = np.array([r.get(p2, 0.0) for r in results])
        X3 = np.array([r.get(p3, 0.0) for r in results])
        X_mat = np.column_stack([X1, X2, X3, np.ones(n)])
        coef, r2_mv = ols_fit_nd(X_mat, Y)
        Y_p = X_mat @ coef
        rms = rmse(Y, Y_p)
        label   = f"{p1} + {p2} + {p3}"
        formula = (f"{coef[0]:+.6f}·{p1} + {coef[1]:+.6f}·{p2} + "
                   f"{coef[2]:+.6f}·{p3} + {coef[3]:+.6f}")
        notes = "★ best triple" if r2_mv > best_triple_r2 else ""
        if r2_mv > best_triple_r2:
            best_triple_r2  = r2_mv
            best_triple_row = {
                "backend": backend, "tag": "triple",
                "predictor": label, "formula": formula,
                "alpha": coef[0], "beta": coef[3],
                "r2": r2_mv, "rmse_ms": rms,
                "mae_ms": mae(Y, Y_p),
                "max_err_ms": float(np.abs(Y - Y_p).max()),
                "n_points": n,
            }
        print(f"  {label:<60}  {r2_mv:>8.6f}  {rms:>8.5f}  {notes}")
        multivar_rows.append({
            "backend": backend, "tag": "triple",
            "predictor": label, "formula": formula,
            "alpha": coef[0], "beta": coef[3],
            "r2": r2_mv, "rmse_ms": rms,
            "mae_ms": mae(Y, Y_p),
            "max_err_ms": float(np.abs(Y - Y_p).max()),
            "n_points": n,
        })

    # ── 5. Summary ────────────────────────────────────────────────────────
    best_single = fit_rows[0]
    print(f"\n  {'='*78}")
    print(f"  FINAL SUMMARY — {title}")
    print(f"  {'─'*78}")
    print(f"  BGMV    (bs×max_rank)  R²={r2_b:.6f}  RMSE={rmse_b:.5f} ms  "
          f"α={a_b:+.8f}  β={b_b:+.6f}")
    print(f"  MBGMV   (Σrank_i)      R²={r2_m:.6f}  RMSE={rmse_m:.5f} ms  "
          f"α={a_m:+.8f}  β={b_m:+.6f}")
    print(f"  ★ Best single  [{best_single['predictor']}]  "
          f"R²={best_single['r2']:.6f}  RMSE={best_single['rmse_ms']:.5f} ms")
    print(f"    {best_single['formula']}")
    if best_pair_row:
        print(f"  ★ Best pair    [{best_pair_row['predictor']}]  "
              f"R²={best_pair_r2:.6f}  RMSE={best_pair_row['rmse_ms']:.5f} ms")
        print(f"    {best_pair_row['formula']}")
    if best_triple_row:
        print(f"  ★ Best triple  [{best_triple_row['predictor']}]  "
              f"R²={best_triple_r2:.6f}  RMSE={best_triple_row['rmse_ms']:.5f} ms")
        print(f"    {best_triple_row['formula']}")

    # BGMV vs MBGMV vs best interpretation
    print(f"\n  Interpretation:")
    if r2_b > r2_m + 0.02:
        print(f"  → BGMV model (bs×max_rank) fits {title} better than MBGMV: "
              f"ΔR²={r2_b-r2_m:+.4f}")
        print(f"     This matches the BGMV padding-to-max-rank bandwidth hypothesis.")
    elif r2_m > r2_b + 0.02:
        print(f"  → MBGMV model (Σrank_i) fits {title} better than BGMV: "
              f"ΔR²={r2_m-r2_b:+.4f}")
        print(f"     This matches the MBGMV per-adapter actual-rank bandwidth hypothesis.")
    else:
        print(f"  → BGMV and MBGMV models are comparable for {title} "
              f"(ΔR²={abs(r2_b-r2_m):.4f} < 0.02)")
        print(f"     Both predictors explain similar variance in decode latency.")

    # ── 6. Optional plots ─────────────────────────────────────────────────
    if plot_dir is not None:
        _save_plots(results, Y, backend, fit_rows, a_b, b_b, a_m, b_m, plot_dir)

    all_rows = fit_rows + multivar_rows
    summary = {
        "backend":  backend,
        "n":        n,
        "bgmv":     {"alpha": a_b,  "beta": b_b,  "r2": r2_b,  "rmse": rmse_b},
        "mbgmv":    {"alpha": a_m,  "beta": b_m,  "r2": r2_m,  "rmse": rmse_m},
        "best_single":  best_single,
        "best_pair":    best_pair_row,
        "best_triple":  best_triple_row,
    }
    return summary, all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Plotting (optional, requires matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def _save_plots(
    results:  List[Dict],
    Y:        np.ndarray,
    backend:  str,
    fit_rows: List[Dict],
    a_b:      float,
    b_b:      float,
    a_m:      float,
    b_m:      float,
    plot_dir: str,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available — skipping plots.")
        return

    os.makedirs(plot_dir, exist_ok=True)

    X_bgmv  = np.array([r["bs_x_max_rank"] for r in results])
    X_mbgmv = np.array([r["sum_rank"]       for r in results])
    best    = fit_rows[0]
    X_best  = np.array([r.get(best["predictor"], 0.0) for r in results])

    # ── Plot 1: BGMV — predicted vs actual ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{backend.upper()} — BGMV baseline: latency ~ α·(bs×max_rank) + β",
                 fontsize=12, fontweight="bold")

    Y_pred_bgmv = a_b * X_bgmv + b_b
    ax = axes[0]
    ax.scatter(X_bgmv, Y, alpha=0.6, s=20, label="Measured", color="steelblue")
    xs = np.linspace(X_bgmv.min(), X_bgmv.max(), 200)
    ax.plot(xs, a_b * xs + b_b, "r-", lw=2, label=f"Fit (R²={best['r2']:.4f})")
    ax.set_xlabel("bs × max_rank"); ax.set_ylabel("Latency (ms)")
    ax.set_title("Predicted vs Measured"); ax.legend()

    res_bgmv = Y - Y_pred_bgmv
    ax = axes[1]
    ax.hist(res_bgmv, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Residual (ms)"); ax.set_ylabel("Count")
    ax.set_title(f"Residuals  RMSE={float(np.sqrt(np.mean(res_bgmv**2))):.5f} ms")
    plt.tight_layout()
    path = os.path.join(plot_dir, f"{backend}_bgmv_fit.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  [PLOT] {path}")

    # ── Plot 2: MBGMV — predicted vs actual ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{backend.upper()} — MBGMV baseline: latency ~ α·Σrank_i + β",
                 fontsize=12, fontweight="bold")

    Y_pred_mbgmv = a_m * X_mbgmv + b_m
    ax = axes[0]
    ax.scatter(X_mbgmv, Y, alpha=0.6, s=20, label="Measured", color="darkorange")
    xs = np.linspace(X_mbgmv.min(), X_mbgmv.max(), 200)
    ax.plot(xs, a_m * xs + b_m, "r-", lw=2)
    ax.set_xlabel("Σ rank_i"); ax.set_ylabel("Latency (ms)")
    ax.set_title("Predicted vs Measured"); ax.legend()

    res_mbgmv = Y - Y_pred_mbgmv
    ax = axes[1]
    ax.hist(res_mbgmv, bins=30, color="darkorange", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Residual (ms)"); ax.set_ylabel("Count")
    ax.set_title(f"Residuals  RMSE={float(np.sqrt(np.mean(res_mbgmv**2))):.5f} ms")
    plt.tight_layout()
    path = os.path.join(plot_dir, f"{backend}_mbgmv_fit.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  [PLOT] {path}")

    # ── Plot 3: Best single predictor ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{backend.upper()} — Best predictor: {best['predictor']}\n"
        f"R²={best['r2']:.6f}  formula: {best['formula']}",
        fontsize=10, fontweight="bold",
    )
    Y_pred_best = best["alpha"] * X_best + best["beta"]
    ax = axes[0]
    ax.scatter(X_best, Y, alpha=0.6, s=20, color="seagreen", label="Measured")
    xs = np.linspace(X_best.min(), X_best.max(), 200)
    ax.plot(xs, best["alpha"] * xs + best["beta"], "r-", lw=2, label="Fit")
    ax.set_xlabel(best["predictor"]); ax.set_ylabel("Latency (ms)")
    ax.set_title("Predicted vs Measured"); ax.legend()

    res_best = Y - Y_pred_best
    ax = axes[1]
    ax.hist(res_best, bins=30, color="seagreen", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Residual (ms)"); ax.set_ylabel("Count")
    ax.set_title(f"Residuals  RMSE={float(np.sqrt(np.mean(res_best**2))):.5f} ms")
    plt.tight_layout()
    path = os.path.join(plot_dir, f"{backend}_best_fit.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  [PLOT] {path}")

    # ── Plot 4: R² bar chart for top-15 predictors ───────────────────────
    top15 = fit_rows[:15]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d62728" if fr["predictor"] in ("bs_x_max_rank", "sum_rank")
              else "#1f77b4" for fr in top15]
    ax.barh([fr["predictor"] for fr in reversed(top15)],
            [fr["r2"] for fr in reversed(top15)],
            color=list(reversed(colors)), edgecolor="white")
    ax.axvline(0.96, color="green", ls="--", lw=1.5, label="R²=0.96 (paper target)")
    ax.set_xlabel("R²"); ax.set_title(f"{backend.upper()} — Top-15 predictors R²")
    ax.legend(); plt.tight_layout()
    path = os.path.join(plot_dir, f"{backend}_r2_bar.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  [PLOT] {path}")

    # ── Plot 5: BGMV vs MBGMV side-by-side scatter ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{backend.upper()} — BGMV vs MBGMV predictor comparison",
                 fontsize=12, fontweight="bold")
    for ax, Xv, label, color, a, b in [
        (axes[0], X_bgmv,  "bs×max_rank (BGMV)",  "steelblue",  a_b, b_b),
        (axes[1], X_mbgmv, "Σrank_i (MBGMV)",     "darkorange", a_m, b_m),
    ]:
        ax.scatter(Xv, Y, alpha=0.5, s=15, color=color)
        xs = np.linspace(Xv.min(), Xv.max(), 200)
        ax.plot(xs, a * xs + b, "r-", lw=2)
        r2v = 1 - np.sum((Y - (a * Xv + b))**2) / np.sum((Y - Y.mean())**2)
        ax.set_xlabel(label); ax.set_ylabel("Latency (ms)")
        ax.set_title(f"R²={r2v:.6f}")
    plt.tight_layout()
    path = os.path.join(plot_dir, f"{backend}_bgmv_vs_mbgmv.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  [PLOT] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-backend comparison
# ─────────────────────────────────────────────────────────────────────────────

def cross_backend_compare(triton_res: List[Dict], csgmv_res: List[Dict]):
    print(f"\n{'='*80}")
    print("  CROSS-BACKEND R² COMPARISON")
    print(f"{'='*80}")
    preds = [
        ("bs_x_max_rank",   "BGMV: bs×max_rank (paper)"),
        ("sum_rank",         "MBGMV: Σrank_i (paper)"),
        ("bs_x_sum_rank",    "bs × Σrank"),
        ("sum_x_max_rank",   "Σrank × max_rank"),
        ("tok_x_max_rank",   "total_tokens × max_rank"),
        ("tok_x_sum_rank",   "total_tokens × Σrank"),
        ("sum_rank_sq",      "Σrank²"),
        ("bs_sq_x_max_rank", "bs² × max_rank"),
    ]
    print(f"\n  {'Predictor':<26}  {'Description':<28}  "
          f"{'Triton R²':>10}  {'CSGMV R²':>10}  {'Better':>7}")
    print(f"  {'─'*26}  {'─'*28}  {'─'*10}  {'─'*10}  {'─'*7}")
    for pred, desc in preds:
        r2_t = r2_c = float("nan")
        if triton_res:
            Y = np.array([r["median_ms"] for r in triton_res])
            X = np.array([r.get(pred, np.nan) for r in triton_res])
            if np.nanstd(X) > 1e-10: _, _, r2_t = ols_fit_1d(X, Y)
        if csgmv_res:
            Y = np.array([r["median_ms"] for r in csgmv_res])
            X = np.array([r.get(pred, np.nan) for r in csgmv_res])
            if np.nanstd(X) > 1e-10: _, _, r2_c = ols_fit_1d(X, Y)
        win = "─"
        if not (np.isnan(r2_t) or np.isnan(r2_c)):
            if r2_t > r2_c + 0.005:   win = "Triton"
            elif r2_c > r2_t + 0.005: win = "CSGMV"
        print(f"  {pred:<26}  {desc:<28}  "
              f"{r2_t:>10.6f}  {r2_c:>10.6f}  {win:>7}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_measurement_csv(results: List[Dict], path: str):
    if not results: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"\n  [CSV] Measurements → {path}  ({len(results)} rows)")


def save_fit_csv(fit_rows: List[Dict], path: str):
    if not fit_rows: return
    # Collect all keys across rows (multivar rows may have extra fields)
    all_keys = []
    seen_keys = set()
    for row in fit_rows:
        for k in row:
            if k not in seen_keys:
                all_keys.append(k); seen_keys.add(k)
    sorted_ = sorted(fit_rows, key=lambda x: (x["backend"], x.get("tag",""), -x["r2"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for row in sorted_:
            w.writerow({k: row.get(k, "") for k in all_keys})
    print(f"  [CSV] Fit results  → {path}  ({len(sorted_)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LoRA Kernel Bench — Triton (BGMV) vs CSGMV (MBGMV)\n"
                    "Real adapters · All modules · Total latency · Comprehensive fitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend",        default="both",
                        choices=["triton", "csgmv", "both"])
    parser.add_argument("--batch-sizes",    type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--rank-values",    type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--seq-len",        type=int, default=1)
    parser.add_argument("--mode",           default="all",
                        choices=["homogeneous", "heterogeneous", "all"])
    parser.add_argument("--warmup",         type=int, default=20)
    parser.add_argument("--repeat",         type=int, default=200)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--fit",            action="store_true")
    parser.add_argument("--top-k-multivar", type=int, default=5,
                        help="Top-K single predictors to use in multivariate search")
    parser.add_argument("--save-csv",       default=None)
    parser.add_argument("--save-fit-csv",   default=None)
    parser.add_argument("--plot-dir",       default=None,
                        help="Directory to save matplotlib plots (requires matplotlib)")
    parser.add_argument("--quiet",          action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required."

    # ── Header ────────────────────────────────────────────────────────────
    print("=" * 80)
    print("  LoRA Kernel Microbenchmark — Triton (BGMV) vs CSGMV (MBGMV)")
    print("  Real adapters · All 4 modules · Total latency · Comprehensive fitting")
    print("=" * 80)
    print(f"  GPU             : {torch.cuda.get_device_name(0)}")
    print(f"  Backend(s)      : {args.backend}")
    print(f"  Batch sizes     : {args.batch_sizes}")
    print(f"  Rank values     : {args.rank_values}")
    print(f"  Seq len         : {args.seq_len} "
          f"({'decode' if args.seq_len == 1 else 'prefill'})")
    print(f"  Mode            : {args.mode}")
    print(f"  Warmup / repeat : {args.warmup} / {args.repeat}")
    print(f"  Seed            : {args.seed}")
    print(f"  Top-K multivar  : {args.top_k_multivar}")
    if args.plot_dir:
        print(f"  Plot dir        : {args.plot_dir}")
    print(f"\n  Modules timed per call (total latency = sum of all 4):")
    for m, (i, o) in MODULES.items():
        print(f"    {m:15s}  in={i:6d}  out={o:6d}")
    print(f"\n  Predictor space: {len(CANDIDATE_PREDICTORS)} candidates "
          f"(bs + ranks only, NO output_dim)")

    # ── Build adapter pool ────────────────────────────────────────────────
    pool = AdapterPool(ADAPTER_REGISTRY, seed=args.seed)

    # ── Build rank configurations ─────────────────────────────────────────
    rank_configs = build_rank_configs(args.batch_sizes, args.rank_values, args.mode)
    print(f"\n  Rank configurations : {len(rank_configs)}")
    print(f"  Data points/backend : {len(rank_configs)}")

    triton_res:   List[Dict] = []
    csgmv_res:    List[Dict] = []
    all_fit_rows: List[Dict] = []

    # ── Triton backend ────────────────────────────────────────────────────
    if args.backend in ("triton", "both"):
        print(f"\n{'─'*80}")
        print("  TRITON  ─  sgemm_lora_b_fwd")
        print("  BGMV semantics: all adapters padded to max_rank in the batch")
        print("  Paper hypothesis: decode latency ∝ bs × max_rank")
        print(f"{'─'*80}")
        triton_res = run_all_configs(
            backend      = "triton",
            rank_configs = rank_configs,
            pool         = pool,
            seq_len      = args.seq_len,
            warmup       = args.warmup,
            repeat       = args.repeat,
            verbose      = not args.quiet,
        )
        if args.fit:
            _, rows = fit_and_explore(
                results         = triton_res,
                backend         = "triton",
                plot_dir        = args.plot_dir,
                top_k_multivar  = args.top_k_multivar,
            )
            all_fit_rows.extend(rows)

    # ── CSGMV backend ─────────────────────────────────────────────────────
    if args.backend in ("csgmv", "both"):
        print(f"\n{'─'*80}")
        print("  CSGMV  ─  chunked_sgmv_lora_expand_forward")
        print("  MBGMV semantics: each adapter uses its actual rank (no padding)")
        print("  Paper hypothesis: decode latency ∝ Σ rank_i")
        print(f"  BLOCK_M >= {CSGMV_MIN_BLOCK_M}  "
              f"BLOCK_K = {CSGMV_BLOCK_K}  BLOCK_N = {CSGMV_BLOCK_N}")
        print(f"{'─'*80}")
        csgmv_res = run_all_configs(
            backend      = "csgmv",
            rank_configs = rank_configs,
            pool         = pool,
            seq_len      = args.seq_len,
            warmup       = args.warmup,
            repeat       = args.repeat,
            verbose      = not args.quiet,
        )
        if args.fit:
            _, rows = fit_and_explore(
                results         = csgmv_res,
                backend         = "csgmv",
                plot_dir        = args.plot_dir,
                top_k_multivar  = args.top_k_multivar,
            )
            all_fit_rows.extend(rows)

    # ── Cross-backend comparison ──────────────────────────────────────────
    if args.fit and triton_res and csgmv_res:
        cross_backend_compare(triton_res, csgmv_res)

    # ── Save measurement CSV ──────────────────────────────────────────────
    all_meas = triton_res + csgmv_res
    if args.save_csv and all_meas:
        save_measurement_csv(all_meas, args.save_csv)

    # ── Save fit CSV ──────────────────────────────────────────────────────
    fit_path = args.save_fit_csv
    if args.fit and all_fit_rows:
        # Auto-derive fit CSV path from measurement CSV path if not specified
        if not fit_path and args.save_csv:
            base = args.save_csv
            fit_path = (
                base.rsplit(".", 1)[0] + "_fit.csv"
                if "." in base
                else base + "_fit.csv"
            )
        if fit_path:
            save_fit_csv(all_fit_rows, fit_path)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Done.")
    print(f"  Measurements : {len(all_meas)} rows "
          f"({len(triton_res)} triton + {len(csgmv_res)} csgmv)")
    if all_fit_rows:
        single_rows = [r for r in all_fit_rows if r.get("tag") == "single"]
        pair_rows   = [r for r in all_fit_rows if r.get("tag") == "pair"]
        triple_rows = [r for r in all_fit_rows if r.get("tag") == "triple"]
        print(f"  Fit rows     : {len(all_fit_rows)} total  "
              f"({len(single_rows)} single + {len(pair_rows)} pair + "
              f"{len(triple_rows)} triple)")
    if args.save_csv:
        print(f"  Measurements → {args.save_csv}")
    if fit_path:
        print(f"  Fit results  → {fit_path}")
    if args.plot_dir:
        print(f"  Plots        → {args.plot_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()