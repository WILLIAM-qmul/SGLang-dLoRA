"""
bench_kernel_perf.py
=====================
Kernel-level microbenchmark for the TOPPINGS paper's performance model study.

Tests two LoRA kernel types in the SGLang/dLoRA codebase:
  - "triton" (sgemm_lora_b_fwd): BGMV-style, pads low-rank adapters to max rank.
  - "csgmv" (chunked_sgmv_lora_expand_forward): MBGMV-style, no padding.

Both kernels are first evaluated with the unified baseline formula:
    latency ~ α·(bs × max_rank) + β

Then an exhaustive search over candidate predictors finds the best-fit
linear formula for each kernel automatically.

Usage:
    python bench_kernel_perf.py --backend both --fit --save-csv results.csv
"""

import argparse
import itertools
from typing import List, Dict, Tuple

import torch
import numpy as np

from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.lora.triton_ops import sgemm_lora_b_fwd
from sglang.srt.lora.triton_ops import chunked_sgmv_lora_expand_forward

DEVICE = "cuda"
DTYPE = torch.float16

# chunked_sgmv_expand kernel: BLOCK_M = batch_info.max_len, must be >= 16
CSGMV_MIN_BLOCK_M = 16


# -----------------------------------------------------------------------
# Batch info builders
# -----------------------------------------------------------------------

def make_triton_batch_info(
    ranks: List[int],
    seq_len: int = 1,
    max_loras: int = None,
) -> LoRABatchInfo:
    bs = len(ranks)
    max_loras = max_loras or bs
    seg_lens = torch.ones(bs, dtype=torch.int32, device=DEVICE) * seq_len
    seg_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
    seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
    weight_indices = torch.arange(bs, dtype=torch.int32, device=DEVICE)
    lora_ranks_t = torch.zeros(max_loras, dtype=torch.int32, device=DEVICE)
    scalings_t = torch.ones(max_loras, dtype=torch.float32, device=DEVICE)
    for i, r in enumerate(ranks):
        lora_ranks_t[i] = r
    return LoRABatchInfo(
        bs=bs, num_segments=bs, max_len=seq_len, use_cuda_graph=False,
        seg_lens=seg_lens, seg_indptr=seg_indptr, weight_indices=weight_indices,
        lora_ranks=lora_ranks_t, scalings=scalings_t, permutation=None,
    )


def make_csgmv_batch_info(
    ranks: List[int],
    seq_len: int = 1,
    max_loras: int = None,
) -> LoRABatchInfo:
    bs = len(ranks)
    max_loras = max_loras or bs
    total_tokens = bs * seq_len
    permutation = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)
    seg_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
    seg_indptr[1:] = torch.arange(1, bs + 1, dtype=torch.int32, device=DEVICE) * seq_len
    weight_indices = torch.arange(bs, dtype=torch.int32, device=DEVICE)
    lora_ranks_t = torch.zeros(max_loras, dtype=torch.int32, device=DEVICE)
    scalings_t = torch.ones(max_loras, dtype=torch.float32, device=DEVICE)
    for i, r in enumerate(ranks):
        lora_ranks_t[i] = r
    block_m = max(seq_len, CSGMV_MIN_BLOCK_M)
    return LoRABatchInfo(
        bs=bs, num_segments=bs, max_len=block_m, use_cuda_graph=False,
        seg_lens=torch.ones(bs, dtype=torch.int32, device=DEVICE) * seq_len,
        seg_indptr=seg_indptr, weight_indices=weight_indices,
        lora_ranks=lora_ranks_t, scalings=scalings_t, permutation=permutation,
    )


# -----------------------------------------------------------------------
# Tensor builders
# -----------------------------------------------------------------------

def make_tensors_for_triton(ranks: List[int], output_dim: int, seq_len: int = 1):
    bs, max_rank = len(ranks), max(ranks)
    x = torch.randn(bs * seq_len, max_rank, dtype=DTYPE, device=DEVICE)
    weights = torch.randn(bs, output_dim, max_rank, dtype=DTYPE, device=DEVICE)
    return x, weights


def make_tensors_for_csgmv(ranks: List[int], output_dim: int, seq_len: int = 1):
    bs, max_rank = len(ranks), max(ranks)
    x = torch.randn(bs * seq_len, max_rank, dtype=DTYPE, device=DEVICE)
    weights = torch.randn(bs, output_dim, max_rank, dtype=DTYPE, device=DEVICE)
    slice_offsets = torch.tensor([0, output_dim], dtype=torch.int32, device=DEVICE)
    return x, weights, slice_offsets


# -----------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------

def warmup_and_time(fn, warmup: int = 10, repeat: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    return float(np.median([s.elapsed_time(e) for s, e in zip(starts, ends)]))


# -----------------------------------------------------------------------
# Feature engineering: compute ALL candidate predictors for one config
# -----------------------------------------------------------------------

def compute_features(ranks: List[int], seq_len: int) -> Dict[str, float]:
    """
    Compute all candidate scalar predictors for one (ranks, seq_len) config.
    These cover the TOPPINGS paper hypotheses and plausible alternatives.
    """
    bs = len(ranks)
    max_rank = max(ranks)
    min_rank = min(ranks)
    sum_rank = sum(ranks)
    mean_rank = sum_rank / bs
    # variance / std of ranks in batch (heterogeneity signal)
    var_rank = float(np.var(ranks))
    std_rank = float(np.std(ranks))
    # ratio of max to min (another heterogeneity signal)
    rank_ratio = max_rank / min_rank if min_rank > 0 else max_rank

    total_tokens = bs * seq_len

    return {
        # ---- Paper baseline (used for BOTH kernels) ----
        "bs_x_max_rank":        bs * max_rank,          # BGMV hypothesis

        # ---- TOPPINGS MBGMV hypothesis ----
        "sum_rank":             sum_rank,                # MBGMV hypothesis

        # ---- Alternative single predictors ----
        "bs":                   bs,
        "max_rank":             max_rank,
        "min_rank":             min_rank,
        "mean_rank":            mean_rank,
        "total_tokens":         total_tokens,
        "total_tokens_x_max_rank": total_tokens * max_rank,
        "total_tokens_x_sum_rank": total_tokens * sum_rank,

        # ---- Heterogeneity features ----
        "var_rank":             var_rank,
        "std_rank":             std_rank,
        "rank_ratio":           rank_ratio,
        "max_rank_x_std_rank":  max_rank * std_rank,

        # ---- Two-variable products ----
        "bs_x_sum_rank":        bs * sum_rank,
        "bs_x_mean_rank":       bs * mean_rank,
        "bs_x_min_rank":        bs * min_rank,
        "bs_x_var_rank":        bs * var_rank,
        "sum_rank_x_max_rank":  sum_rank * max_rank,

        # ---- Nonlinear transforms of primary predictors ----
        "sqrt_bs_x_max_rank":   np.sqrt(bs * max_rank),
        "log_bs_x_max_rank":    np.log1p(bs * max_rank),
        "sqrt_sum_rank":        np.sqrt(sum_rank),
        "log_sum_rank":         np.log1p(sum_rank),
        "(bs_x_max_rank)^2":    (bs * max_rank) ** 2,
        "sum_rank^2":           sum_rank ** 2,
    }


# -----------------------------------------------------------------------
# Benchmark runners
# -----------------------------------------------------------------------

def _run_configs(
    rank_configs: List[List[int]],
    backend: str,
    output_dim: int,
    seq_len: int,
    warmup: int,
    repeat: int,
) -> List[Dict]:
    results = []
    for ranks in rank_configs:
        bs = len(ranks)

        if backend == "triton":
            batch_info = make_triton_batch_info(ranks, seq_len=seq_len)
            x, weights = make_tensors_for_triton(ranks, output_dim, seq_len)
            base_out = torch.zeros(bs * seq_len, output_dim, dtype=DTYPE, device=DEVICE)
            def fn():
                return sgemm_lora_b_fwd(x=x, weights=weights,
                                        batch_info=batch_info,
                                        base_output=base_out.clone())
        else:  # csgmv
            batch_info = make_csgmv_batch_info(ranks, seq_len=seq_len)
            x, weights, slc = make_tensors_for_csgmv(ranks, output_dim, seq_len)
            base_out = torch.zeros(bs * seq_len, output_dim, dtype=DTYPE, device=DEVICE)
            def fn():
                return chunked_sgmv_lora_expand_forward(
                    x=x, weights=weights, batch_info=batch_info,
                    slice_offsets=slc, max_slice_size=output_dim,
                    base_output=base_out.clone())

        lat = warmup_and_time(fn, warmup=warmup, repeat=repeat)
        feats = compute_features(ranks, seq_len)
        row = {
            "backend":    backend,
            "batch_size": bs,
            "ranks":      ranks,
            "latency_ms": lat,
            **feats,
        }
        results.append(row)

        # Console line: always show the baseline predictor
        print(
            f"[{backend:6s}] bs={bs:3d} "
            f"max_r={feats['max_rank']:3d} "
            f"sum_r={feats['sum_rank']:5d} "
            f"bs×max_r={feats['bs_x_max_rank']:5.0f} | "
            f"lat={lat:.4f} ms"
        )
    return results


def bench_triton(rank_configs, output_dim, seq_len, warmup, repeat):
    print("\n[TRITON / BGMV-style kernel]")
    print("  Baseline formula: latency ~ α·(bs × max_rank) + β")
    print("-" * 70)
    return _run_configs(rank_configs, "triton", output_dim, seq_len, warmup, repeat)


def bench_csgmv(rank_configs, output_dim, seq_len, warmup, repeat):
    print("\n[CSGMV / MBGMV-style kernel]")
    print("  Baseline formula: latency ~ α·(bs × max_rank) + β")
    print(f"  (BLOCK_M clamped to >= {CSGMV_MIN_BLOCK_M} for tl.dot)")
    print("-" * 70)
    return _run_configs(rank_configs, "csgmv", output_dim, seq_len, warmup, repeat)


# -----------------------------------------------------------------------
# Fitting: baseline + exhaustive single-predictor search
# -----------------------------------------------------------------------

CANDIDATE_PREDICTORS = [
    "bs_x_max_rank",
    "sum_rank",
    "bs",
    "max_rank",
    "min_rank",
    "mean_rank",
    "total_tokens",
    "total_tokens_x_max_rank",
    "total_tokens_x_sum_rank",
    "var_rank",
    "std_rank",
    "rank_ratio",
    "max_rank_x_std_rank",
    "bs_x_sum_rank",
    "bs_x_mean_rank",
    "bs_x_min_rank",
    "bs_x_var_rank",
    "sum_rank_x_max_rank",
    "sqrt_bs_x_max_rank",
    "log_bs_x_max_rank",
    "sqrt_sum_rank",
    "log_sum_rank",
    "(bs_x_max_rank)^2",
    "sum_rank^2",
]


def fit_one(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float, float]:
    """OLS: Y = alpha*X + beta.  Returns (alpha, beta, R²)."""
    from scipy.stats import linregress
    slope, intercept, r, _p, _se = linregress(X, Y)
    return slope, intercept, r ** 2


def fit_and_explore(results: List[Dict], backend: str):
    """
    1. Print the baseline fit: latency ~ α·(bs × max_rank) + β
    2. Exhaustively search all CANDIDATE_PREDICTORS for the best R².
    3. Print a ranked table.
    """
    Y = np.array([r["latency_ms"] for r in results], dtype=float)

    print(f"\n{'='*70}")
    print(f"  FIT ANALYSIS — {backend.upper()} backend")
    print(f"{'='*70}")

    # ---- 1. Mandatory baseline ----
    X_base = np.array([r["bs_x_max_rank"] for r in results], dtype=float)
    a, b, r2 = fit_one(X_base, Y)
    print(f"\n[BASELINE] latency ~ α·(bs × max_rank) + β")
    print(f"           α={a:.6f}  β={b:.4f}  R²={r2:.4f}")

    # ---- 2. Exhaustive search ----
    fit_results = []
    for pred in CANDIDATE_PREDICTORS:
        X = np.array([r[pred] for r in results], dtype=float)
        # skip degenerate columns (zero variance)
        if np.std(X) < 1e-12:
            continue
        a_p, b_p, r2_p = fit_one(X, Y)
        fit_results.append((pred, a_p, b_p, r2_p))

    # sort by R² descending
    fit_results.sort(key=lambda x: x[3], reverse=True)

    print(f"\n[EXHAUSTIVE SEARCH] All single-predictor linear fits, ranked by R²:")
    print(f"  {'Predictor':<30s}  {'α':>12s}  {'β':>10s}  {'R²':>8s}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*10}  {'-'*8}")
    for pred, a_p, b_p, r2_p in fit_results:
        marker = " ◀ BEST" if pred == fit_results[0][0] else ""
        baseline_mark = " ◀ BASELINE" if pred == "bs_x_max_rank" else ""
        print(
            f"  {pred:<30s}  {a_p:>12.6f}  {b_p:>10.4f}  {r2_p:>8.4f}"
            f"{marker}{baseline_mark}"
        )

    # ---- 3. Best vs baseline summary ----
    best_pred, best_a, best_b, best_r2 = fit_results[0]
    print(f"\n[SUMMARY for {backend.upper()}]")
    print(f"  Baseline  (bs × max_rank) :  R² = {r2:.4f}")
    print(f"  Best fit  ({best_pred}) :  R² = {best_r2:.4f}  "
          f"(Δ = {best_r2 - r2:+.4f})")
    if best_pred != "bs_x_max_rank":
        print(f"  → Best formula: latency ~ {best_a:.6f}·{best_pred} + {best_b:.4f}")
    else:
        print(f"  → Baseline IS the best formula.")

    return fit_results


# -----------------------------------------------------------------------
# Rank-config builder
# -----------------------------------------------------------------------

def build_rank_configs(
    batch_sizes: List[int],
    rank_values: List[int],
    mode: str = "all",
) -> List[List[int]]:
    configs = []
    if mode in ("homogeneous", "all"):
        for bs in batch_sizes:
            for r in rank_values:
                configs.append([r] * bs)
    if mode in ("heterogeneous", "all"):
        for bs in batch_sizes:
            if bs < 2:
                continue
            for r_min, r_max in itertools.combinations(rank_values, 2):
                half = bs // 2
                configs.append([r_min] * half + [r_max] * (bs - half))
    seen, unique = set(), []
    for c in configs:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRA Kernel Performance Benchmark")
    parser.add_argument("--backend", type=str, default="both",
                        choices=["triton", "csgmv", "both"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--rank-values", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--output-dim", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=1,
                        help="Tokens per sequence: 1=decode, >1=prefill")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["homogeneous", "heterogeneous", "all"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--fit", action="store_true",
                        help="Run baseline fit + exhaustive predictor search")
    parser.add_argument("--save-csv", type=str, default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required."
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Backend(s):  {args.backend}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Rank values: {args.rank_values}")
    print(f"Output dim:  {args.output_dim}")
    print(f"Seq len:     {args.seq_len}  ({'decode' if args.seq_len == 1 else 'prefill'})")
    print(f"Mode:        {args.mode}")
    print("=" * 70)

    rank_configs = build_rank_configs(args.batch_sizes, args.rank_values, args.mode)
    all_results = []

    if args.backend in ("triton", "both"):
        res = bench_triton(rank_configs, args.output_dim, args.seq_len,
                           args.warmup, args.repeat)
        all_results.extend(res)
        if args.fit:
            fit_and_explore(res, "triton")

    if args.backend in ("csgmv", "both"):
        res = bench_csgmv(rank_configs, args.output_dim, args.seq_len,
                          args.warmup, args.repeat)
        all_results.extend(res)
        if args.fit:
            fit_and_explore(res, "csgmv")

    if args.save_csv and all_results:
        import csv
        # build full field list (features are dynamic)
        all_keys = list(all_results[0].keys())
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for row in all_results:
                row = dict(row)
                row["ranks"] = str(row["ranks"])
                writer.writerow(row)
        print(f"\nResults saved to: {args.save_csv}")

    print("\n" + "=" * 70)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()