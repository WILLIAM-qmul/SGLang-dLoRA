"""
rank_aware_scheduler.py
========================
TOPPINGS Algorithm 1 — Rank-Aware Scheduling Policy.

Performance models (profiled on RTX 4090, Llama-2-7b, all 4 modules, seq_len=1):

  Triton:
    Best predictor : bs_x_max_rank = batch_size × max(rank_i)
    Formula        : latency_ms = 0.00225994 · bs_x_max_rank + 17.98761
    R²             = 0.869

  CSGMV:
    Best predictor : sqrt_bs_x_max_rank = sqrt(batch_size × max(rank_i))
    Formula        : latency_ms = 0.03286053 · sqrt_bs_x_max_rank + 17.55006
    R²             = 0.542

Algorithm 1 (CalcCost) — Updated:
    exists   = running_batch + queue
    Δprefill = Σ_i [ DecPerf(queue[:i] + req) - DecPerf(queue[:i]) ] × seq_len_i
             (per-request prefill cost, each scaled by its own input length)
    Δdecode  = DecPerf(exists + req) − DecPerf(exists)
    cost     = (Δprefill / avg_resp_len) + Δdecode
    if DecPerf(exists + req) > SLO:  cost += inf
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Performance model parameters — update after re-profiling on new hardware
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PerfModelParams:
    """Linear decode/prefill latency model parameters for one backend."""
    # Decode: latency_ms = dec_alpha * predictor(S) + dec_beta
    dec_alpha: float
    dec_beta: float
    # Prefill: now computed as DecPerf(S) × seq_len per request
    # pre_alpha / pre_beta kept for backward compat but no longer used in _pre_perf
    pre_alpha: float
    pre_beta: float
    # SLO per decode step (ms)
    slo_ms: float = 300.0


# Model Performance (RTX 4090, Llama-2-7b)
PERF_PARAMS: Dict[str, PerfModelParams] = {
    "triton": PerfModelParams(
        dec_alpha=0.00225994,
        dec_beta=17.98761,
        pre_alpha=0.00225994,
        pre_beta=17.98761,
    ),
    "csgmv": PerfModelParams(
        dec_alpha=0.03286053,
        dec_beta=17.55006,
        pre_alpha=0.03286053,
        pre_beta=17.55006,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Predictor functions (pure, side-effect-free)
# ─────────────────────────────────────────────────────────────────────────────

def _pred_dec(backend: str, ranks: List[int], bs: int = 1) -> float:
    """Decode-step predictor value for a given rank list."""
    if not ranks:
        return 0.0
    max_r = max(ranks)
    if backend == "triton":
        # bs_x_max_rank
        return float(bs * max_r)
    else:
        # sqrt_bs_x_max_rank
        return float(math.sqrt(bs * max_r))


# ─────────────────────────────────────────────────────────────────────────────
# Engine state snapshot
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EngineRankState:
    """
    Lightweight snapshot from GET /get_rank_aware_stats on one engine.
    Now includes per-request seq_len for waiting requests.
    """
    engine_id:       int
    running_ranks:   List[int] = field(default_factory=list)
    waiting_ranks:   List[int] = field(default_factory=list)
    waiting_seq_lens: List[int] = field(default_factory=list)   # NEW: per-request input length
    timestamp:       float = field(default_factory=time.time)
    fetch_ok:        bool = True   # False → HTTP error; treat as empty

    @classmethod
    def from_dict(cls, engine_id: int, data: dict) -> "EngineRankState":
        return cls(
            engine_id=engine_id,
            running_ranks=[int(r) for r in data.get("running_ranks", [])],
            waiting_ranks=[int(r) for r in data.get("waiting_ranks", [])],
            waiting_seq_lens=[int(s) for s in data.get("waiting_seq_lens", [])],
            timestamp=data.get("timestamp", time.time()),
            fetch_ok=True,
        )

    @classmethod
    def empty(cls, engine_id: int) -> "EngineRankState":
        return cls(engine_id=engine_id, fetch_ok=False)

    @property
    def total_requests(self) -> int:
        return len(self.running_ranks) + len(self.waiting_ranks)


# ─────────────────────────────────────────────────────────────────────────────
# Rank-Aware Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class RankAwareScheduler:
    """
    TOPPINGS Algorithm 1 — Rank-Aware Scheduling Policy.

    Usage
    -----
    scheduler = RankAwareScheduler(
        instance_urls=["http://host:30001", "http://host:30002"],
        backend="csgmv",           # or "triton"
        avg_resp_len=200.0,        # expected generation length (tokens)
        slo_ms=300.0,              # SLO per decode step (ms)
        default_req_rank=16,       # fallback rank when caller provides 0
    )

    # On each incoming request:
    engine_id = await scheduler.select_engine(
        req_rank=32,               # LoRA rank of the incoming request
        req_seq_len=256,           # input sequence length of the new request
        candidate_ids=[0, 1],      # optional subset; None = all
    )
    """

    def __init__(
        self,
        instance_urls:    List[str],
        backend:          str = "csgmv",
        avg_resp_len:     float = 200.0,
        slo_ms:           float = 300.0,
        default_req_rank: int = 16,
        default_seq_len:  int = 256,
        stats_timeout:    float = 1.0,
        perf_params:      Optional[PerfModelParams] = None,
    ):
        if backend not in PERF_PARAMS:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose from: {list(PERF_PARAMS)}"
            )
        self.instance_urls   = instance_urls
        self.num_instances   = len(instance_urls)
        self.backend         = backend
        self.avg_resp_len    = max(avg_resp_len, 1.0)
        self.default_req_rank = max(default_req_rank, 1)
        self.default_seq_len  = max(default_seq_len, 1)
        self.stats_timeout   = stats_timeout

        # Merge SLO into a copy of params so callers can override
        self.params = perf_params or PERF_PARAMS[backend]
        self.params.slo_ms = slo_ms

        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(
            f"[RankAwareScheduler] backend={backend}  "
            f"default_req_rank={default_req_rank}  avg_resp_len={avg_resp_len}  "
            f"default_seq_len={default_seq_len}  "
            f"SLO={slo_ms}ms  engines={self.num_instances}"
        )

    # ── HTTP helpers ─────────────────────────────────────────────────────────

    async def _session_(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.stats_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _fetch_one(
        self, session: aiohttp.ClientSession, eid: int
    ) -> EngineRankState:
        """Fetch /get_rank_aware_stats from one engine; return empty on error."""
        url = f"{self.instance_urls[eid]}/get_rank_aware_stats"
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning(
                        f"[RankAware] engine {eid}: HTTP {resp.status}"
                    )
                    return EngineRankState.empty(eid)
                data = await resp.json()
                return EngineRankState.from_dict(eid, data)
        except asyncio.TimeoutError:
            logger.warning(f"[RankAware] engine {eid}: timeout")
            return EngineRankState.empty(eid)
        except Exception as exc:
            logger.error(f"[RankAware] engine {eid}: {exc}")
            return EngineRankState.empty(eid)

    async def _fetch_all(
        self, candidate_ids: List[int]
    ) -> Dict[int, EngineRankState]:
        """Fetch rank states from all candidate engines in parallel."""
        session = await self._session_()
        tasks   = [self._fetch_one(session, eid) for eid in candidate_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        states: Dict[int, EngineRankState] = {}
        for eid, res in zip(candidate_ids, results):
            if isinstance(res, Exception):
                logger.error(f"[RankAware] engine {eid} exception: {res}")
                states[eid] = EngineRankState.empty(eid)
            else:
                states[eid] = res
        return states

    # ── Performance model ────────────────────────────────────────────────────

    def _dec_perf(self, ranks: List[int]) -> float:
        """DecPerf(S) in ms — decode-step latency for rank set S."""
        p = self.params
        return p.dec_alpha * _pred_dec(self.backend, ranks) + p.dec_beta

    def _pre_perf_per_request(
        self,
        waiting_ranks: List[int],
        waiting_seq_lens: List[int],
        new_req_rank: int,
        new_req_seq_len: int,
    ) -> float:
        """
        Compute the total prefill cost of adding the new request to the
        waiting queue. Each request's prefill cost = its decode-step
        marginal cost × its own input sequence length.

        Δprefill = Σ_i [ (DecPerf(S[:i] ∪ {req}) - DecPerf(S[:i])) × seq_len_i ]

        where i iterates over all requests that will be prefilled together
        (the waiting queue + the new request).
        """
        # All requests to be prefilled: existing waiting + new request
        all_ranks    = list(waiting_ranks) + [new_req_rank]
        all_seq_lens = list(waiting_seq_lens) + [new_req_seq_len]

        total_prefill_cost = 0.0
        accumulated_ranks: List[int] = []

        for rank_i, seq_len_i in zip(all_ranks, all_seq_lens):
            # Marginal decode cost of adding request i to the accumulated set
            perf_with    = self._dec_perf(accumulated_ranks + [rank_i])
            perf_without = self._dec_perf(accumulated_ranks)
            marginal_decode = perf_with - perf_without

            # Prefill cost for this request = marginal decode × its input length
            total_prefill_cost += marginal_decode * max(seq_len_i, 1)

            accumulated_ranks.append(rank_i)

        return total_prefill_cost

    # ── Algorithm 1: CalcCost (Updated) ──────────────────────────────────────

    def _calc_cost(
        self,
        req_rank:         int,
        req_seq_len:      int,
        running_ranks:    List[int],
        waiting_ranks:    List[int],
        waiting_seq_lens: List[int],
    ) -> float:
        """
        Algorithm 1, Function CalcCost(req, running_batch, queue) — Updated:

            exists   = running_batch + queue

            # Per-request prefill: each request's decode marginal × its seq_len
            Δprefill = _pre_perf_per_request(queue, new_req)

            # Baseline: prefill cost of queue alone (without the new request)
            baseline_prefill = _pre_perf_per_request(queue_without_new, ...)

            Δdecode  = DecPerf(exists + req) − DecPerf(exists)
            cost     = ((Δprefill - baseline_prefill) / avg_resp_len) + Δdecode
            if DecPerf(exists + req) > SLO:  cost += inf
        """
        exists_ranks = running_ranks + waiting_ranks

        # Prefill cost WITH new request in the queue
        prefill_with_new = self._pre_perf_per_request(
            waiting_ranks=waiting_ranks,
            waiting_seq_lens=waiting_seq_lens,
            new_req_rank=req_rank,
            new_req_seq_len=req_seq_len,
        )

        # Prefill cost WITHOUT new request (baseline)
        # Use a dummy 0-rank, 0-len request to get just the existing queue cost
        prefill_baseline = self._pre_perf_per_request(
            waiting_ranks=[],
            waiting_seq_lens=[],
            new_req_rank=0,
            new_req_seq_len=0,
        )
        # Actually recompute baseline as the sum over existing waiting only
        prefill_baseline = 0.0
        accumulated: List[int] = []
        for rank_i, seq_len_i in zip(waiting_ranks, waiting_seq_lens):
            perf_with    = self._dec_perf(accumulated + [rank_i])
            perf_without = self._dec_perf(accumulated)
            prefill_baseline += (perf_with - perf_without) * max(seq_len_i, 1)
            accumulated.append(rank_i)

        delta_prefill = prefill_with_new - prefill_baseline

        # Decode cost delta
        perf_exists_plus = self._dec_perf(exists_ranks + [req_rank])
        delta_decode     = perf_exists_plus - self._dec_perf(exists_ranks)

        cost = (delta_prefill / self.avg_resp_len) + delta_decode

        if perf_exists_plus > self.params.slo_ms:
            cost += math.inf

        return cost

    # ── Public API ────────────────────────────────────────────────────────────

    async def select_engine(
        self,
        req_rank:      int,
        req_seq_len:   int = 0,
        candidate_ids: Optional[List[int]] = None,
    ) -> int:
        """
        Algorithm 1, main loop body.

        Parameters
        ----------
        req_rank      : LoRA rank of the incoming request (0 → use default)
        req_seq_len   : input sequence length of the new request (0 → use default)
        candidate_ids : engine IDs to consider; None = all

        Returns
        -------
        engine_id : int
        """
        if req_rank <= 0:
            req_rank = self.default_req_rank
        if req_seq_len <= 0:
            req_seq_len = self.default_seq_len

        ids = candidate_ids if candidate_ids is not None else list(
            range(self.num_instances)
        )
        if not ids:
            logger.warning("[RankAware] no candidate engines; returning 0")
            return 0

        states = await self._fetch_all(ids)

        best_eid   = ids[0]
        best_cost  = math.inf

        for eid in ids:
            state = states[eid]

            # Ensure waiting_seq_lens matches waiting_ranks length
            w_seq_lens = state.waiting_seq_lens
            if len(w_seq_lens) != len(state.waiting_ranks):
                # Fallback: use default_seq_len for missing entries
                w_seq_lens = [
                    w_seq_lens[i] if i < len(w_seq_lens) else self.default_seq_len
                    for i in range(len(state.waiting_ranks))
                ]

            cost = self._calc_cost(
                req_rank=req_rank,
                req_seq_len=req_seq_len,
                running_ranks=state.running_ranks,
                waiting_ranks=state.waiting_ranks,
                waiting_seq_lens=w_seq_lens,
            )
            # Algorithm 1: total_cost = cost × (|running| + |waiting|)
            num_reqs   = max(state.total_requests, 1)
            total_cost = cost * num_reqs

            logger.debug(
                f"[RankAware] engine={eid}  req_rank={req_rank}  "
                f"req_seq_len={req_seq_len}  "
                f"run={len(state.running_ranks)}  wait={len(state.waiting_ranks)}  "
                f"cost={cost:.6f}  total={total_cost:.6f}"
            )

            if total_cost < best_cost:
                best_cost = total_cost
                best_eid  = eid

        logger.info(
            f"[RankAware] req_rank={req_rank} seq_len={req_seq_len} "
            f"→ engine {best_eid}  (total_cost={best_cost:.6f})"
        )
        return best_eid

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()