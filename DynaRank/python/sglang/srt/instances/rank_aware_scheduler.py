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

Algorithm 1 (CalcCost):
    exists   = running_batch + queue
    Δprefill = PrePerf(queue + req) − PrePerf(queue)
    Δdecode  = DecPerf(exists + req) − DecPerf(exists)
    cost     = (Δprefill / avg_resp_len) + Δdecode
    if DecPerf(exists + req) > SLO:  cost += inf

Where:
    DecPerf(S) = alpha * predictor(bs=|S|, max_rank=max(S)) + beta
    PrePerf(S) = DecPerf(S) × max_seq_len(S)
      (one decode step for the whole batch, scaled by the longest input sequence)
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
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class PerfModelParams:
    """Linear decode/prefill latency model parameters for one backend."""
    # Decode: latency_ms = dec_alpha * predictor(S) + dec_beta
    dec_alpha: float
    dec_beta: float
    # Prefill: kept for backward compat; PrePerf now = DecPerf × max_seq_len
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
    """
    Decode-step predictor value for a given rank list.

    Parameters
    ----------
    backend : "triton" or "csgmv"
    ranks   : list of LoRA ranks in the batch (used to get max_rank)
    bs      : batch size (= number of requests in the batch = len(ranks))
    """
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
    Includes per-request seq_len for waiting requests.
    """
    engine_id:       int
    running_ranks:   List[int] = field(default_factory=list)
    waiting_ranks:   List[int] = field(default_factory=list)
    waiting_seq_lens: List[int] = field(default_factory=list)
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
        """
        DecPerf(S) in ms — decode-step latency for rank set S.

        Uses len(ranks) as the batch size, so the predictor correctly
        reflects the cost of processing the entire batch in one step.

        FIX: Previously called _pred_dec with default bs=1; now passes
        bs=len(ranks) so the profiled formula is applied correctly.
        """
        if not ranks:
            p = self.params
            return p.dec_beta  # empty batch → just the base latency
        p = self.params
        bs = len(ranks)
        return p.dec_alpha * _pred_dec(self.backend, ranks, bs=bs) + p.dec_beta

    def _pre_perf(
        self,
        ranks:    List[int],
        seq_lens: List[int],
    ) -> float:
        """
        PrePerf(S) in ms — prefill latency for a set of requests S.

        Prefill processes the whole batch like one decode step, but the
        cost is proportional to the longest input sequence in the batch:

            PrePerf(S) = DecPerf(S) × max(seq_len_i for i in S)

        FIX: Previously used per-request iterative marginal costs.
        Now computes one whole-batch DecPerf and scales by max seq_len,
        matching the Algorithm 1 paper specification.
        """
        if not ranks:
            return 0.0
        dec_cost = self._dec_perf(ranks)
        max_seq_len = max(seq_lens) if seq_lens else 1
        return dec_cost * max(max_seq_len, 1)

    # ── Algorithm 1: CalcCost ─────────────────────────────────────────────────

    def _calc_cost(
        self,
        req_rank:         int,
        req_seq_len:      int,
        running_ranks:    List[int],
        waiting_ranks:    List[int],
        waiting_seq_lens: List[int],
    ) -> float:
        """
        Algorithm 1, Function CalcCost(req, running_batch, queue):

            exists   = running_batch + queue

            # Prefill cost: whole-batch decode cost × max input seq_len
            Δprefill = PrePerf(queue + req) − PrePerf(queue)

            # Decode cost: marginal decode cost of adding req to exists
            Δdecode  = DecPerf(exists + req) − DecPerf(exists)

            cost     = (Δprefill / avg_resp_len) + Δdecode
            if DecPerf(exists + req) > SLO:  cost += inf
        """
        exists_ranks = running_ranks + waiting_ranks

        # ── Δprefill ─────────────────────────────────────────────────────
        # PrePerf(queue + req) - PrePerf(queue)
        # Each is: DecPerf(batch) × max_seq_len(batch)
        queue_plus_req_ranks    = waiting_ranks + [req_rank]
        queue_plus_req_seq_lens = waiting_seq_lens + [req_seq_len]

        prefill_with_new = self._pre_perf(queue_plus_req_ranks, queue_plus_req_seq_lens)
        prefill_baseline = self._pre_perf(waiting_ranks, waiting_seq_lens)

        delta_prefill = prefill_with_new - prefill_baseline

        # ── Δdecode ──────────────────────────────────────────────────────
        # DecPerf(exists + req) - DecPerf(exists)
        perf_exists_plus = self._dec_perf(exists_ranks + [req_rank])
        delta_decode     = perf_exists_plus - self._dec_perf(exists_ranks)

        cost = (delta_prefill / self.avg_resp_len) + delta_decode

        # SLO violation penalty
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