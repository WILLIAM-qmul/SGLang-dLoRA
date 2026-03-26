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
    # Prefill: latency_ms = pre_alpha * predictor(S, seq_len) + pre_beta
    # Default: same slope as decode scaled by seq_len (good approximation)
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


def _pred_pre(backend: str, ranks: List[int], avg_seq_len: float) -> float:
    """Prefill predictor: decode predictor × avg_seq_len (approximate)."""
    return _pred_dec(backend, ranks) * max(avg_seq_len, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Engine state snapshot
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EngineRankState:
    """
    Lightweight snapshot from GET /get_rank_aware_stats on one engine.
    Only rank integers — no paths, no output_dim.
    """
    engine_id:     int
    running_ranks: List[int] = field(default_factory=list)
    waiting_ranks: List[int] = field(default_factory=list)
    timestamp:     float = field(default_factory=time.time)
    fetch_ok:      bool = True   # False → HTTP error; treat as empty

    @classmethod
    def from_dict(cls, engine_id: int, data: dict) -> "EngineRankState":
        return cls(
            engine_id=engine_id,
            running_ranks=[int(r) for r in data.get("running_ranks", [])],
            waiting_ranks=[int(r) for r in data.get("waiting_ranks", [])],
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
        self.stats_timeout   = stats_timeout

        # Merge SLO into a copy of params so callers can override
        self.params = perf_params or PERF_PARAMS[backend]
        self.params.slo_ms = slo_ms

        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(
            f"[RankAwareScheduler] backend={backend}  "
            f"default_req_rank={default_req_rank}  avg_resp_len={avg_resp_len}  "
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
        """DecPerf(S) in ms."""
        p = self.params
        return p.dec_alpha * _pred_dec(self.backend, ranks) + p.dec_beta

    def _pre_perf(self, ranks: List[int]) -> float:
        """PrePerf(S) in ms (uses avg_resp_len as proxy for avg seq_len)."""
        p = self.params
        return p.pre_alpha * _pred_pre(self.backend, ranks, self.avg_resp_len) + p.pre_beta

    # ── Algorithm 1: CalcCost ─────────────────────────────────────────────────

    def _calc_cost(
        self,
        req_rank:      int,
        running_ranks: List[int],
        waiting_ranks: List[int],
    ) -> float:
        """
        Algorithm 1, Function CalcCost(req, running_batch, queue):

            exists   = running_batch + queue
            Δprefill = PrePerf(queue + req) − PrePerf(queue)
            Δdecode  = DecPerf(exists + req) − DecPerf(exists)
            cost     = (Δprefill / avg_resp_len) + Δdecode
            if DecPerf(exists + req) > SLO:  cost += inf
        """
        exists_ranks = running_ranks + waiting_ranks

        delta_prefill = (
            self._pre_perf(waiting_ranks + [req_rank])
            - self._pre_perf(waiting_ranks)
        )
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
        candidate_ids: Optional[List[int]] = None,
    ) -> int:
        """
        Algorithm 1, main loop body.

        Parameters
        ----------
        req_rank      : LoRA rank of the incoming request (0 → use default)
        candidate_ids : engine IDs to consider; None = all

        Returns
        -------
        engine_id : int
        """
        if req_rank <= 0:
            req_rank = self.default_req_rank

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
            cost  = self._calc_cost(
                req_rank=req_rank,
                running_ranks=state.running_ranks,
                waiting_ranks=state.waiting_ranks,
            )
            # Algorithm 1: total_cost = cost × (|running| + |waiting|)
            num_reqs   = max(state.total_requests, 1)
            total_cost = cost * num_reqs

            logger.debug(
                f"[RankAware] engine={eid}  req_rank={req_rank}  "
                f"run={len(state.running_ranks)}  wait={len(state.waiting_ranks)}  "
                f"cost={cost:.6f}  total={total_cost:.6f}"
            )

            if total_cost < best_cost:
                best_cost = total_cost
                best_eid  = eid

        logger.info(
            f"[RankAware] req_rank={req_rank} → engine {best_eid}  "
            f"(total_cost={best_cost:.6f})"
        )
        return best_eid

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()