#!/usr/bin/env python3
"""Replica autoscaling controllers, all behind one interface.

    decide(state_history: List[ClusterState], t: float) -> int   # target replicas

`state_history` is the list of scrapes so far, oldest first, each in the REAL
router's vocabulary (see sim_cluster.ClusterState).  A controller may read only:
  * the six VERIFIED vllm:* metric families (per ready replica),
  * `state.router_inflight` -- per-request tokens-emitted counts, which the real
    router observes because it proxies the token stream,
  * `state.arrivals_since_last` / `arrival_classes_since_last` -- request counts
    and the thinking_mode the router itself set on each request,
  * `n_ready / n_booting / n_draining / target` -- its own actuation state.
Nothing else.  `Oracle` deliberately violates this and is labelled accordingly.

THE SIX CONTROLLERS
-------------------
  1 StaticK              reference points; the trivial Pareto front
  2 HPA                  Kubernetes HPA semantics (see HPA_SEMANTICS_SOURCE)
  3 QueueConcurrency     target per-replica in-flight concurrency
  4 LeadingIndicator     TokenScale-style work-velocity threshold (strongest reactive)
  5 ForecastThreshold    EWMA / Holt-Winters / ridge forecast + the same threshold
                         (forecast-then-provision baseline)
  6 Oracle               knows true output lengths at admission; upper bound

WHAT IS FAITHFUL AND WHAT IS APPROXIMATED is documented per class below and
Read the class docstrings before citing any baseline as
"the published method": several are our reimplementations in spirit, not the
authors' code, and one (LeadingIndicator) cannot be fully reproduced here
because we have no disaggregated prefill/decode deployment.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from length_model import LengthModel, load_length_model

# Source of the HPA semantics implemented in `HPA`, recorded verbatim so a
# reviewer can check the version rather than trust our memory.
HPA_SEMANTICS_SOURCE = (
    "Kubernetes documentation, 'Horizontal Pod Autoscaling' "
    "(kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/), "
    "page last modified 2026-03-15, retrieved 2026-08-12. Implements: the "
    "desiredReplicas = ceil(currentReplicas * currentMetricValue / desiredMetricValue) "
    "ratio rule; the tolerance band (default 0.1, documented as FEATURE STATE "
    "Kubernetes v1.35 [beta] for the per-direction `tolerance` field, and as the "
    "cluster-wide --horizontal-pod-autoscaler-tolerance default before that); the "
    "scaleDown stabilization window (default 300 s, rolling MAXIMUM over previously "
    "computed desired states); scaleUp stabilization window default 0; and the "
    "default scaling policies (scaleUp: 100%/15s and 4 pods/15s with selectPolicy Max; "
    "scaleDown: 100%/15s). Control cadence 15 s matches the documented default "
    "--horizontal-pod-autoscaler-sync-period."
)


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------
class Controller:
    """Base class.  Subclasses implement `decide`."""

    name = "base"
    #: parameters the tuner is allowed to sweep -> handled by tune.py
    param_space: Dict[str, Sequence] = {}
    #: set True only for controllers that read privileged information
    uses_privileged_info = False

    def __init__(self, cfg, **params):
        self.cfg = cfg
        self.k_min = int(cfg.f("n_replicas_min"))
        self.k_max = int(cfg.f("n_replicas_max"))
        self.dt = float(cfg.f("control_interval_s"))
        self.d = float(cfg.f("replica_cold_start_s"))
        self.params = dict(params)
        for k, v in params.items():
            setattr(self, k, v)
        self.log: List[Dict] = []

    def clamp(self, k) -> int:
        return int(max(self.k_min, min(self.k_max, int(math.ceil(k - 1e-9)))))

    def decide(self, hist: List, t: float) -> int:
        raise NotImplementedError

    def describe(self) -> Dict:
        return {"name": self.name, "params": dict(self.params),
                "uses_privileged_info": self.uses_privileged_info}

    # -- shared signal helpers -----------------------------------------
    def arrival_rate(self, hist: List, window_s: float = 60.0) -> float:
        """Observed request arrival rate over a trailing window (req/s)."""
        if not hist:
            return 0.0
        t_now = hist[-1].t
        n, span = 0, 0.0
        prev_t = None
        for st in reversed(hist):
            if t_now - st.t > window_s:
                break
            n += st.arrivals_since_last
            prev_t = st.t
        if prev_t is None:
            return 0.0
        span = max(t_now - prev_t, self.dt)
        return n / span

    def token_rate(self, hist: List, window_s: float = 60.0) -> float:
        """Observed generation throughput (tok/s) from the counter difference.

        Uses vllm:generation_tokens_total, which is a per-replica COUNTER.  A
        replica that goes away takes its counter with it, so the difference is
        clamped at zero -- otherwise a scale-down reads as negative throughput.
        """
        if len(hist) < 2:
            return 0.0
        t_now = hist[-1].t
        older = None
        for st in reversed(hist[:-1]):
            older = st
            if t_now - st.t >= window_s:
                break
        if older is None:
            return 0.0
        dt = max(hist[-1].t - older.t, 1e-6)
        d = hist[-1].total("vllm:generation_tokens_total") - older.total("vllm:generation_tokens_total")
        return max(0.0, d) / dt


# ---------------------------------------------------------------------------
# 1. STATIC-k
# ---------------------------------------------------------------------------
class StaticK(Controller):
    """Hold k replicas for the whole run.

    Sweeping k over the feasible range gives the reference points and the trivial
    cost-SLO Pareto front that any adaptive controller must beat to be worth its
    complexity.  On a 4-replica testbed this is a 4-point front, so it is a
    genuinely strong baseline: an autoscaler only wins if it lands strictly
    inside it (cheaper at equal SLO, or better SLO at equal cost).
    """
    name = "static"
    param_space = {"k": (1, 2, 3, 4)}

    def __init__(self, cfg, k: int = 2, **kw):
        super().__init__(cfg, k=k, **kw)

    def decide(self, hist: List, t: float) -> int:
        return self.clamp(self.k)


# ---------------------------------------------------------------------------
# 2. HPA-equivalent
# ---------------------------------------------------------------------------
class HPA(Controller):
    """Kubernetes Horizontal Pod Autoscaler semantics.

    FAITHFUL to the documented algorithm (source string: HPA_SEMANTICS_SOURCE):
      * the ratio rule desiredReplicas = ceil(currentReplicas * cur/target),
        computed on the AVERAGE of a per-pod metric across ready pods, which is
        what targetAverageValue / targetAverageUtilization mean;
      * the tolerance band: no action while |cur/target - 1| <= tolerance
        (default 0.1);
      * the scale-DOWN stabilization window: the controller takes the MAXIMUM of
        the desired replica counts computed over the trailing window (documented
        as "uses the highest value from the specified interval", approximating a
        rolling maximum), default 300 s;
      * the scale-UP stabilization window, default 0 s;
      * the default scaling POLICIES: scale-up limited to max(100% of current,
        4 pods) per 15 s with selectPolicy: Max; scale-down limited to 100% per
        15 s (i.e. effectively unlimited down to minReplicas);
      * currentReplicas counts pods that exist but are not yet Ready (a booting
        replica is in the deployment's replica count), matching the fact that HPA
        scales a Deployment's `spec.replicas`.

    APPROXIMATED / NOT REPRODUCED, deliberately:
      * The documented special handling of MISSING metrics and not-yet-ready pods
        (missing metrics assumed at 100% of target for scale-down and 0% for
        scale-up; CPU metrics set aside during
        --horizontal-pod-autoscaler-initial-readiness-delay / 
        --horizontal-pod-autoscaler-cpu-initialization-period).  In our setting a
        booting vLLM replica exposes NO metrics endpoint at all, so it is simply
        absent from the average; we do not synthesise a 0%/100% reading for it.
        This is the honest analogue, and it makes HPA slightly LESS jumpy than
        real HPA during a scale-up, i.e. it does not disadvantage the baseline.
      * Multiple metrics / selectPolicy across several metric sources: we drive a
        single metric, which is the standard single-signal deployment.
      * The `behavior.tolerance` field is applied symmetrically via one
        `tolerance` parameter (the API allows separate scaleUp/scaleDown values).

    METRIC CHOICE IS A TUNED PARAMETER, not an assumption.  A CPU-utilization HPA
    is meaningless for an LLM server (the GPU is the resource), so the sweep gives
    HPA its best shot across three honest utilization signals available from the
    verified metric set:
      "kv"        mean vllm:kv_cache_usage_perc          (GPU-memory pressure)
      "conc"      mean (running + waiting) per replica   (concurrency)
      "running"   mean vllm:num_requests_running         (batch occupancy)
    """
    name = "hpa"
    #: NOTE the target space is PER METRIC.  A single shared `target` tuple would
    #: be a broken sweep: `kv` is a FRACTION in [0,1], so a target of 12.0 is
    #: unreachable and the ratio rule would never scale up -- a degenerate config
    #: that silently turns HPA into static-1.  `param_space_for` expands the
    #: correct grid per metric, and tune.py uses it instead of the flat product.
    param_space = {
        "metric": ("kv", "conc", "running"),
        "target": (0.3, 0.4, 0.5, 0.6, 0.7, 0.8),      # kv default range
        "tolerance": (0.1,),
        "down_stabilization_s": (60.0, 300.0),
    }
    TARGETS_BY_METRIC = {
        "kv":      (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        "conc":    (2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0, 30.0),
        "running": (2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0, 30.0),
    }

    @classmethod
    def param_space_for(cls) -> List[Dict]:
        """Explicit config list with metric-appropriate targets."""
        out = []
        for m in cls.param_space["metric"]:
            for tg in cls.TARGETS_BY_METRIC[m]:
                for tol in cls.param_space["tolerance"]:
                    for ds in cls.param_space["down_stabilization_s"]:
                        out.append({"metric": m, "target": tg, "tolerance": tol,
                                    "down_stabilization_s": ds})
        return out

    def __init__(self, cfg, metric: str = "kv", target: float = 0.6,
                 tolerance: float = 0.1, down_stabilization_s: float = 300.0,
                 up_stabilization_s: float = 0.0, **kw):
        super().__init__(cfg, metric=metric, target=target, tolerance=tolerance,
                         down_stabilization_s=down_stabilization_s,
                         up_stabilization_s=up_stabilization_s, **kw)
        self._desired_hist: List[Tuple[float, int]] = []
        self._last_change_t = -1e9
        self._last_k = None

    def _metric_value(self, st) -> float:
        if self.metric == "kv":
            return st.mean("vllm:kv_cache_usage_perc")
        if self.metric == "conc":
            n = max(st.n_ready, 1)
            return (st.n_running + st.n_waiting) / n
        if self.metric == "running":
            return st.n_running / max(st.n_ready, 1)
        raise ValueError(f"unknown HPA metric {self.metric!r}")

    def decide(self, hist: List, t: float) -> int:
        st = hist[-1]
        # currentReplicas: pods in the deployment, including booting/draining
        cur = max(st.n_replicas_effective, self.k_min)
        m = self._metric_value(st)
        if self.target <= 0:
            return self.clamp(cur)
        ratio = m / self.target
        # tolerance band: skip scaling when sufficiently close to 1.0
        if abs(ratio - 1.0) <= self.tolerance:
            desired = cur
        else:
            desired = int(math.ceil(cur * ratio))
        desired = max(self.k_min, min(self.k_max, desired))
        self._desired_hist.append((t, desired))

        if desired > cur:
            # scaleUp stabilization window (default 0 -> immediate); when >0 the
            # documented behaviour for the opposite direction is a rolling MIN
            if self.up_stabilization_s > 0:
                w = [d for (tt, d) in self._desired_hist if t - tt <= self.up_stabilization_s]
                desired = min(w) if w else desired
            # scaleUp policies: max(100% of current, 4 pods) per 15 s, selectPolicy Max
            allowed = max(cur + 4, cur * 2)
            desired = min(desired, allowed)
        elif desired < cur:
            # scaleDown stabilization window: rolling MAXIMUM of desired states
            w = [d for (tt, d) in self._desired_hist if t - tt <= self.down_stabilization_s]
            desired = max(w) if w else desired
            # scaleDown policy: 100% per 15 s -> may go straight to minReplicas
            desired = max(desired, self.k_min)
        return self.clamp(desired)


# ---------------------------------------------------------------------------
# 3. QUEUE / CONCURRENCY target
# ---------------------------------------------------------------------------
class QueueConcurrency(Controller):
    """Target a per-replica in-flight concurrency.

    This is the family used by Knative-style and vLLM-production-stack-style
    autoscalers: pick the replica count that would put average in-flight
    concurrency (running + waiting) at a target.  Unlike HPA it is not a
    multiplicative ratio on the CURRENT count, so it does not inherit HPA's
    dependence on how many replicas happen to be up:

        desired = ceil((running + waiting) / target_concurrency)

    Optional hysteresis (`hyst`) suppresses changes smaller than a fraction of
    the current count, and `cooldown_s` blocks a reversal within a window; both
    are given to the tuner because without them this controller thrashes on S5
    and a thrashing baseline is an unfair one.
    """
    name = "queue"
    param_space = {
        "target_concurrency": (2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0),
        "hyst": (0.0, 0.15),
        "cooldown_s": (0.0, 30.0, 60.0),
    }

    def __init__(self, cfg, target_concurrency: float = 8.0, hyst: float = 0.0,
                 cooldown_s: float = 0.0, **kw):
        super().__init__(cfg, target_concurrency=target_concurrency, hyst=hyst,
                         cooldown_s=cooldown_s, **kw)
        self._last_change_t = -1e9
        self._last_dir = 0

    def decide(self, hist: List, t: float) -> int:
        st = hist[-1]
        inflight = st.n_running + st.n_waiting
        cur = max(st.n_replicas_effective, self.k_min)
        desired = int(math.ceil(inflight / max(self.target_concurrency, 1e-9)))
        desired = max(self.k_min, min(self.k_max, desired))
        if self.hyst > 0 and abs(desired - cur) <= self.hyst * max(cur, 1):
            desired = cur
        d = (desired > cur) - (desired < cur)
        if d != 0 and self.cooldown_s > 0:
            if (t - self._last_change_t) < self.cooldown_s and d != self._last_dir:
                return self.clamp(cur)
            self._last_change_t = t
            self._last_dir = d
        return self.clamp(desired)


# ---------------------------------------------------------------------------
# 4. LEADING-INDICATOR reactive (TokenScale-style)
# ---------------------------------------------------------------------------
class LeadingIndicator(Controller):
    """Work-velocity threshold controller: our strongest reactive baseline.

    IN THE SPIRIT OF TokenScale (arXiv 2512.03416), which we cite as established
    background for the finding that request-rate-threshold autoscalers lag on LLM
    serving.  We are NOT claiming that finding and we are NOT claiming to
    reproduce their system.

    WHAT WE REPRODUCE: the core idea that a token-level work signal leads a
    request-level one.  The controller drives on measured token throughput
    (vllm:generation_tokens_total differenced) PLUS its rate of change, so it
    responds to token load that is still building rather than to a queue that has
    already formed:

        signal = tok_rate + kappa * d(tok_rate)/dt          [tok/s]
        desired = ceil(signal / per_replica_token_capacity)

    and, because a saturated cluster cannot report a token rate higher than what
    it can serve, it takes the max with a backlog term over waiting requests --
    without this, throughput-driven control has a blind spot exactly when it is
    overloaded (measured throughput saturates, so the signal stops rising).

    WHAT WE EXPLICITLY DO NOT REPRODUCE, and why:
      * TokenScale's disaggregated PREFILL/DECODE deployment and its separate
        scaling of prefill and decode pools.  Our testbed runs 4 co-resident
        monolithic vLLM replicas on ONE GPU; there is no prefill/decode split to
        scale independently.  Any comparison here is therefore between our
        reimplementation of the signal idea and the other controllers, on this topology.
      * Their SLO-attainment predictor and their admission/routing changes.  We
        scale replicas only, which is the decision this benchmark is about.
      * Their absolute thresholds and hyperparameters, which are specific to
        their model/hardware.  Ours are TUNED on our dev traces by the same
        protocol as every other baseline (tune.py), which is the only fair way to
        compare given we cannot reuse their constants.
    UNVERIFIED: we have not run the authors' code, so nothing here should be read
    as a measurement of TokenScale.
    """
    name = "leading"
    param_space = {
        "per_replica_tok_s": (200.0, 300.0, 400.0, 460.0),
        "kappa": (0.0, 5.0, 15.0, 30.0),
        "window_s": (30.0, 60.0),
        "backlog_target": (4.0, 8.0, 12.0),
    }

    def __init__(self, cfg, per_replica_tok_s: float = 400.0, kappa: float = 15.0,
                 window_s: float = 30.0, backlog_target: float = 8.0, **kw):
        super().__init__(cfg, per_replica_tok_s=per_replica_tok_s, kappa=kappa,
                         window_s=window_s, backlog_target=backlog_target, **kw)
        self._prev_rate = None
        self._prev_t = None

    def decide(self, hist: List, t: float) -> int:
        st = hist[-1]
        rate = self.token_rate(hist, self.window_s)
        drate = 0.0
        if self._prev_rate is not None and self._prev_t is not None and t > self._prev_t:
            drate = (rate - self._prev_rate) / (t - self._prev_t)
        self._prev_rate, self._prev_t = rate, t
        signal = max(0.0, rate + self.kappa * drate)
        desired_tok = signal / max(self.per_replica_tok_s, 1e-9)
        # saturation blind spot: a full cluster cannot report a higher token rate
        inflight = st.n_running + st.n_waiting
        desired_backlog = inflight / max(self.backlog_target, 1e-9)
        return self.clamp(max(desired_tok, desired_backlog, float(self.k_min)))


# ---------------------------------------------------------------------------
# 5. FORECAST-THEN-THRESHOLD  (the KEY baseline)
# ---------------------------------------------------------------------------
class _EWMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.level: Optional[float] = None

    def update(self, x: float) -> None:
        self.level = x if self.level is None else self.alpha * x + (1 - self.alpha) * self.level

    def forecast(self, h_steps: int) -> float:
        return 0.0 if self.level is None else self.level


class _HoltWinters:
    """Holt's linear (double-exponential) smoothing: level + trend.

    Named Holt-Winters in the task; we implement the TREND form without a
    seasonal component by default, and expose an optional additive seasonal term
    for the diurnal scenario S4.  With a 300 s horizon and a 15 s control
    interval there are only 20 control points per run, so a seasonal period must
    be short to be estimable at all -- `season_len=0` (off) is the honest default
    and the tuner is allowed to turn it on.
    """

    def __init__(self, alpha: float, beta: float, season_len: int = 0, gamma: float = 0.3):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.season_len = int(season_len)
        self.level: Optional[float] = None
        self.trend = 0.0
        self.season: List[float] = [0.0] * self.season_len if self.season_len else []
        self._i = 0

    def update(self, x: float) -> None:
        s_idx = self._i % self.season_len if self.season_len else None
        s = self.season[s_idx] if s_idx is not None else 0.0
        if self.level is None:
            self.level, self.trend = x, 0.0
        else:
            prev = self.level
            self.level = self.alpha * (x - s) + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - prev) + (1 - self.beta) * self.trend
        if s_idx is not None:
            self.season[s_idx] = self.gamma * (x - self.level) + (1 - self.gamma) * s
        self._i += 1

    def forecast(self, h_steps: int) -> float:
        if self.level is None:
            return 0.0
        v = self.level + h_steps * self.trend
        if self.season_len:
            v += self.season[(self._i + h_steps - 1) % self.season_len]
        return v


class ForecastThreshold(Controller):
    """Forecast the arrival process d+H seconds ahead, then apply a threshold rule.

    THIS IS THE PAPER'S KEY BASELINE.  The central ablation is planning-vs-
    forecasting, so this controller gets the same lookahead our method has: it
    forecasts to t + d + H, where d is the MEASURED cold start (31.9 s), so it is
    not handicapped by ignoring actuation lag.  Family: ADAPT (2605.15788) /
    SageServe-style forecast-then-provision.  It is our reimplementation of that
    pattern, not the authors' code (UNVERIFIED as a reproduction).

    THE DELIBERATE DESIGN POINT: what it forecasts is the ARRIVAL PROCESS.
    `signal="rate"` forecasts requests/s; `signal="tokens"` forecasts observed
    token throughput.  Both are extrapolations of a scalar time series.  On S3
    that is exactly the information a rate-driven controller cannot have: arrivals
    are byte-identical to the fixed-mix twin, so ANY forecaster of the arrival
    rate -- however good -- predicts no change, while offered token load rises
    steeply (measured on S3_dev: 405 -> 1895 tok/s, +368%, while the request rate
    moves -1.1%, inside Poisson counting noise).
    The `tokens` variant is the stronger and fairer version: it can see token
    load, but only AFTER it has been generated, so it still lags by construction
    rather than by weak tuning.

    To make sure the comparison is not just "our forecaster is better", three
    forecasters are provided and the tuner picks the best per scenario family:
      "ewma"  single exponential smoothing (level only)
      "holt"  Holt's linear trend (optionally + additive season)
      "ridge" a small LEARNED autoregressive forecaster: ridge regression on the
              last `ar_lags` observations, fit ON DEV TRACES ONLY by tune.py and
              passed in as `ar_coef`.  If no coefficients were fit, it falls back
              to Holt and records `ridge_fallback=True` so the fallback cannot be
              mistaken for a fitted model.
    """
    name = "forecast"
    param_space = {
        "forecaster": ("ewma", "holt", "ridge"),
        "signal": ("rate", "tokens"),
        "alpha": (0.2, 0.4, 0.7),
        "beta": (0.1, 0.3),
        "H_s": (0.0, 15.0, 30.0, 60.0),
        "per_replica_capacity": (0.5, 0.75, 1.0, 1.5),      # req/s, for signal="rate"
        "per_replica_tok_s": (200.0, 300.0, 400.0, 460.0),  # tok/s, for signal="tokens"
        "safety": (1.0, 1.2, 1.5),
    }

    def __init__(self, cfg, forecaster: str = "holt", signal: str = "tokens",
                 alpha: float = 0.4, beta: float = 0.3, H_s: float = 30.0,
                 per_replica_capacity: float = 1.0, per_replica_tok_s: float = 400.0,
                 safety: float = 1.2, season_len: int = 0, ar_lags: int = 4,
                 ar_coef: Optional[Sequence[float]] = None, **kw):
        super().__init__(cfg, forecaster=forecaster, signal=signal, alpha=alpha, beta=beta,
                         H_s=H_s, per_replica_capacity=per_replica_capacity,
                         per_replica_tok_s=per_replica_tok_s, safety=safety,
                         season_len=season_len, ar_lags=ar_lags, **kw)
        self.ar_coef = list(ar_coef) if ar_coef else None
        self.ridge_fallback = (forecaster == "ridge" and not self.ar_coef)
        self._ewma = _EWMA(alpha)
        self._holt = _HoltWinters(alpha, beta, season_len)
        self._series: List[float] = []

    def _observe(self, hist: List) -> float:
        if self.signal == "rate":
            return self.arrival_rate(hist, window_s=max(self.dt, 30.0))
        return self.token_rate(hist, window_s=max(self.dt, 30.0))

    def decide(self, hist: List, t: float) -> int:
        st = hist[-1]
        x = self._observe(hist)
        self._series.append(x)
        self._ewma.update(x)
        self._holt.update(x)
        # forecast to t + d + H, in units of the control interval
        h_steps = max(1, int(round((self.d + self.H_s) / self.dt)))
        if self.forecaster == "ewma":
            pred = self._ewma.forecast(h_steps)
        elif self.forecaster == "ridge" and self.ar_coef:
            lags = self.ar_coef[:-1]
            b0 = self.ar_coef[-1]
            xs = self._series[-len(lags):]
            if len(xs) < len(lags):
                xs = [self._series[0]] * (len(lags) - len(xs)) + xs
            pred = b0 + sum(c * v for c, v in zip(lags, reversed(xs)))
            # the ridge model is fit one-step-ahead; iterate it out to h_steps
            for _ in range(h_steps - 1):
                xs = xs[1:] + [pred]
                pred = b0 + sum(c * v for c, v in zip(lags, reversed(xs)))
        else:
            pred = self._holt.forecast(h_steps)
        pred = max(0.0, pred) * self.safety
        if self.signal == "rate":
            desired = pred / max(self.per_replica_capacity, 1e-9)
        else:
            desired = pred / max(self.per_replica_tok_s, 1e-9)
        # same backlog guard the reactive baselines get, so the forecaster is not
        # blind to a queue that has already formed
        inflight = st.n_running + st.n_waiting
        desired = max(desired, inflight / 12.0)
        return self.clamp(max(desired, float(self.k_min)))


# ---------------------------------------------------------------------------
# 6. ORACLE
# ---------------------------------------------------------------------------
class Oracle(Controller):
    """Privileged reference: knows every request's TRUE output length at admission.

    USES PRIVILEGED INFORMATION -- it is handed the trace and reads
    `output_tokens_true`, which no deployable controller can observe.  It is a
    bound, never a baseline, and `uses_privileged_info=True` makes every result
    row carry that fact.

    It is deliberately NOT a full optimal-control solution (which would require
    solving the whole trajectory offline): it computes exact residual token work
    over the requests in flight plus the exact token work of arrivals in the next
    d + H seconds, and provisions for it with the same analytic capacity model
    the capacity-based controllers use.  It is a reference for the value of
    KNOWING the realisation, i.e. for what a better length predictor could buy.
    It is not an upper bound on achievable cost: a deployable controller can
    beat it (see BASELINES.md).
    """
    name = "oracle"
    uses_privileged_info = True
    param_space = {"H_s": (0.0, 30.0, 60.0), "safety": (1.0, 1.2)}

    def __init__(self, cfg, trace: Optional[List] = None, H_s: float = 30.0,
                 safety: float = 1.0, **kw):
        super().__init__(cfg, H_s=H_s, safety=safety, **kw)
        self.trace = trace or []
        self._by_seq = {r.seq: r for r in self.trace}
        st = cfg.f("decode_step_time_s")
        self.a0 = float(st["a0_s"]); self.a1 = float(st["a1_s_per_seq"])
        self.kv_budget = float(cfg.f("kv_tokens_per_replica"))

    def _capacity_tok_s(self, n_ready: float, n_conc: float,
                        mean_footprint_tokens: float) -> float:
        """Token service rate of k replicas, RESPECTING THE KV BOUND.

        The KV bound is what actually limits an LLM replica: a sequence holds
        (prompt + generated) KV tokens, so at most
            b_max = k * kv_budget / mean_footprint
        sequences can be RESIDENT, and requests beyond that wait.  An earlier
        version of this method omitted the bound and reported that a single
        replica could serve 20 concurrent streams at 1065 tok/s; the oracle
        therefore concluded k=1 always sufficed and produced a 0.48 TTFT
        violation rate -- an "upper bound" worse than every baseline, which is
        how the bug was caught.
        """
        if n_ready <= 0:
            return 0.0
        b_max = n_ready * self.kv_budget / max(mean_footprint_tokens, 1.0)
        b = max(1.0, min(n_conc, b_max))
        per_rep = b / n_ready
        return n_ready * per_rep / (self.a0 + self.a1 * per_rep)

    def decide(self, hist: List, t: float) -> int:
        st = hist[-1]
        # exact residual work of in-flight requests (privileged: true lengths)
        W = 0.0
        fp = 0.0
        for r in st.router_inflight:
            true = self._by_seq.get(r["seq"])
            if true is not None:
                W += max(0, true.output_tokens_true - r["tokens_emitted"])
            fp += r["prompt_tokens"] + r["tokens_emitted"]
        # exact work of arrivals in the next d + H seconds
        t_end = t + self.d + self.H_s
        future = 0.0
        n_future = 0
        fut_fp = 0.0
        for r in self.trace:
            if t < r.t_arrival <= t_end:
                future += r.output_tokens_true
                n_future += 1
                # a new arrival's eventual KV footprint, known exactly here
                fut_fp += r.prompt_tokens + r.output_tokens_true * 0.5
        need_tokens = (W + future) * self.safety
        span = max(self.d + self.H_s, self.dt)
        need_tok_s = need_tokens / span
        n_conc = max(1.0, len(st.router_inflight) + n_future)
        mean_fp = max(1.0, (fp + fut_fp) / n_conc)
        # PEAK KV RESIDENCY, not just average token rate.  Requests in flight must
        # FIT in KV or they wait/preempt regardless of aggregate throughput: with
        # the measured 55200-token budget, k replicas can hold
        # k*55200/mean_footprint sequences.  Requiring residency is what makes this
        # an upper bound rather than an average-rate heuristic -- on S1 at 1
        # replica the average-rate test passed while the run took 342 preemptions
        # and a 58 s p95 TTFT.
        kv_need = (fp + fut_fp) * self.safety
        for k in range(self.k_min, self.k_max + 1):
            rate_ok = self._capacity_tok_s(k, n_conc, mean_fp) >= need_tok_s
            kv_ok = k * self.kv_budget >= kv_need
            if rate_ok and kv_ok:
                return self.clamp(k)
        return self.clamp(self.k_max)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
REGISTRY: Dict[str, type] = {
    "static": StaticK,
    "hpa": HPA,
    "queue": QueueConcurrency,
    "leading": LeadingIndicator,
    "forecast": ForecastThreshold,
    "oracle": Oracle,
}

CONTROLLER_ORDER = ["static", "hpa", "queue", "leading", "forecast", "oracle"]


def make(name: str, cfg, **params) -> Controller:
    if name not in REGISTRY:
        raise KeyError(f"unknown controller {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](cfg, **params)


if __name__ == "__main__":
    import sim_cluster
    os.chdir(_HERE)
    cfg = sim_cluster.load_config("cluster_config.json")
    lm = load_length_model("measured_lengths.json")
    print(f"length model: {lm.label()}")
    print(f"config calibrated={cfg.calibrated}\n")
    print("HPA semantics source:\n  " + HPA_SEMANTICS_SOURCE[:200] + "...\n")
    for n in CONTROLLER_ORDER:
        c = make(n, cfg)
        print(f"{n:<12} params={ {k: v for k, v in list(c.params.items())[:4]} } "
              f"privileged={c.uses_privileged_info}")
