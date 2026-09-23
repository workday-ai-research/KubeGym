"""Analytic autoscaling controllers on the KubeGym **Gym-facing** interface.

Every controller here consumes the same 37-element observation vector an RL
policy consumes, decoded by `obsmap.ObsView`, and emits an action in the same
`Discrete` action space.  The information mapping is documented in
`obsmap.py`; the per-controller deviations from the source implementation are
documented in each class and collected in `BASELINES.md`.

PROVENANCE
----------
Ported from the source project's `controllers.py` (artifact
`090ce69d-3bcb-4e49-8691-24cace42e0bc`), classes `StaticK`, `HPA`,
`QueueConcurrency`, `LeadingIndicator`, `ForecastThreshold`, `Oracle`.  The
decision logic is copied, not re-derived; the changes are (a) reading the
observation instead of a `ClusterState`, (b) the two signal substitutions forced
by the `request_service` service model having no token stream, and (c) the
metric-target grids, which had to be re-ranged for a 4-slot replica (see
`HPAGym.TARGETS_BY_METRIC`).  `parity_port.py` verifies (a) is
decision-identical and quantifies where it is not.

The Kubernetes HPA semantics and their primary-source attribution are carried
over verbatim in `HPA_SEMANTICS_SOURCE`.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .obsmap import ObsView, Scrape

# Primary-source attribution for the HPA semantics, copied verbatim from the
# source project's controllers.py so the version a reviewer checks is the
# version that was read.  Cross-checked against the primary-source extract
# shipped as artifact d38a4dbe-68b1-4e01-9db3-dd481b0caf60.
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
class GymController:
    """Base class: `act(obs) -> action`, with `target(obs) -> replica count`.

    Subclasses implement `_target(scrapes, t)`, receiving the decoded scrape
    window (oldest first) and the episode time in seconds.  Both come from the
    observation and nothing else.
    """

    name = "base"
    #: parameters the tuner may sweep
    param_space: Dict[str, Sequence] = {}
    uses_privileged_info = False
    #: set True by controllers whose signal is an adaptation rather than a port
    is_adaptation = False

    def __init__(self, view: ObsView, cfg, action_mode: str = "absolute", **params):
        self.view = view
        self.cfg = cfg
        self.action_mode = action_mode
        self.k_min = int(cfg.f("n_replicas_min"))
        self.k_max = int(cfg.f("n_replicas_max"))
        self.dt = float(cfg.f("control_interval_s"))
        self.d = float(cfg.f("replica_cold_start_s"))
        self.params = dict(params)
        for k, v in params.items():
            setattr(self, k, v)
        self.n_saturated_steps = 0
        self.reset()

    # -- lifecycle -----------------------------------------------------
    def reset(self, env=None) -> None:
        """Clear per-episode internal state.  `env` is ignored by every
        controller except the privileged `OracleGym`."""
        self.n_saturated_steps = 0

    def clamp(self, k) -> int:
        return int(max(self.k_min, min(self.k_max, int(math.ceil(float(k) - 1e-9)))))

    # -- the decision --------------------------------------------------
    def _target(self, scrapes: List[Scrape], t: float, progress: float) -> int:
        raise NotImplementedError

    def desired_replicas(self, obs) -> int:
        """The replica target this controller wants, given the observation.

        Named `desired_replicas` and not `target` because `target` is an HPA
        *parameter* name (the metric target) and the base class binds every
        parameter as an attribute -- a method called `target` would be silently
        shadowed by `HPAGym(target=0.6)`.
        """
        scrapes, t, progress = self.view.decode(obs)
        if scrapes[-1].saturated:
            self.n_saturated_steps += 1
        return self.clamp(self._target(scrapes, t, progress))

    def act(self, obs) -> int:
        """Map the desired target into the env's action space.

        `absolute` mode is exact: action `a` requests `k_min + a`.  `delta` mode
        is rate-limited to the task's `delta_set`, so a controller that wants to
        move more than the largest delta in one interval cannot; the shortfall is
        carried by the next step because the target is read back from the
        observation.
        """
        k = self.desired_replicas(obs)
        if self.action_mode == "absolute":
            return int(k - self.k_min)
        scrapes, _, _ = self.view.decode(obs)
        cur = int(round(scrapes[-1].target))
        deltas = self._delta_set
        want = k - cur
        return int(min(range(len(deltas)), key=lambda i: (abs(deltas[i] - want), i)))

    #: set by the runner when action_mode == "delta"
    _delta_set: Tuple[int, ...] = (-2, -1, 0, 1, 2)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "params": dict(self.params),
                "uses_privileged_info": self.uses_privileged_info,
                "is_adaptation": self.is_adaptation}

    # -- shared signal helpers, computed from the decoded window -------
    def arrival_rate(self, scrapes: List[Scrape], window_s: float,
                     progress: float) -> float:
        """Arrivals per second over a trailing window -- source-exact.

        Mirrors the source project's `Controller.arrival_rate` exactly, including
        its convention that a scrape at `t' ` is *inside* the window when
        `t_now - t' <= window_s`, so a 30 s window over a 15 s cadence sums
        **three** scrapes and divides by a 30 s span.  That convention is not
        re-derived here: it is what the prior work's parameters were tuned
        against, and reproducing it is what makes the port comparable.

        Two constraints the observation imposes, both reported rather than
        absorbed:

        * The window reaches at most `(obs_history - 1) * dt` = 45 s at the
          shipped settings, because there are 4 lags.  A longer `window_s` is
          truncated, so the swept grid contains no value above 45 s.
        * Early in an episode the observation is **zero-padded**, and a padded
          lag is indistinguishable from a lag with no arrivals *unless* the
          controller counts steps.  `episode_progress` is in the observation, so
          the number of real lags is recoverable and is used -- which is what
          makes the port match the source on the first three steps instead of
          reading three fictitious zero-arrival intervals.
        """
        n_steps_taken = int(round(progress * self.view.n_control_steps))
        n_real = max(1, min(len(scrapes), n_steps_taken + 1))
        n_want = int(round(window_s / self.dt)) + 1        # source's <= convention
        n_used = max(1, min(n_real, n_want))
        used = scrapes[-n_used:]
        span = max((n_used - 1) * self.dt, self.dt)
        return float(sum(s.arrivals for s in used)) / span


# ---------------------------------------------------------------------------
# 1. STATIC-k
# ---------------------------------------------------------------------------
class StaticKGym(GymController):
    """Hold k replicas for the whole episode -- the trivial reference front.

    Swept at **every feasible k** (1..n_replicas_max), which for
    `request_service_default.json` is a 20-point front.  An adaptive controller
    is only worth its complexity if it lands strictly inside that front.

    Reads no observation at all, so its port is exact by construction.
    """

    name = "static"
    param_space = {"k": tuple(range(1, 21))}

    def __init__(self, view, cfg, k: int = 2, **kw):
        super().__init__(view, cfg, k=int(k), **kw)

    def _target(self, scrapes, t, progress) -> int:
        return self.k


# ---------------------------------------------------------------------------
# 2. HPA -- faithful Kubernetes semantics
# ---------------------------------------------------------------------------
class HPAGym(GymController):
    """Kubernetes Horizontal Pod Autoscaler semantics on the Gym observation.

    FAITHFUL to the documented algorithm (source string `HPA_SEMANTICS_SOURCE`)
    and byte-identical in decision logic to the source project's `HPA`:
      * ratio rule `desiredReplicas = ceil(currentReplicas * cur/target)` on the
        average of a per-pod metric across ready pods;
      * tolerance band: no action while `|cur/target - 1| <= tolerance` (0.1);
      * scale-DOWN stabilization window as a rolling MAXIMUM over the desired
        states computed in the trailing window (default 300 s);
      * scale-UP stabilization window default 0 s;
      * default policies: scale-up to `max(current + 4, 2 * current)` per 15 s
        with `selectPolicy: Max`; scale-down 100 %/15 s, i.e. straight to
        `minReplicas`;
      * `currentReplicas` counts pods that exist but are not yet Ready, matching
        HPA scaling a Deployment's `spec.replicas` -- here
        `n_ready + n_booting + n_draining`.

    PORT DEVIATIONS, all reported rather than absorbed:
      * The 300 s stabilization window needs 20 control intervals of memory and
        the observation carries 4.  The window is over the controller's **own
        previously computed desired states**, which is controller-internal state,
        not observation, so `_desired_hist` is kept exactly as the source keeps
        it.  This is the only place the port holds state longer than the
        observation window, and it holds no *environment* information -- only its
        own arithmetic.
      * The metric is read from the observation: `kv` -> `sat_mean` (the
        saturation of the binding per-replica resource, which is
        `svc:concurrency_usage_perc` under `request_service` and
        `vllm:kv_cache_usage_perc` under `llm_serving`); `conc` ->
        `(n_running + n_waiting) / max(n_ready, 1)`; `running` ->
        `n_running / max(n_ready, 1)`.  All three are exact.

    NOT REPRODUCED, carried over from the source project's list: the documented
    special handling of missing metrics and not-yet-ready pods (a booting replica
    exposes no metrics endpoint, so it is simply absent from the average rather
    than synthesised at 0 %/100 %); multiple metric sources; separate
    scaleUp/scaleDown tolerance fields.

    METRIC AND TARGET ARE TUNED, not assumed.  The source's `conc`/`running`
    target grid was `2..30`, ranged for an LLM replica with a large
    `max_num_seqs`.  On `request_service_default.json` a replica has
    `rs_max_concurrency = 4`, so a per-replica concurrency target of 30 is
    unreachable and the ratio rule would never fire -- HPA would silently
    degenerate to static-`k0`.  That is the same class of bug the source
    project's tuning protocol caught (a KV *fraction* target of 12.0), so the
    grid is re-ranged per service model here and the re-ranging is recorded.
    """

    name = "hpa"
    param_space = {
        "metric": ("kv", "conc", "running"),
        "tolerance": (0.1,),
        "down_stabilization_s": (60.0, 300.0),
    }
    #: per-metric target grids, per service model.  `kv` is a fraction in [0,1].
    TARGETS_BY_METRIC = {
        "request_service": {
            "kv": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            "conc": (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
            "running": (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
        },
        "llm_serving": {
            "kv": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            "conc": (2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0, 30.0),
            "running": (2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0, 30.0),
        },
    }

    @classmethod
    def configs(cls, service_model: str = "request_service") -> List[Dict[str, Any]]:
        """Explicit config list with metric-appropriate targets."""
        grids = cls.TARGETS_BY_METRIC[service_model]
        out = []
        for m in cls.param_space["metric"]:
            for tg in grids[m]:
                for tol in cls.param_space["tolerance"]:
                    for ds in cls.param_space["down_stabilization_s"]:
                        out.append({"metric": m, "target": tg, "tolerance": tol,
                                    "down_stabilization_s": ds})
        return out

    def __init__(self, view, cfg, metric: str = "kv", target: float = 0.6,
                 tolerance: float = 0.1, down_stabilization_s: float = 300.0,
                 up_stabilization_s: float = 0.0, **kw):
        super().__init__(view, cfg, metric=metric, target=float(target),
                         tolerance=float(tolerance),
                         down_stabilization_s=float(down_stabilization_s),
                         up_stabilization_s=float(up_stabilization_s), **kw)

    def reset(self, env=None) -> None:
        super().reset(env)
        self._desired_hist: List[Tuple[float, int]] = []

    def _metric_value(self, st: Scrape) -> float:
        if self.metric == "kv":
            return st.sat_mean
        n = max(st.n_ready, 1.0)
        if self.metric == "conc":
            return (st.n_running + st.n_waiting) / n
        if self.metric == "running":
            return st.n_running / n
        raise ValueError(f"unknown HPA metric {self.metric!r}")

    def _target(self, scrapes, t, progress) -> int:
        st = scrapes[-1]
        cur = max(int(round(st.n_effective)), self.k_min)
        m = self._metric_value(st)
        if self.target <= 0:
            return self.clamp(cur)
        ratio = m / self.target
        if abs(ratio - 1.0) <= self.tolerance:
            desired = cur
        else:
            desired = int(math.ceil(cur * ratio))
        desired = max(self.k_min, min(self.k_max, desired))
        self._desired_hist.append((t, desired))

        if desired > cur:
            if self.up_stabilization_s > 0:
                w = [d for (tt, d) in self._desired_hist
                     if t - tt <= self.up_stabilization_s]
                desired = min(w) if w else desired
            allowed = max(cur + 4, cur * 2)                  # selectPolicy: Max
            desired = min(desired, allowed)
        elif desired < cur:
            w = [d for (tt, d) in self._desired_hist
                 if t - tt <= self.down_stabilization_s]
            desired = max(w) if w else desired
            desired = max(desired, self.k_min)
        return self.clamp(desired)


# ---------------------------------------------------------------------------
# 3. QUEUE / CONCURRENCY target
# ---------------------------------------------------------------------------
class QueueConcurrencyGym(GymController):
    """Target a per-replica in-flight concurrency (Knative-style).

        desired = ceil((n_running + n_waiting) / target_concurrency)

    Not a multiplicative ratio on the current count, so unlike HPA it does not
    inherit a dependence on how many replicas happen to be up.  Optional
    hysteresis suppresses sub-fractional moves and `cooldown_s` blocks a
    direction reversal inside a window; both are swept, because without them
    this controller thrashes and a thrashing baseline is an unfair one.

    Port deviation: none beyond reading the observation.  Every quantity it uses
    (`n_running`, `n_waiting`, `n_effective`, `t`) is exact in the decode, and
    the cooldown timer is controller-internal state in the source too.
    """

    name = "queue"
    param_space = {
        "target_concurrency": (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0),
        "hyst": (0.0, 0.15),
        "cooldown_s": (0.0, 30.0, 60.0),
    }

    def __init__(self, view, cfg, target_concurrency: float = 2.0, hyst: float = 0.0,
                 cooldown_s: float = 0.0, **kw):
        super().__init__(view, cfg, target_concurrency=float(target_concurrency),
                         hyst=float(hyst), cooldown_s=float(cooldown_s), **kw)

    def reset(self, env=None) -> None:
        super().reset(env)
        self._last_change_t = -1e9
        self._last_dir = 0

    def _target(self, scrapes, t, progress) -> int:
        st = scrapes[-1]
        inflight = st.n_running + st.n_waiting
        cur = max(int(round(st.n_effective)), self.k_min)
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
# 4. LEADING-INDICATOR  (an ADAPTATION, not a port)
# ---------------------------------------------------------------------------
class LeadingIndicatorGym(GymController):
    """Work-velocity threshold controller: signal plus its rate of change.

    **This is an adaptation, not a faithful port, and `is_adaptation = True`.**
    The source controller drives on measured token throughput
    (`vllm:generation_tokens_total` differenced) plus its derivative.  The
    `request_service` service model has no token stream -- it exposes
    `svc:num_requests_running`, `svc:num_requests_waiting`,
    `svc:concurrency_usage_perc`, `svc:num_drops_total`,
    `svc:request_success_total`, `svc:service_seconds_total` -- and no token
    counter appears in the Gym observation for any service model.  The signal is
    therefore the observed **arrival rate**:

        signal  = lambda_hat + kappa * d(lambda_hat)/dt        [req/s]
        desired = ceil(signal / per_replica_rps)

    max'd with a backlog term over in-flight requests, because a saturated
    cluster cannot report an arrival-driven signal high enough to react to a
    queue that has already formed.

    What survives the adaptation is the structural idea the source cites
    TokenScale (arXiv 2512.03416) for: a leading work signal plus its velocity
    reacts to load that is still building.  What does not survive is any claim
    about token-level control, and nothing here is a measurement of TokenScale.
    """

    name = "leading"
    is_adaptation = True
    param_space = {
        "per_replica_rps": (1.0, 2.0, 4.0, 8.0),
        "kappa": (0.0, 5.0, 15.0, 30.0),
        "window_s": (15.0, 30.0, 45.0),
        "backlog_target": (1.0, 2.0, 4.0),
    }

    def __init__(self, view, cfg, per_replica_rps: float = 4.0, kappa: float = 15.0,
                 window_s: float = 30.0, backlog_target: float = 2.0, **kw):
        super().__init__(view, cfg, per_replica_rps=float(per_replica_rps),
                         kappa=float(kappa), window_s=float(window_s),
                         backlog_target=float(backlog_target), **kw)

    def reset(self, env=None) -> None:
        super().reset(env)
        self._prev_rate: Optional[float] = None
        self._prev_t: Optional[float] = None

    def _target(self, scrapes, t, progress) -> int:
        st = scrapes[-1]
        rate = self.arrival_rate(scrapes, self.window_s, progress)
        drate = 0.0
        if self._prev_rate is not None and self._prev_t is not None and t > self._prev_t:
            drate = (rate - self._prev_rate) / (t - self._prev_t)
        self._prev_rate, self._prev_t = rate, t
        signal = max(0.0, rate + self.kappa * drate)
        desired_rate = signal / max(self.per_replica_rps, 1e-9)
        inflight = st.n_running + st.n_waiting
        desired_backlog = inflight / max(self.backlog_target, 1e-9)
        return self.clamp(max(desired_rate, desired_backlog, float(self.k_min)))


# ---------------------------------------------------------------------------
# 5. FORECAST-THEN-THRESHOLD
# ---------------------------------------------------------------------------
class _EWMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.level: Optional[float] = None

    def update(self, x: float) -> None:
        self.level = x if self.level is None else self.alpha * x + (1 - self.alpha) * self.level

    def forecast(self, h_steps: int) -> float:
        return 0.0 if self.level is None else self.level


class _Holt:
    """Holt's linear (double-exponential) smoothing: level + trend.

    Copied from the source project's `_HoltWinters` with the seasonal term
    retained but off by default (`season_len=0`), which is the source's own
    honest default.
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


class ForecastThresholdGym(GymController):
    """Forecast the arrival process `d + H` seconds ahead, then threshold.

    Family: ADAPT / SageServe-style forecast-then-provision, as reimplemented by
    the source project -- not the original authors' code, and not a measurement
    of theirs.  It forecasts to `t + d + H` with `d` the config's
    `replica_cold_start_s`, so it is not handicapped by ignoring actuation lag.

    Three forecasters, selected on dev by the tuner:
      `ewma`   single exponential smoothing (level only)
      `holt`   Holt's linear trend
      `ridge`  a small learned autoregressive forecaster: ridge regression on
               `ar_lags` lags of the controller-visible signal, **fit on dev
               traces only** and passed in as `ar_coef`.  With no coefficients it
               falls back to Holt and sets `ridge_fallback = True`, so an
               unfitted model can never be reported as a fitted one.

    PORT DEVIATION: the source offered `signal in {"rate", "tokens"}`.  Only
    `"rate"` survives, for the reason given in `LeadingIndicatorGym`.  The
    forecast series is the arrival rate over the decoded window, so the
    forecaster's input series is one value per control interval -- the same
    cadence the source used.
    """

    name = "forecast"
    param_space = {
        "forecaster": ("ewma", "holt", "ridge"),
        "alpha": (0.2, 0.4, 0.7),
        "beta": (0.1, 0.3),
        "H_s": (0.0, 15.0, 30.0, 60.0),
        "per_replica_rps": (1.0, 2.0, 4.0, 8.0),
        "safety": (1.0, 1.2, 1.5),
        "backlog_target": (2.0, 4.0),
    }

    def __init__(self, view, cfg, forecaster: str = "holt", alpha: float = 0.4,
                 beta: float = 0.3, H_s: float = 30.0, per_replica_rps: float = 4.0,
                 safety: float = 1.2, backlog_target: float = 4.0,
                 season_len: int = 0, ar_lags: int = 4,
                 ar_coef: Optional[Sequence[float]] = None, **kw):
        self.ar_coef = list(ar_coef) if ar_coef else None
        super().__init__(view, cfg, forecaster=forecaster, alpha=float(alpha),
                         beta=float(beta), H_s=float(H_s),
                         per_replica_rps=float(per_replica_rps), safety=float(safety),
                         backlog_target=float(backlog_target),
                         season_len=int(season_len), ar_lags=int(ar_lags), **kw)
        self.ridge_fallback = (forecaster == "ridge" and not self.ar_coef)

    def reset(self, env=None) -> None:
        super().reset(env)
        self._ewma = _EWMA(getattr(self, "alpha", 0.4))
        self._holt = _Holt(getattr(self, "alpha", 0.4), getattr(self, "beta", 0.3),
                           getattr(self, "season_len", 0))
        self._series: List[float] = []

    def _target(self, scrapes, t, progress) -> int:
        st = scrapes[-1]
        x = self.arrival_rate(scrapes, max(self.dt, 30.0), progress)
        self._series.append(x)
        self._ewma.update(x)
        self._holt.update(x)
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
            for _ in range(h_steps - 1):
                xs = xs[1:] + [pred]
                pred = b0 + sum(c * v for c, v in zip(lags, reversed(xs)))
        else:
            pred = self._holt.forecast(h_steps)
        pred = max(0.0, pred) * self.safety
        desired = pred / max(self.per_replica_rps, 1e-9)
        inflight = st.n_running + st.n_waiting
        desired = max(desired, inflight / max(self.backlog_target, 1e-9))
        return self.clamp(max(desired, float(self.k_min)))

    def describe(self):
        d = super().describe()
        d["ridge_fallback"] = self.ridge_fallback
        d["ridge_fitted"] = self.ar_coef is not None
        return d


# ---------------------------------------------------------------------------
# 6. ORACLE -- privileged, a bound and never a baseline
# ---------------------------------------------------------------------------
class OracleGym(GymController):
    """Upper-bound reference that reads the trace's TRUE per-request work.

    **USES PRIVILEGED INFORMATION.**  It reads `Request.demand` (total service
    seconds) and `Request.t_arrival` for every request in the episode, including
    requests that have not arrived yet, and `Request.progress` for the residual
    work of requests in flight.  No deployable controller can observe any of
    that.  `uses_privileged_info = True` propagates onto every result row and it
    is excluded from the headline comparison and from the Pareto front.

    It is deliberately NOT an optimal-control solution: it provisions with the
    same analytic capacity model an analytic baseline would use, given exact
    knowledge of the offered work over the next `d + H` seconds.  So the gap
    between it and the best deployable controller measures the value of *knowing
    the realisation*, holding the provisioning rule fixed.

    Capacity model for `request_service`: a replica serves at most
    `rs_max_concurrency` requests concurrently with `rs_contention_slowdown = 0`,
    so it delivers `rs_max_concurrency` service-seconds per wall second.  Two
    bounds are taken:
      * rate:      `k >= work_seconds / (window_s * rs_max_concurrency)`
      * residency: `k >= n_concurrent / rs_max_concurrency`   (mode
                   `"rate+residency"` only; the queue is unbounded so residency
                   is an SLO bound, not a hard one)

    `H_s`, `safety` and `mode` are swept on dev under the same enforced budget
    as every other method -- an untuned oracle is not an upper bound on anything.
    """

    name = "oracle"
    uses_privileged_info = True
    param_space = {
        "H_s": (0.0, 15.0, 30.0, 60.0),
        "safety": (0.5, 1.0, 1.5),
        "mode": ("rate", "rate+residency"),
    }

    def __init__(self, view, cfg, H_s: float = 30.0, safety: float = 1.0,
                 mode: str = "rate+residency", **kw):
        super().__init__(view, cfg, H_s=float(H_s), safety=float(safety),
                         mode=str(mode), **kw)
        self.slots = float(cfg.f("rs_max_concurrency"))

    def reset(self, env=None) -> None:
        super().reset(env)
        self._sim = getattr(getattr(env, "unwrapped", env), "sim", None) if env is not None else None
        self._arr = None
        self._last_t = -1.0
        self._ptr = 0
        self._active: List[Any] = []
        if self._sim is not None and getattr(self._sim, "requests", None):
            reqs = self._sim.requests
            self._reqs = reqs
            self._arr = np.asarray([r.t_arrival for r in reqs], dtype=np.float64)
            self._dem = np.asarray([r.demand for r in reqs], dtype=np.float64)
            self._order = np.argsort(self._arr, kind="stable")
            self._arr_sorted = self._arr[self._order]
            self._dem_sorted = self._dem[self._order]
            self._cum = np.concatenate([[0.0], np.cumsum(self._dem_sorted)])
            # requests in arrival order, so the in-flight set can be maintained
            # incrementally instead of rescanning every request at every step:
            # the heaviest corpus trace holds 44,530 requests (burst family; the
            # per-family maxima are burst 44,530, azure_replay 37,283, variable
            # 27,261, diurnal 23,970, constant 23,927) over 240 control steps, so
            # a full rescan costs ~10.7M attribute reads per episode.
            self._by_arrival = [reqs[i] for i in self._order]
        else:
            self._reqs = []
            self._by_arrival = []

    def _target(self, scrapes, t, progress) -> int:
        if self._arr is None or not len(self._reqs):
            return self.k_min
        window = max(self.d + self.H_s, self.dt)
        t_end = t + window
        # future arrivals inside the window: exact demand, exact count
        i0 = int(np.searchsorted(self._arr_sorted, t, side="right"))
        i1 = int(np.searchsorted(self._arr_sorted, t_end, side="right"))
        future_work = float(self._cum[i1] - self._cum[i0])
        n_future = i1 - i0
        # residual work of requests already in the system, maintained
        # incrementally: admit newly arrived requests, retire completed ones.
        if t < self._last_t:                        # non-monotone call: rebuild
            self._ptr, self._active = 0, []
        self._last_t = t
        while self._ptr < len(self._by_arrival) and self._by_arrival[self._ptr].t_arrival <= t:
            self._active.append(self._by_arrival[self._ptr])
            self._ptr += 1
        residual = 0.0
        still: List[Any] = []
        for r in self._active:
            if r.t_done is None:
                residual += max(0.0, float(r.demand) - float(r.progress))
                still.append(r)
        self._active = still
        n_inflight = len(still)
        need_work = (residual + future_work) * self.safety
        k_rate = need_work / (window * max(self.slots, 1e-9))
        k = k_rate
        if self.mode == "rate+residency":
            k_res = (n_inflight + n_future) * self.safety / max(self.slots, 1e-9)
            k = max(k_rate, k_res)
        return self.clamp(max(k, float(self.k_min)))


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
REGISTRY: Dict[str, type] = {
    "static": StaticKGym,
    "hpa": HPAGym,
    "queue": QueueConcurrencyGym,
    "leading": LeadingIndicatorGym,
    "forecast": ForecastThresholdGym,
    "oracle": OracleGym,
}

#: display order used in every table and figure
CONTROLLER_ORDER = ["static", "hpa", "queue", "leading", "forecast", "oracle"]

#: the deployable set: everything that reads no privileged information
DEPLOYABLE = [n for n, c in REGISTRY.items() if not c.uses_privileged_info]


def make(name: str, view, cfg, **params) -> GymController:
    if name not in REGISTRY:
        raise KeyError(f"unknown controller {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](view, cfg, **params)
