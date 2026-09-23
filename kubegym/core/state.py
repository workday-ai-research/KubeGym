"""Cluster telemetry, metric-name guards, and observation assembly.

WHAT A CONTROLLER MAY SEE
-------------------------
`ClusterState` is one scrape of the cluster in the *deployed system's own
vocabulary*.  A controller (or an RL policy) sees exactly this and nothing else.
Per-replica metric lists cover READY replicas only, in replica-id order: a
booting replica has no metrics endpoint on a real cluster, so it contributes no
entry here either.

THE METRIC-NAME GUARD IS LOAD-BEARING
-------------------------------------
`vllm:gpu_cache_usage_perc` does not exist on vLLM 0.25.1 -- verified absent
from all 66 `vllm:*` metric families on the live server.  The correct KV metric
is `vllm:kv_cache_usage_perc`.  A controller coded against the retired name
would read KV pressure as a permanent zero: a silently broken baseline that
still produces plausible-looking numbers.  `ClusterState.get()` therefore
**raises** on a retired name instead of returning zero or a default.

`RETIRED_METRIC_NAMES` here is the always-on floor, applied whatever service
model is active, so the mistake cannot be reintroduced by a new model that
forgets to declare it.  A service model may add more via
`ServiceModel.retired_metric_names()`.  Entries are never removed to make a
controller run.

Unknown-but-not-retired names also raise, with the available names listed.
Returning 0.0 for a missing metric is exactly the failure this class exists to
prevent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:                                              # numpy is a hard dep of the package
    import numpy as _np
except Exception:                                 # pragma: no cover
    _np = None


#: Metric names that MUST NOT be readable, whatever the service model.
#: name -> reason (quoted verbatim in the exception).
RETIRED_METRIC_NAMES: Dict[str, str] = {
    "vllm:gpu_cache_usage_perc":
        "does not exist on vLLM 0.25.1 (verified absent from all 66 vllm:* families on the "
        "live server); use vllm:kv_cache_usage_perc",
    "vllm:cpu_cache_usage_perc":
        "not exposed by the testbed build; do not depend on it",
}


class RetiredMetricError(KeyError):
    """Raised when a consumer reads a metric that does not exist on the real system."""


class ClusterState:
    """A single scrape of the cluster.

    Attributes
    ----------
    t                  scrape time, seconds from episode start
    metrics            name -> list of per-ready-replica values, in rid order
    n_ready            replicas serving traffic now
    n_booting          replicas paying cold start (invisible in `metrics`)
    n_draining         replicas finishing work and still being billed
    target             the last requested replica count
    arrivals_since_last / arrival_classes_since_last
                       router-side arrival counts and the class label the router
                       itself observed (never the request's true `demand`)
    router_inflight    per-request observations the router can legitimately make
                       because it proxies the stream: tokens forwarded so far,
                       admission time, prompt size, observed class.  Contains NO
                       information about a request's true total work.
    calibrated         the claims gate of the config this run used
    """

    __slots__ = ("t", "metrics", "n_ready", "n_booting", "n_draining", "target",
                 "arrivals_since_last", "arrival_classes_since_last", "router_inflight",
                 "calibrated", "_retired", "extra")

    def __init__(self, t: float, metrics: Dict[str, List[float]], n_ready: int,
                 n_booting: int, n_draining: int, target: int,
                 arrivals_since_last: int = 0,
                 arrival_classes_since_last: Optional[List[str]] = None,
                 router_inflight: Optional[List[Dict[str, Any]]] = None,
                 calibrated: bool = False,
                 retired: Optional[Dict[str, str]] = None,
                 extra: Optional[Dict[str, Any]] = None):
        self.t = t
        self.metrics = metrics
        self.n_ready = n_ready
        self.n_booting = n_booting
        self.n_draining = n_draining
        self.target = target
        self.arrivals_since_last = arrivals_since_last
        self.arrival_classes_since_last = arrival_classes_since_last or []
        self.router_inflight = router_inflight or []
        self.calibrated = calibrated
        self._retired = dict(RETIRED_METRIC_NAMES)
        if retired:
            self._retired.update(retired)
        self.extra = extra or {}

    # -- metric access -------------------------------------------------
    def get(self, name: str) -> List[float]:
        """Per-ready-replica values for `name`.  Raises on retired/unknown names."""
        if name in self._retired:
            raise RetiredMetricError(
                f"metric {name!r} {self._retired[name]}. Reading it would silently return "
                "zero on the real server and produce a broken controller.")
        if name not in self.metrics:
            raise KeyError(f"metric {name!r} not exposed; have {sorted(self.metrics)}")
        return self.metrics[name]

    def total(self, name: str) -> float:
        return float(sum(self.get(name)))

    def mean(self, name: str) -> float:
        v = self.get(name)
        return float(sum(v) / len(v)) if v else 0.0

    def max(self, name: str) -> float:
        v = self.get(name)
        return float(max(v)) if v else 0.0

    def has(self, name: str) -> bool:
        """True if `name` is readable.  False for absent AND for retired names."""
        return name in self.metrics and name not in self._retired

    # -- convenience aggregates ---------------------------------------
    # Names follow the deployed system's vocabulary.  `llm_serving` exposes the
    # six verified vllm:* families; `request_service` exposes svc:* equivalents.
    def _first_present(self, *names: str) -> Optional[str]:
        for n in names:
            if self.has(n):
                return n
        return None

    @property
    def n_running(self) -> float:
        n = self._first_present("vllm:num_requests_running", "svc:num_requests_running")
        return self.total(n) if n else 0.0

    @property
    def n_waiting(self) -> float:
        n = self._first_present("vllm:num_requests_waiting", "svc:num_requests_waiting")
        return self.total(n) if n else 0.0

    @property
    def kv_mean(self) -> float:
        """Mean saturation of the binding per-replica resource, in [0, 1].

        KV-cache occupancy for `llm_serving`; concurrency-slot occupancy for
        `request_service`.  0.0 if the active model exposes neither.
        """
        n = self._first_present("vllm:kv_cache_usage_perc", "svc:concurrency_usage_perc")
        return self.mean(n) if n else 0.0

    @property
    def kv_max(self) -> float:
        n = self._first_present("vllm:kv_cache_usage_perc", "svc:concurrency_usage_perc")
        return self.max(n) if n else 0.0

    @property
    def preemptions_total(self) -> float:
        n = self._first_present("vllm:num_preemptions_total", "svc:num_drops_total")
        return self.total(n) if n else 0.0

    @property
    def n_replicas_effective(self) -> int:
        """What the Kubernetes HPA calls currentReplicas: every replica now
        costing money -- ready + booting + draining."""
        return self.n_ready + self.n_booting + self.n_draining

    def as_row(self) -> Dict[str, Any]:
        row = {"t": self.t, "n_ready": self.n_ready, "n_booting": self.n_booting,
               "n_draining": self.n_draining, "target": self.target,
               "n_running": self.n_running, "n_waiting": self.n_waiting,
               "kv_mean": self.kv_mean, "kv_max": self.kv_max,
               "arrivals_since_last": self.arrivals_since_last,
               "n_inflight_router": len(self.router_inflight),
               "calibrated": self.calibrated}
        for name, vals in self.metrics.items():
            if name.endswith("_total"):
                row[name.split(":")[-1]] = float(sum(vals))
        return row

    def __repr__(self) -> str:
        return (f"ClusterState(t={self.t:.2f}, ready={self.n_ready}, booting={self.n_booting}, "
                f"draining={self.n_draining}, running={self.n_running:.0f}, "
                f"waiting={self.n_waiting:.0f}, sat={self.kv_mean:.2f})")


# ---------------------------------------------------------------------------
# Observation assembly (consumed by the Gymnasium wrapper in a later phase)
# ---------------------------------------------------------------------------
#: name -> callable(ClusterState) -> float.  Every feature is derived ONLY from
#: what a `ClusterState` exposes, so an RL policy can never see more than a
#: hand-written controller.
FEATURES: Dict[str, Callable[[ClusterState], float]] = {
    "n_ready": lambda s: float(s.n_ready),
    "n_booting": lambda s: float(s.n_booting),
    "n_draining": lambda s: float(s.n_draining),
    "n_effective": lambda s: float(s.n_replicas_effective),
    "target": lambda s: float(s.target),
    "n_running": lambda s: float(s.n_running),
    "n_waiting": lambda s: float(s.n_waiting),
    "sat_mean": lambda s: float(s.kv_mean),
    "sat_max": lambda s: float(s.kv_max),
    "arrivals": lambda s: float(s.arrivals_since_last),
    "n_inflight": lambda s: float(len(s.router_inflight)),
    "running_per_ready": lambda s: float(s.n_running) / max(1.0, float(s.n_ready)),
    "waiting_per_ready": lambda s: float(s.n_waiting) / max(1.0, float(s.n_ready)),
    "preemptions_total": lambda s: float(s.preemptions_total),
}


@dataclass(frozen=True)
class ObservationSpec:
    """A named, ordered, stackable observation vector.

    `features` are keys of `FEATURES` (extend that dict to add one).  `history`
    stacks the last `history` scrapes, oldest first, zero-padded at episode
    start, so a policy can see a trend without the env keeping hidden state.

    Rate features (`arrivals`, `*_total`) are per control interval as scraped;
    normalisation is deliberately left to the wrapper, which knows the action
    space and the replica ceiling.
    """

    features: Tuple[str, ...] = ("n_ready", "n_booting", "n_draining", "n_running",
                                 "n_waiting", "sat_mean", "sat_max", "arrivals")
    history: int = 1

    def __post_init__(self) -> None:
        bad = [f for f in self.features if f not in FEATURES]
        if bad:
            raise KeyError(f"unknown observation features {bad}; have {sorted(FEATURES)}")
        if self.history < 1:
            raise ValueError("history must be >= 1")

    @property
    def dim(self) -> int:
        return len(self.features) * self.history

    def names(self) -> List[str]:
        if self.history == 1:
            return list(self.features)
        return [f"{f}[t-{k}]" for k in range(self.history - 1, -1, -1) for f in self.features]

    def build(self, hist: Sequence[ClusterState]):
        """Assemble the observation from a scrape history (oldest first)."""
        if _np is None:                                        # pragma: no cover
            raise RuntimeError("numpy is required for ObservationSpec.build")
        window = list(hist[-self.history:])
        pad = self.history - len(window)
        vec: List[float] = [0.0] * (pad * len(self.features))
        for st in window:
            vec.extend(float(FEATURES[f](st)) for f in self.features)
        return _np.asarray(vec, dtype=_np.float32)
