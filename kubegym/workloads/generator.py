"""Parametrized synthetic workload generator: constant / variable / burst / diurnal.

RELATION TO THE PRIOR PROJECT'S GENERATOR
-----------------------------------------
The rate-schedule classes and the substream seeding scheme are taken from the
prior project's `workload.py` rather than reinvented: `ConstantRate`,
`DiurnalRate` and `BurstRate` keep their parameter names and their `spec()`
shape, arrival times still come from Lewis-Shedler thinning of a
non-homogeneous Poisson process, and rates are still expressed as **fractions of
single-replica capacity** so that a trace is portable across service-model
configs.  Two things are new:

1. **`VariableRate`** -- the `variable` family, which the prior source did not
   have.  It was implicit there ("some rate that moves"), which is exactly the
   kind of undefined family a benchmark should not ship, so it is defined
   explicitly below.

2. **The rate process gets its own substream.**  A stochastic rate schedule adds
   randomness to a component that used to be deterministic; giving it the
   dedicated `rate_process` stream preserves the prior generator's independence
   properties (changing the service-time distribution cannot move the arrival
   times, and vice versa).

THE `variable` FAMILY, DEFINED
------------------------------
`VariableRate` is a **log-Ornstein-Uhlenbeck (mean-reverting) rate process**,
evaluated on a piecewise-constant grid of `grid_dt_s` seconds (default 60 s, to
match the minute granularity at which the real Azure trace is recorded):

    x_0    ~ N(0, sigma^2)                                  (stationary start)
    x_{k+1} = a x_k + sigma sqrt(1 - a^2) z_k,   a = exp(-grid_dt_s / tau_s)
    lam(t) = mean_load * exp(clip(x_k, -clip_sigmas*sigma, +clip_sigmas*sigma)
                             - sigma^2 / 2)                 for t in grid cell k

Parameters and what each controls:

    mean_load    E[lam] in units of single-replica capacity.  The `- sigma^2/2`
                 term makes the mean of the *unclipped* process exactly
                 `mean_load`; clipping biases it slightly low (about -0.3% at
                 the default `clip_sigmas=3`, reported in the trace metadata).
    sigma        standard deviation of log-rate: how far the rate wanders.
                 sigma=0.3 is a +-35% band, sigma=1.0 spans an order of
                 magnitude.
    tau_s        mean-reversion time: how *fast* it wanders.  This is the
                 parameter that interacts with `replica_cold_start_s`: a rate
                 process with tau below the cold-start time cannot be tracked by
                 any reactive controller, which is the point of having the family.
    grid_dt_s    the rate is piecewise constant on this grid.
    clip_sigmas  the log-rate is clipped so `lam_max` is finite and
                 Lewis-Shedler thinning terminates.

Mean reversion (rather than a pure random walk) is chosen so the family is
**stationary**: two traces from the same spec with different seeds have the same
long-run mean load, so a controller comparison across seeds is not confounded by
one seed happening to drift into overload.  A pure random walk is available as
`tau_s = inf`, which is a documented non-stationary arm rather than the default.

UNITS AND CALIBRATION
---------------------
`load` is a fraction of single-replica capacity.  Converting to req/s needs a
capacity number, and no capacity of any real platform has been measured for this
benchmark, so `Capacity` carries `calibrated=False` and the provisional value it
used.  Every synthetic trace's metadata says so.  The provisional default is
derived from the shipped `request_service_default.json`
(`rs_max_concurrency / mean reconstructed service time`), which is itself built
from placeholder constants -- it is an arithmetic consequence of placeholders,
not a measurement.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.service import Request
from ..core.workload import WorkloadSource
from .reconstruction import invert_icdf
from .seeding import normal_from_uniform, np_substream

FAMILIES: Tuple[str, ...] = ("constant", "variable", "burst", "diurnal")


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Capacity:
    """Single-replica serving capacity in req/s.  UNCALIBRATED by default."""
    rps: float
    calibrated: bool = False
    source: str = ("PROVISIONAL, NOT MEASURED. rs_max_concurrency / mean reconstructed "
                   "service time from the shipped request_service_default.json, whose "
                   "constants are placeholders. An arithmetic consequence of placeholders "
                   "is still a placeholder.")

    def provenance(self) -> Dict[str, Any]:
        return {"single_replica_capacity_rps": self.rps, "calibrated": self.calibrated,
                "source": self.source}


#: Provisional default: rs_max_concurrency = 4 from the shipped
#: request_service_default.json, divided by the Count-weighted mean of the pooled
#: reconstructed service-time distribution of the TRAIN applications (0.900 s),
#: rounded to two significant figures.  Both inputs are placeholders, so this is
#: a labelled placeholder too.
PROVISIONAL_CAPACITY_RPS = 4.4


def default_capacity() -> Capacity:
    return Capacity(rps=PROVISIONAL_CAPACITY_RPS, calibrated=False)


# ---------------------------------------------------------------------------
# Rate schedules
# ---------------------------------------------------------------------------
class RateSchedule:
    """A (possibly stochastic) offered-rate path, in capacity fractions."""

    kind = "abstract"
    family = "abstract"

    def realise(self, horizon_s: float, seed: int, tag: str) -> "RatePath":
        raise NotImplementedError

    def spec(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class RatePath:
    """One realised rate path: piecewise-constant `lam_frac` on a time grid."""
    edges_s: np.ndarray          # grid edges, length n+1, edges[0] = 0
    lam_frac: np.ndarray         # length n, capacity fractions
    deterministic: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def lam_max_frac(self) -> float:
        return float(self.lam_frac.max()) if self.lam_frac.size else 0.0

    def at(self, t: np.ndarray) -> np.ndarray:
        i = np.clip(np.searchsorted(self.edges_s, t, side="right") - 1,
                    0, self.lam_frac.size - 1)
        return self.lam_frac[i]

    def mean_frac(self) -> float:
        w = np.diff(self.edges_s)
        return float((self.lam_frac * w).sum() / w.sum()) if w.sum() > 0 else 0.0


def _uniform_grid(horizon_s: float, dt: float) -> np.ndarray:
    n = int(math.ceil(horizon_s / dt))
    e = np.arange(n + 1, dtype=np.float64) * dt
    e[-1] = min(e[-1], horizon_s)
    return e


@dataclass(frozen=True)
class ConstantRate(RateSchedule):
    """Homogeneous Poisson at `load` x single-replica capacity."""
    load: float
    kind: str = "constant"
    family: str = "constant"

    def realise(self, horizon_s: float, seed: int, tag: str) -> RatePath:
        return RatePath(np.array([0.0, float(horizon_s)]), np.array([float(self.load)]),
                        deterministic=True)

    def spec(self) -> Dict[str, Any]:
        return {"kind": self.kind, "load": self.load}


@dataclass(frozen=True)
class DiurnalRate(RateSchedule):
    """Sinusoidal rate: `mean_load * (1 + amplitude * sin(2 pi (t/period + phase)))`."""
    mean_load: float
    amplitude: float
    period_s: float
    phase: float = 0.0
    grid_dt_s: float = 15.0
    kind: str = "diurnal"
    family: str = "diurnal"

    def realise(self, horizon_s: float, seed: int, tag: str) -> RatePath:
        e = _uniform_grid(horizon_s, self.grid_dt_s)
        mid = 0.5 * (e[:-1] + e[1:])
        lam = self.mean_load * (1.0 + self.amplitude *
                                np.sin(2 * np.pi * (mid / self.period_s + self.phase)))
        return RatePath(e, np.maximum(lam, 0.0), deterministic=True)

    def spec(self) -> Dict[str, Any]:
        return {"kind": self.kind, "mean_load": self.mean_load, "amplitude": self.amplitude,
                "period_s": self.period_s, "phase": self.phase, "grid_dt_s": self.grid_dt_s}


@dataclass(frozen=True)
class BurstRate(RateSchedule):
    """Step change: `base_load`, multiplied by `magnitude` for `duration_s` from
    `start_s`, repeating every `repeat_every_s` if set."""
    base_load: float
    magnitude: float
    start_s: float
    duration_s: float
    repeat_every_s: Optional[float] = None
    grid_dt_s: float = 15.0
    kind: str = "burst"
    family: str = "burst"

    def _in_burst(self, t: np.ndarray) -> np.ndarray:
        out = np.zeros(t.shape, dtype=bool)
        after = t >= self.start_s
        if self.repeat_every_s:
            phase = np.where(after, (t - self.start_s) % self.repeat_every_s, -1.0)
            out = after & (phase < self.duration_s)
        else:
            out = after & (t < self.start_s + self.duration_s)
        return out

    def realise(self, horizon_s: float, seed: int, tag: str) -> RatePath:
        e = _uniform_grid(horizon_s, self.grid_dt_s)
        mid = 0.5 * (e[:-1] + e[1:])
        lam = self.base_load * np.where(self._in_burst(mid), self.magnitude, 1.0)
        return RatePath(e, lam, deterministic=True)

    def spec(self) -> Dict[str, Any]:
        return {"kind": self.kind, "base_load": self.base_load, "magnitude": self.magnitude,
                "start_s": self.start_s, "duration_s": self.duration_s,
                "repeat_every_s": self.repeat_every_s, "grid_dt_s": self.grid_dt_s}


@dataclass(frozen=True)
class VariableRate(RateSchedule):
    """Log-Ornstein-Uhlenbeck rate process.  See the module docstring."""
    mean_load: float
    sigma: float
    tau_s: float
    grid_dt_s: float = 60.0
    clip_sigmas: float = 3.0
    kind: str = "variable"
    family: str = "variable"

    def realise(self, horizon_s: float, seed: int, tag: str) -> RatePath:
        e = _uniform_grid(horizon_s, self.grid_dt_s)
        n = e.size - 1
        rng = np_substream(seed, "rate_process", tag)
        z = normal_from_uniform(rng.random(n))
        a = 0.0 if not math.isfinite(self.tau_s) else math.exp(-self.grid_dt_s / self.tau_s)
        x = np.empty(n, dtype=np.float64)
        if math.isfinite(self.tau_s):
            # stationary start, then the exact AR(1) discretisation of the OU SDE
            x[0] = self.sigma * z[0]
            s = self.sigma * math.sqrt(max(0.0, 1.0 - a * a))
            for k in range(1, n):
                x[k] = a * x[k - 1] + s * z[k]
            walk = False
        else:
            # tau = inf: a pure random walk in log-rate.  NON-STATIONARY arm.
            step = self.sigma * math.sqrt(self.grid_dt_s / 3600.0)
            x = np.cumsum(step * z)
            walk = True
        lo, hi = -self.clip_sigmas * self.sigma, self.clip_sigmas * self.sigma
        xc = np.clip(x, lo, hi)
        lam = self.mean_load * np.exp(xc - 0.5 * self.sigma ** 2)
        diag = {"clipped_fraction": float((x != xc).mean()) if n else 0.0,
                "realised_mean_over_target": float(lam.mean() / self.mean_load)
                if self.mean_load > 0 else None,
                "random_walk_arm": walk,
                "ar1_coefficient": a}
        return RatePath(e, lam, deterministic=False, diagnostics=diag)

    def spec(self) -> Dict[str, Any]:
        return {"kind": self.kind, "mean_load": self.mean_load, "sigma": self.sigma,
                "tau_s": self.tau_s, "grid_dt_s": self.grid_dt_s,
                "clip_sigmas": self.clip_sigmas}


SCHEDULE_CLASSES = {"constant": ConstantRate, "diurnal": DiurnalRate,
                    "burst": BurstRate, "variable": VariableRate}


def schedule_from_spec(spec: Dict[str, Any]) -> RateSchedule:
    """Rebuild a schedule from its `spec()` dict -- the regeneration path."""
    s = dict(spec)
    kind = s.pop("kind")
    if kind not in SCHEDULE_CLASSES:
        raise ValueError(f"unknown schedule kind {kind!r}; have {sorted(SCHEDULE_CLASSES)}")
    s.pop("family", None)
    return SCHEDULE_CLASSES[kind](**s)


# ---------------------------------------------------------------------------
# Arrival sampling
# ---------------------------------------------------------------------------
def arrival_times(path: RatePath, horizon_s: float, capacity_rps: float,
                  seed: int, tag: str) -> np.ndarray:
    """Arrivals of a non-homogeneous Poisson process with intensity
    `path.at(t) * capacity_rps`, by Lewis-Shedler thinning.

    Depends only on `(path, horizon, capacity, seed, tag)`.  Draws only
    `Generator.random`, in blocks, so the result is independent of numpy version.
    """
    lam_max = path.lam_max_frac * capacity_rps
    if lam_max <= 0.0:
        return np.zeros(0, dtype=np.float64)
    rng = np_substream(seed, "arrival", tag)
    out: List[np.ndarray] = []
    t = 0.0
    # Expected candidate count plus a wide margin, drawn in one block per round.
    block = max(1024, int(1.4 * lam_max * horizon_s) + 64)
    while True:
        u = rng.random(block)
        gaps = -np.log1p(-np.clip(u, 0.0, 1.0 - 1e-15)) / lam_max
        cand = t + np.cumsum(gaps)
        t = float(cand[-1])
        keep = cand[cand < horizon_s]
        out.append(keep)
        if cand[-1] >= horizon_s:
            break
        block = max(1024, int(1.4 * lam_max * (horizon_s - t)) + 64)
    cand = np.concatenate(out) if out else np.zeros(0)
    if cand.size == 0:
        return cand
    if path.lam_frac.size == 1:
        return cand                                    # homogeneous: no thinning needed
    v = rng.random(cand.size)
    return cand[v <= path.at(cand) * capacity_rps / lam_max]


# ---------------------------------------------------------------------------
# Service-time model for synthetic families
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ServiceTimeModel:
    """How a synthetic trace draws per-request service times.

    Two options, both labelled uncalibrated:

    `azure_pooled`
        the inverse CDF of the pooled reconstructed service-time distribution of
        the selected Azure applications, supplied as `(q, v_s)` knots.  This is
        the corpus default, and it is the reason the validation section can
        attribute every generated-vs-real discrepancy to the **arrival process**:
        the service-time marginal is held fixed by construction.  It inherits
        every caveat of reconstruction assumption A2.

    `lognormal`
        the placeholder lognormal of `request_service_default.json`, offered so a
        family can be generated without the Azure extract present.
    """
    kind: str = "azure_pooled"
    q: Optional[Tuple[float, ...]] = None
    v_s: Optional[Tuple[float, ...]] = None
    mu: float = 0.0
    sigma: float = 1.0
    min_s: float = 0.001
    max_s: float = 3600.0

    def __post_init__(self) -> None:
        if self.kind not in ("azure_pooled", "lognormal"):
            raise ValueError("ServiceTimeModel.kind must be 'azure_pooled' or 'lognormal'")
        if self.kind == "azure_pooled" and not (self.q and self.v_s):
            raise ValueError("azure_pooled needs (q, v_s) knots")

    def draw(self, n: int, seed: int, tag: str) -> np.ndarray:
        rng = np_substream(seed, "service", tag)
        u = rng.random(n)
        if self.kind == "azure_pooled":
            out = invert_icdf(u, np.asarray(self.q), np.asarray(self.v_s), "linear")
        else:
            out = np.exp(self.mu + self.sigma * normal_from_uniform(u))
        return np.clip(out, self.min_s, self.max_s)

    def spec(self) -> Dict[str, Any]:
        d = {"kind": self.kind, "min_s": self.min_s, "max_s": self.max_s}
        if self.kind == "azure_pooled":
            d["q"] = list(self.q)
            d["v_s"] = list(self.v_s)
        else:
            d["mu"], d["sigma"] = self.mu, self.sigma
        return d

    def provenance(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "calibrated": False,
            "source": ("Inverse CDF of the POOLED RECONSTRUCTED service-time distribution of "
                       "the selected Azure applications. Not measured: it inherits "
                       "reconstruction assumption A2 in full (percentiles of 30-second "
                       "averages, so under-dispersed relative to the true per-request "
                       "distribution -- OPTIMISTIC). Held fixed across all synthetic families "
                       "on purpose, so that generated-vs-real differences in the validation "
                       "report are attributable to the arrival process alone.")
            if self.kind == "azure_pooled" else
                      ("PLACEHOLDER lognormal from request_service_default.json; not an "
                       "estimate of any workload."),
        }


# ---------------------------------------------------------------------------
# Trace spec + source
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SyntheticSpec:
    """Everything needed to regenerate one synthetic trace, plus its seed."""
    name: str
    family: str
    horizon_s: float
    schedule: RateSchedule
    service: ServiceTimeModel
    capacity: Capacity
    seed: int
    tests: str = ""

    def spec(self) -> Dict[str, Any]:
        return {"name": self.name, "family": self.family, "horizon_s": self.horizon_s,
                "seed": self.seed, "schedule": self.schedule.spec(),
                "service": self.service.spec(), "capacity_rps": self.capacity.rps,
                "tests": self.tests}


def synthetic_from_spec(spec: Dict[str, Any]) -> "SyntheticSource":
    """Rebuild a source from its serialised spec -- the regeneration path."""
    svc = dict(spec["service"])
    kind = svc.pop("kind")
    if kind == "azure_pooled":
        model = ServiceTimeModel(kind=kind, q=tuple(svc["q"]), v_s=tuple(svc["v_s"]),
                                 min_s=svc["min_s"], max_s=svc["max_s"])
    else:
        model = ServiceTimeModel(kind=kind, mu=svc["mu"], sigma=svc["sigma"],
                                 min_s=svc["min_s"], max_s=svc["max_s"])
    s = SyntheticSpec(name=spec["name"], family=spec["family"],
                      horizon_s=float(spec["horizon_s"]),
                      schedule=schedule_from_spec(spec["schedule"]), service=model,
                      capacity=Capacity(rps=float(spec["capacity_rps"]), calibrated=False),
                      seed=int(spec["seed"]), tests=spec.get("tests", ""))
    return SyntheticSource(s)


class SyntheticSource(WorkloadSource):
    """`WorkloadSource` for one synthetic trace spec.

    Contract compliance (INTERFACE.md section 4): `build()` is not overridden;
    `begin_build` records the seed; `records()` is a pure function of
    `(episode, seed, spec)`; service times are drawn at **build** time and passed
    through as `service_time_s`, so controller comparisons are paired.

    `episode` is an offset added to the spec's seed, so one source can serve many
    independent realisations of the same parameters (`n_episodes()` is `None`,
    i.e. unbounded/parametric).
    """

    def __init__(self, spec: SyntheticSpec):
        super().__init__()
        self.spec_obj = spec
        self.name = spec.name
        self.family = spec.family
        self._seed = spec.seed
        self._cache_key: Optional[Tuple[int, int]] = None
        self._cache: Optional[List[Dict[str, Any]]] = None
        self._last_diagnostics: Dict[str, Any] = {}

    def n_episodes(self) -> Optional[int]:
        return None

    def horizon_s(self, episode: int) -> Optional[float]:
        return self.spec_obj.horizon_s

    def begin_build(self, episode: int, seed: int, rng) -> None:
        self._seed = int(seed)
        if self._cache_key != (int(episode), int(seed)):
            self._cache = None

    def _tag(self, episode: int) -> str:
        return f"{self.spec_obj.name}|ep{int(episode)}"

    def arrays(self, episode: int = 0, seed: Optional[int] = None) -> Dict[str, Any]:
        """Numeric trace without building `Request`s (used by validation/checksums)."""
        s = self._seed if seed is None else int(seed)
        so = self.spec_obj
        tag = self._tag(episode)
        path = so.schedule.realise(so.horizon_s, s, tag)
        t = arrival_times(path, so.horizon_s, so.capacity.rps, s, tag)
        svc = so.service.draw(t.size, s, tag)
        diag = {"n_requests": int(t.size),
                "mean_rate_rps": float(t.size / so.horizon_s),
                "target_mean_rate_rps": float(path.mean_frac() * so.capacity.rps),
                "offered_load_request_seconds": float(svc.sum()),
                "rate_path": dict(path.diagnostics)}
        return {"t_arrival": t, "service_time_s": svc, "lam_frac": path.lam_frac,
                "edges_s": path.edges_s, "diagnostics": diag}

    def records(self, episode: int) -> Sequence[Any]:
        key = (int(episode), int(self._seed))
        if self._cache is not None and self._cache_key == key:
            return self._cache
        out = self.arrays(int(episode))
        self._last_diagnostics = out["diagnostics"]
        t, svc = out["t_arrival"], out["service_time_s"]
        fam = self.family
        recs = [{"seq": i, "t_arrival": float(t[i]), "service_time_s": float(svc[i]),
                 "cls": fam, "attrs": {"family": fam, "synthetic": True}}
                for i in range(t.size)]
        self._cache, self._cache_key = recs, key
        return recs

    def make_request(self, rec: Any, i: int, rng) -> Request:
        return Request(seq=int(rec["seq"]), t_arrival=float(rec["t_arrival"]),
                       demand=float(rec["service_time_s"]), size=0.0,
                       cls=str(rec["cls"]), attrs=dict(rec["attrs"]))

    def manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name, "class": type(self).__name__, "family": self.family,
            "calibrated": False, "is_measured_data": False,
            "spec": self.spec_obj.spec(),
            "rate_units": "load values are FRACTIONS of single-replica capacity; "
                          "req/s = load x single_replica_capacity_rps",
            "capacity": self.spec_obj.capacity.provenance(),
            "service_time_model": self.spec_obj.service.provenance(),
            "tests": self.spec_obj.tests,
            "last_build_diagnostics": dict(self._last_diagnostics),
        }
