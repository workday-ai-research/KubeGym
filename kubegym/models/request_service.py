"""Generic request-service model for serverless / microservice traces.

STATUS: MODELLING SCAFFOLD, NOT A CALIBRATED ARTIFACT.
------------------------------------------------------
Every constant in `request_service_default.json` is a PLACEHOLDER.  None of them
was measured on any system.  They are round, obviously-synthetic numbers so that
an accidentally-unlabelled result is easy to spot, and every field's `source`
string says so.  The claims gate therefore reports `calibrated: false` for any
run using this model, and it will keep doing so until the constants are replaced
via `ProvenancedConfig.override(..., attest=...)` with a real measurement.

This model exists so that the workload corpus (Azure Functions 2019 replay, a
later phase) has a physics layer to land on, and so that the `ServiceModel` seam
is exercised by something structurally different from LLM decoding: no token
stream, no growing memory footprint, no preemption.

PHYSICAL MODEL
--------------
Per replica: at most `rs_max_concurrency` requests in service simultaneously;
the rest queue FIFO on that replica.  A request needs `demand` seconds of
service (drawn per request at episode build time from
`rs_service_time_distribution`).  With `b` requests in service, each progresses
at rate

    rate(b) = 1 / (1 + rs_contention_slowdown * (b - 1))

so `rs_contention_slowdown = 0` means the concurrency slots are independent
(ideal), and larger values model shared-CPU contention.  The default is 0.0,
which is the OPTIMISTIC end: it makes a loaded replica look as fast as an idle
one, so under-provisioning appears cheaper than it would be on real hardware.

Replica cold start is the pool's (`replica_cold_start_s`), shared with every
other service model -- a cold container and a cold vLLM engine differ only in
the constant.  There is no per-request cold start: a request dispatched to a
booting replica waits in that replica's queue, which is the "cold start
queueing" behaviour a scale-from-zero serverless platform exhibits.

There is no preemption and no drop: `svc:num_drops_total` exists and is always
0, reserved for a future admission-control variant.  Queue length is unbounded.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence

from ..core.service import INF, BaseReplicaEngine, Request, Slot
from ..core.workload import WorkloadSource
from ..provenance import ProvenancedConfig

#: Completion tolerance for accumulated service progress.
#:
#: `Request.progress` is a running SUM of delivered work, so on a long episode it
#: reaches O(1e2) work-units while float64 carries only ~1e-16 RELATIVE precision.
#: An absolute tolerance is then smaller than one ULP of the accumulator and can
#: never be satisfied: a residual survives (observed 1.68e-12 at t=16391 s), and
#: `_reschedule` turns that residual into a next-event time that rounds back onto
#: the current time, so the engine's event loop makes no progress and spins
#: forever. The bug is selective -- it only bites once the accumulator is large,
#: which is why it showed up on long Azure-replay episodes at small replica counts
#: and not in the short reference-trace tests.
#:
#: The tolerance must therefore scale with the magnitude it is compared against.
#: The relative term dominates for realistic demands; the absolute term keeps
#: near-zero demands well behaved. Completing 1e-9 of a request's work early is
#: physically irrelevant and cannot be observed through any reported metric.
COMPLETION_ABS_TOL = 1e-12
COMPLETION_REL_TOL = 1e-9


def _completion_tol(r: Request) -> float:
    """Scaled tolerance for 'this request has finished its work'."""
    return COMPLETION_ABS_TOL + COMPLETION_REL_TOL * max(abs(r.demand), abs(r.progress))


METRIC_NAMES = (
    "svc:num_requests_running",
    "svc:num_requests_waiting",
    "svc:concurrency_usage_perc",
    "svc:num_drops_total",
    "svc:service_seconds_total",
    "svc:request_success_total",
)

#: This model exposes no vLLM metrics.  The always-on floor in
#: `kubegym.core.state.RETIRED_METRIC_NAMES` still raises on the retired vLLM
#: names, and any other unknown name raises with the available list, so a
#: controller written for the LLM model fails loudly here rather than reading
#: zeros.
RETIRED_METRIC_NAMES: Dict[str, str] = {}

CONFIG_FIELDS = (
    "rs_max_concurrency", "rs_contention_slowdown", "rs_service_time_distribution",
    "rs_queue_discipline",
)


class RequestServiceEngine(BaseReplicaEngine):
    """One container/pod: a bounded-concurrency server with a FIFO queue."""

    def __init__(self, rid: int, slot: Slot, *, max_concurrency: int,
                 contention_slowdown: float):
        super().__init__(rid, slot)
        self.max_concurrency = int(max_concurrency)
        self.contention = float(contention_slowdown)
        self.service_seconds = 0.0
        self.n_drops = 0
        self._last_t: Optional[float] = None

    # -- physics -------------------------------------------------------
    def rate(self) -> float:
        b = len(self.running)
        if b <= 0:
            return 0.0
        return 1.0 / (1.0 + self.contention * (b - 1))

    def can_admit(self, r: Request) -> bool:
        return self.accepting and len(self.running) < self.max_concurrency

    @staticmethod
    def _completion_tol_for(demand: float, progress: float) -> float:
        return COMPLETION_ABS_TOL + COMPLETION_REL_TOL * max(abs(demand), abs(progress))

    def _advance(self, t: float) -> List[Request]:
        """Integrate service delivered since the last event; return completions."""
        if self._last_t is None:
            self._last_t = t
            return []
        dt = t - self._last_t
        self._last_t = t
        if dt <= 0 or not self.running:
            return []
        rate = self.rate()
        delivered = dt * rate
        finished: List[Request] = []
        for r in self.running:
            r.progress += delivered
            self.work_units += delivered
            self.service_seconds += delivered
            if r.progress >= r.demand - _completion_tol(r):
                r.t_done = t
                finished.append(r)
        for r in finished:
            self.running.remove(r)
            self.n_success += 1
        return finished

    def _reschedule(self, t: float) -> None:
        if not self.running:
            self.next_step_t = INF
            return
        rate = self.rate()
        rem = min(max(0.0, r.demand - r.progress) for r in self.running)
        if rate <= 0:
            self.next_step_t = INF
            return
        nxt = t + rem / rate
        # Time must strictly advance. Even with the scaled completion tolerance
        # above, `rem / rate` can be smaller than one ULP of `t` late in a long
        # episode, which would put the next event back on the current instant and
        # stall the event loop. Snap to the next representable instant instead: at
        # this magnitude the step is far below any reported quantity's precision,
        # and a guaranteed-monotonic clock is worth more than a nominal 0-length
        # step.
        self.next_step_t = nxt if nxt > t else math.nextafter(t, INF)

    # -- ReplicaEngine surface ----------------------------------------
    def service_step(self, t: float) -> List[Request]:
        finished = self._advance(t)
        self._reschedule(t)
        return finished

    def on_tick(self, t: float) -> None:
        self._advance(t)
        while self.slot and self.can_admit(self.slot[0]):
            r = self.slot.popleft()
            if r.t_admit is None:
                r.t_admit = t
            # No token stream here: "first output" is the moment service starts,
            # so `Request.ttft` is the queueing delay and the TBT accumulators
            # stay empty. Documented in INTERFACE.md.
            self._mark_output(r, t)
            self.running.append(r)
        self._reschedule(t)

    def metrics(self) -> Dict[str, float]:
        return {
            "svc:num_requests_running": float(len(self.running)),
            "svc:num_requests_waiting": float(len(self.slot)),
            "svc:concurrency_usage_perc": min(1.0, len(self.running) / float(self.max_concurrency)),
            "svc:num_drops_total": float(self.n_drops),
            "svc:service_seconds_total": float(self.service_seconds),
            "svc:request_success_total": float(self.n_success),
        }


class RequestServiceModel:
    """`ServiceModel` for generic request/response workloads.  All constants are
    placeholders; see the module docstring."""

    name = "request_service"

    def __init__(self, cfg: ProvenancedConfig):
        self.cfg = cfg.subset(CONFIG_FIELDS)
        self.max_concurrency = int(cfg.f("rs_max_concurrency"))
        self.contention = float(cfg.f("rs_contention_slowdown"))
        self.dist = dict(cfg.f("rs_service_time_distribution"))
        self.queue_discipline = str(cfg.f("rs_queue_discipline"))
        if self.queue_discipline != "fifo":
            raise NotImplementedError(
                f"rs_queue_discipline={self.queue_discipline!r}: only 'fifo' is implemented")

    def config_fields(self) -> Sequence[str]:
        return CONFIG_FIELDS

    def metric_names(self) -> Sequence[str]:
        return METRIC_NAMES

    def retired_metric_names(self) -> Dict[str, str]:
        return dict(RETIRED_METRIC_NAMES)

    def make_replica(self, rid: int, slot: Slot) -> RequestServiceEngine:
        return RequestServiceEngine(rid, slot, max_concurrency=self.max_concurrency,
                                    contention_slowdown=self.contention)

    def reset(self, seed: int) -> None:
        """Service times are drawn by the workload source at build time (paired
        across controllers), so the model owns no randomness."""

    def provenance(self) -> Dict[str, Any]:
        return {
            "service_model": self.name,
            "calibrated": False,
            "status": "MODELLING SCAFFOLD. Every constant is a placeholder; none was measured "
                      "on any system. No number from this model is reportable.",
            "max_concurrency": self.max_concurrency,
            "contention_slowdown": self.contention,
            "service_time_distribution": self.dist,
            "known_infidelities": [
                "rs_contention_slowdown default 0.0 makes a loaded replica as fast as an idle "
                "one: OPTIMISTIC, under-provisioning looks cheaper than on real hardware",
                "no per-request cold start (only replica-level); a warm-container platform "
                "would differ",
                "unbounded queue, no admission control, no drops or timeouts",
                "service time is independent of replica state and of the request payload",
            ],
            "config_provenance": self.cfg.provenance(),
        }


# ---------------------------------------------------------------------------
# Workload source
# ---------------------------------------------------------------------------
def sample_service_time(dist: Dict[str, Any], rng: random.Random) -> float:
    """Draw one service time, in seconds, from a labelled distribution spec.

    Supported families: `deterministic` (`value_s`), `exponential` (`mean_s`),
    `lognormal` (`mu`, `sigma` on the natural-log scale of seconds), `empirical`
    (`samples_s`, bootstrapped).  `min_s`/`max_s` clamp if present.
    """
    fam = str(dist.get("family", "")).lower()
    if fam == "deterministic":
        v = float(dist["value_s"])
    elif fam == "exponential":
        v = rng.expovariate(1.0 / float(dist["mean_s"]))
    elif fam == "lognormal":
        v = math.exp(rng.gauss(float(dist["mu"]), float(dist["sigma"])))
    elif fam == "empirical":
        s = dist["samples_s"]
        if not s:
            raise ValueError("empirical service-time distribution has no samples_s")
        v = float(rng.choice(s))
    else:
        raise ValueError(f"unknown service-time family {fam!r}; supported: deterministic, "
                         "exponential, lognormal, empirical")
    v = max(v, float(dist.get("min_s", 1e-6)))
    if "max_s" in dist:
        v = min(v, float(dist["max_s"]))
    return v


class RequestServiceSource(WorkloadSource):
    """Arrival records + a service-time distribution -> fresh requests.

    Records need only `t_arrival` (plus optional `seq`, `cls`, `attrs`); the
    service time is DRAWN at episode build time so controller comparisons are
    paired.  If a record already carries `service_time_s` (e.g. a replay trace
    with measured durations) that value is used verbatim and the distribution is
    not consulted -- which is how the Azure Functions replay will attach real
    durations without touching this class.
    """

    def __init__(self, records: Sequence[Dict[str, Any]], *,
                 service_time_distribution: Dict[str, Any],
                 name: str = "request_service_trace",
                 horizon_s: Optional[float] = None,
                 manifest: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.name = name
        self._records = [dict(r) for r in records]
        self.dist = dict(service_time_distribution)
        self._horizon = horizon_s
        self._manifest = dict(manifest or {})

    def n_episodes(self) -> Optional[int]:
        return 1

    def records(self, episode: int) -> Sequence[Any]:
        return self._records

    def horizon_s(self, episode: int) -> Optional[float]:
        return self._horizon

    def make_request(self, rec: Any, i: int, rng: random.Random) -> Request:
        if "service_time_s" in rec:
            demand = float(rec["service_time_s"])
            drawn = False
        else:
            demand = sample_service_time(self.dist, rng)
            drawn = True
        attrs = dict(rec.get("attrs", {}))
        attrs.setdefault("observed_cls", rec.get("cls", ""))
        attrs["service_time_drawn"] = drawn
        return Request(seq=int(rec.get("seq", i)), t_arrival=float(rec["t_arrival"]),
                       demand=demand, size=0.0, cls=str(rec.get("cls", "")), attrs=attrs)

    def manifest(self) -> Dict[str, Any]:
        m = {"name": self.name, "class": type(self).__name__,
             "n_records": len(self._records),
             "service_time_distribution": self.dist,
             "calibrated": False,
             "note": "service times are PLACEHOLDER draws unless the records carry a measured "
                     "service_time_s"}
        m.update(self._manifest)
        return m


def poisson_arrivals(rate_rps: float, horizon_s: float, seed: int = 0,
                     cls: str = "generic") -> List[Dict[str, Any]]:
    """Homogeneous-Poisson arrival records, for tests and smoke runs.

    NOT part of the workload corpus (that is a later phase and lands as its own
    provenance-tracked artifact).  A constant-rate Poisson process is a
    deliberately uninteresting reference, not a model of any real trace.
    """
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    t = 0.0
    i = 0
    while True:
        t += rng.expovariate(rate_rps)
        if t > horizon_s:
            break
        out.append({"seq": i, "t_arrival": t, "cls": cls})
        i += 1
    return out
