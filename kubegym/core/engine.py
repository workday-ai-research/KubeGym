"""The discrete-event core: time, events, episodes, control intervals.

The engine is service-model agnostic.  It knows about arrivals, replicas, queue
storage, dispatch, billing and telemetry.  It does not know what a token is.

EVENT LOOP
----------
Time jumps to the earliest of {next arrival, next replica ready, next service
event, t_end}.  Service events fire ONLY when a replica's own service clock is
reached: an arrival must never advance service.  (In the source testbed an
earlier version serviced decode at every event time, which let a replica emit
tokens faster than its step-time model allows whenever arrivals were dense --
"superluminal decode".  The per-replica clock is the fix and is preserved here.)

Idle time at the end of a drain is not billed: `step_to(..., advance_when_idle=
False)` stops at the last event instead of coasting to the cap.  In the source
testbed, coasting billed 24 000 replica-seconds where the true figure was
2 045 -- a 12x cost error.

EPISODE LIFECYCLE
-----------------
`reset(episode, seed)` is the only way to start an episode, and it is the only
place requests come from: it calls `WorkloadSource.build(episode, seed)`, which
constructs fresh `Request` objects.  There is no API that accepts a
caller-supplied request list, because reusing request objects across episodes
silently yields a no-op episode (the engine mutates them in place).  See
`kubegym.core.workload`.

Two calls to `reset(episode, seed)` with the same arguments produce identical
episodes, including identical wall-clock-independent metrics.
`tests/test_episode.py` asserts this and asserts the metrics are non-trivial, so
a regression to the silent-no-op bug fails the suite.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Sequence

from ..provenance import ProvenancedConfig
from .replica import ReplicaPool
from .router import make_policy
from .service import INF, Request, ServiceModel
from .state import ClusterState

EPS = 1e-9

#: Consecutive non-advancing events tolerated before the loop is declared stalled.
#: A handful can legitimately occur when several events share an instant; a long
#: run of them means a service model is not advancing the clock.
STALL_LIMIT = 64


class EngineStallError(RuntimeError):
    """The event loop stopped advancing -- raised instead of spinning forever."""


#: Config fields the engine itself consumes.  A service model declares its own
#: via `ServiceModel.config_fields()`.
ENGINE_CONFIG_FIELDS = (
    "n_replicas_min", "n_replicas_max", "replica_cold_start_s", "replica_teardown_s",
    "router_policy", "control_interval_s",
)


class Simulator:
    """Discrete-event cluster with a pluggable service model.

    Control surface (the names are kept identical to the source testbed's
    backend contract so existing controllers and eval harnesses run unchanged):

        reset(episode, seed)            -> ClusterState at t=0
        set_target_replicas(k, t=None)  -> request k replicas
        step_to(t)                      -> advance to wall time t
        scrape(t=None)                  -> ClusterState
        advance_control_interval()      -> step one control interval, then scrape
        drain_all(t_cap)                -> run past the horizon until work ends
        total_replica_seconds(t=None)   -> billed GPU-seconds
        requests                        -> this episode's requests
    """

    def __init__(self, service: ServiceModel, workload: Any, cfg: ProvenancedConfig, *,
                 k0: int = 1, dispatch: Optional[str] = None,
                 allow_hypothetical_replicas: bool = False,
                 replica_bringup: Optional[str] = None,
                 bill_teardown: bool = False):
        self.service = service
        self.workload = workload
        self.cfg = cfg
        self.k0 = int(k0)

        # fail early and loudly if the config does not supply what we need
        cfg.subset(list(ENGINE_CONFIG_FIELDS) + list(service.config_fields()))

        self.control_interval_s = float(cfg.f("control_interval_s"))
        self.dispatch_name = str(dispatch if dispatch is not None else cfg.f("router_policy"))
        self.policy = make_policy(self.dispatch_name)
        self.bringup = str(replica_bringup if replica_bringup is not None
                           else cfg.get("replica_bringup", "sequential"))
        self.pool = ReplicaPool(
            service,
            cold_start_s=float(cfg.f("replica_cold_start_s")),
            teardown_s=float(cfg.f("replica_teardown_s")),
            k_min=int(cfg.f("n_replicas_min")), k_max=int(cfg.f("n_replicas_max")),
            bringup=self.bringup, allow_hypothetical=bool(allow_hypothetical_replicas),
            bill_teardown=bool(bill_teardown))

        self.t = 0.0
        self.episode = -1
        self.seed = 0
        self.trace: List[Request] = []
        self.hist: List[ClusterState] = []
        self._arrival_i = 0
        self._stalls = 0
        self._arrivals_since_scrape = 0
        self._classes_since_scrape: List[str] = []
        self._n_dispatched = 0
        self._wall_start = 0.0
        self._wall_s = 0.0
        self._truncated = False

    # ------------------------------------------------------------------
    # episode lifecycle
    # ------------------------------------------------------------------
    def reset(self, episode: int = 0, seed: int = 0, k0: Optional[int] = None) -> ClusterState:
        """Start a fresh episode.  Requests are rebuilt; nothing is reused."""
        self.episode = int(episode)
        self.seed = int(seed)
        self.trace = self.workload.build(episode=self.episode, seed=self.seed)
        if any(r.done for r in self.trace):
            raise RuntimeError(
                "workload source returned already-completed requests: episode would be a "
                "silent no-op. Build fresh Request objects per episode.")
        self.service.reset(self.seed)
        self.policy.reset(random.Random((self.seed * 8_675_309) ^ 0xD15))
        self.pool.reset(self.k0 if k0 is None else int(k0))
        self.t = 0.0
        self.pool.t = 0.0
        self.hist = []
        self._arrival_i = 0
        self._stalls = 0
        self._arrivals_since_scrape = 0
        self._classes_since_scrape = []
        self._n_dispatched = 0
        self._truncated = False
        self._wall_start = time.perf_counter()
        self._wall_s = 0.0
        st = self.scrape(0.0)
        self.hist.append(st)
        return st

    # ------------------------------------------------------------------
    # actuation
    # ------------------------------------------------------------------
    def set_target_replicas(self, k: int, t: Optional[float] = None) -> None:
        self.pool.t = self.t
        self.pool.set_target(int(k), self.t if t is None else t)

    @property
    def target(self) -> int:
        return self.pool.target

    # ------------------------------------------------------------------
    # the event loop
    # ------------------------------------------------------------------
    def step_to(self, t_end: float, advance_when_idle: bool = True) -> None:
        """Advance to wall time `t_end`.

        `advance_when_idle=False` stops at the last event rather than coasting,
        so idle replica-seconds after all work is done are not billed.
        """
        pool = self.pool
        replicas = pool.replicas
        trace = self.trace
        n_trace = len(trace)
        while self.t < t_end - EPS:
            t = self.t
            t_arr = trace[self._arrival_i].t_arrival if self._arrival_i < n_trace else INF
            # One pass for both the next-ready time and the next service event.
            # Inlined rather than routed through ReplicaPool/Replica helpers:
            # this is the hot loop, called once per simulated event.
            t_ready = INF
            t_step = INF
            n_draining = 0
            for rep in replicas:
                if rep.t_released is not None:
                    continue
                if rep.draining:
                    n_draining += 1
                if rep.t_ready > t + EPS:
                    if rep.t_ready < t_ready:
                        t_ready = rep.t_ready
                else:
                    eng = rep.engine
                    if eng.running and eng.next_step_t < t_step:
                        t_step = eng.next_step_t

            t_event = t_arr if t_arr < t_ready else t_ready
            if t_step < t_event:
                t_event = t_step
            if t_event == INF or t_event > t_end:
                if advance_when_idle:
                    pool.bill(t_end)
                    self.t = t_end
                    pool.t = t_end
                return
            if t_event <= t:
                # The event loop must always advance. A service model that
                # returns a next-event time at or before the current instant
                # (typically a floating-point residual that its completion test
                # cannot clear) would otherwise spin here forever, burning CPU
                # with no output and no error -- the worst failure mode a
                # simulator has, and one that bit this codebase once already
                # (see COMPLETION_REL_TOL in models/request_service.py). Fail
                # loudly instead, with the diagnostics needed to find the cause.
                self._stalls += 1
                if self._stalls > STALL_LIMIT:
                    raise EngineStallError(
                        f"event loop made no progress at t={t!r} for "
                        f"{self._stalls} consecutive events (t_arr={t_arr!r}, "
                        f"t_ready={t_ready!r}, t_step={t_step!r}). A service "
                        f"model is reporting a next-event time that does not "
                        f"advance the clock; check its completion tolerance "
                        f"against the magnitude of its progress accumulator.")
                t_event = math.nextafter(t, INF)
            else:
                self._stalls = 0
            pool.bill(t_event)
            self.t = t_event
            pool.t = t_event

            # arrivals due now (several may share a timestamp)
            while (self._arrival_i < len(self.trace)
                   and self.trace[self._arrival_i].t_arrival <= self.t + EPS):
                req = self.trace[self._arrival_i]
                self._arrival_i += 1
                self._arrivals_since_scrape += 1
                self._classes_since_scrape.append(
                    req.attrs.get("observed_mode", req.attrs.get("observed_cls", req.cls)))
                self._dispatch(req)

            # service events that are due, then (re)admission on every replica
            t = self.t
            for rep in replicas:
                if rep.t_released is not None or rep.t_ready > t + EPS:
                    continue
                eng = rep.engine
                if eng.running and eng.next_step_t <= t + EPS:
                    eng.service_step(t)
                eng.on_tick(t)
            # `n_draining` was counted before time advanced; a scale-down
            # decision only happens between control intervals, so it cannot
            # become nonzero inside this iteration.
            if n_draining:
                pool.reap(t)

    def _dispatch(self, req: Request) -> None:
        """Route an arrival onto some replica's waiting queue."""
        pool = self.pool
        cand = pool.accepting(self.t)
        if cand:
            tgt = self.policy.pick(cand, req)
        else:
            # No replica can accept: queue on the one that will be ready
            # soonest, which is what a real router's retry loop achieves.
            alive = [r for r in pool.replicas if r.alive and not r.draining]
            if not alive:
                alive = [r for r in pool.replicas if r.alive]
            if not alive:
                alive = [pool.start(ready_now=True)]
            tgt = min(alive, key=lambda r: (r.t_ready, r.rid))
        tgt.slot.push_back(req)
        self._n_dispatched += 1

    def advance_control_interval(self, dt: Optional[float] = None) -> ClusterState:
        """Advance one control interval and return the resulting scrape."""
        step = self.control_interval_s if dt is None else float(dt)
        self.step_to(self.t + step)
        st = self.scrape(self.t)
        self.hist.append(st)
        return st

    def drain_all(self, t_cap: float) -> bool:
        """Run past the horizon until every request finishes, or `t_cap`.

        Returns True if the drain was truncated by the cap.  Uses the same event
        loop, so post-horizon service obeys the same physics, and does not bill
        idle time.
        """
        while self.t < t_cap and self.pool.has_work():
            before = self.t
            self.step_to(t_cap, advance_when_idle=False)
            if self.t <= before + 1e-12:
                break
        self._truncated = bool(self.pool.has_work())
        self._wall_s = time.perf_counter() - self._wall_start
        return self._truncated

    # ------------------------------------------------------------------
    # telemetry
    # ------------------------------------------------------------------
    def scrape(self, t: Optional[float] = None) -> ClusterState:
        t = self.t if t is None else t
        ready = sorted(self.pool.accepting(t), key=lambda r: r.rid)
        booting = self.pool.booting(t)
        draining = self.pool.draining()
        names = list(self.service.metric_names())
        per_replica = [r.engine.metrics() for r in ready]
        metrics: Dict[str, List[float]] = {
            n: [float(m.get(n, 0.0)) for m in per_replica] for n in names}
        inflight: List[Dict[str, Any]] = []
        for rep in self.pool.replicas:
            if rep.alive:
                inflight.extend(rep.engine.inflight())
        st = ClusterState(
            t=t, metrics=metrics, n_ready=len(ready), n_booting=len(booting),
            n_draining=len(draining), target=self.pool.target,
            arrivals_since_last=self._arrivals_since_scrape,
            arrival_classes_since_last=list(self._classes_since_scrape),
            router_inflight=inflight, calibrated=self.cfg.calibrated,
            retired=dict(self.service.retired_metric_names()))
        self._arrivals_since_scrape = 0
        self._classes_since_scrape = []
        return st

    @property
    def requests(self) -> List[Request]:
        """This episode's requests.  Mutated in place; never reuse across episodes."""
        return self.trace

    def total_replica_seconds(self, t: Optional[float] = None) -> float:
        return self.pool.total_replica_seconds(t)

    # ------------------------------------------------------------------
    # convenience: run a controller over an episode
    # ------------------------------------------------------------------
    def run_controller(self, controller: Any, horizon_s: float, *,
                       drain_cap_s: Optional[float] = None,
                       episode: int = 0, seed: int = 0,
                       schedule: Optional[Dict[float, int]] = None) -> Dict[str, Any]:
        """Run one full episode under a controller (or a fixed schedule).

        `controller` must expose `decide(hist, t) -> Optional[int]`, matching the
        source testbed's controller interface; `None` means "no change".  Pass
        `schedule={t: k}` instead for a deterministic open-loop scale schedule
        (used by the parity harness so no controller code is in the loop).
        """
        self.reset(episode=episode, seed=seed)
        n_steps = int(round(horizon_s / self.control_interval_s))
        for i in range(n_steps):
            t_now = self.t
            if schedule is not None:
                if t_now in schedule:
                    self.set_target_replicas(schedule[t_now], t_now)
                else:
                    key = max((k for k in schedule if k <= t_now + EPS), default=None)
                    if key is not None and i == 0:
                        self.set_target_replicas(schedule[key], t_now)
            elif controller is not None:
                k = controller.decide(self.hist, t_now)
                if k is not None:
                    self.set_target_replicas(int(k), t_now)
            self.advance_control_interval()
        cap = (self.t + 10.0 * horizon_s) if drain_cap_s is None else drain_cap_s
        self.drain_all(cap)
        return self.episode_summary(horizon_s)

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------
    def episode_summary(self, horizon_s: Optional[float] = None) -> Dict[str, Any]:
        """Service-model-agnostic episode aggregates, provenance-stamped.

        SLO metrics, cost objectives and reward shaping deliberately live in the
        evaluation/Gym layer, not here: they are experiment choices, whereas
        these are conservation quantities the core is responsible for.
        """
        reqs = self.trace
        done = [r for r in reqs if r.done]
        lat = sorted(r.latency for r in done) if done else []
        row: Dict[str, Any] = {
            "episode": self.episode,
            "seed": self.seed,
            "service_model": self.service.name,
            "workload": getattr(self.workload, "name", "?"),
            "dispatch_policy": self.dispatch_name,
            "replica_bringup": self.bringup,
            "horizon_s": horizon_s,
            "t_end": self.t,
            "n_requests": len(reqs),
            "n_done": len(done),
            "n_dispatched": self._n_dispatched,
            "work_done": sum(r.progress for r in reqs),
            "work_demanded": sum(r.demand for r in reqs),
            "preemptions_total": sum(r.n_preemptions for r in reqs),
            "replica_seconds": self.total_replica_seconds(self.t),
            "mean_latency_s": (sum(lat) / len(lat)) if lat else None,
            "p95_latency_s": (lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None),
            "truncated_drain": self._truncated,
            "wall_s": self._wall_s or (time.perf_counter() - self._wall_start),
            "scale_events": len(self.pool.events),
            "hypothetical_replica_count_used": self.pool.hypothetical,
        }
        self.cfg.stamp(row)
        return row

    def provenance(self) -> Dict[str, Any]:
        p = self.cfg.provenance()
        p.update({
            "backend": "sim",
            "service_model": self.service.name,
            "service_model_provenance": self.service.provenance(),
            "dispatch_policy": self.dispatch_name,
            "workload": self.workload.manifest(),
            "replica_pool": self.pool.provenance(),
            "control_interval_s": self.control_interval_s,
        })
        return p
