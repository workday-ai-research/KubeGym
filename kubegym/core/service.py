"""The seam between the engine and the physics.

DIVISION OF LABOUR
------------------
The **engine** (`kubegym.core.engine.Simulator`) owns:
  * simulated time and the event loop,
  * the replica lifecycle: cold start, ready, drain, teardown, billing
    (`kubegym.core.replica.ReplicaPool`),
  * the per-replica *waiting queue* storage (`Slot`),
  * dispatch of arrivals to slots (`kubegym.core.router`),
  * telemetry assembly (`kubegym.core.state.ClusterState`).

The **service model** owns exactly one question: *what work does a request
represent, and how fast does a replica do it?*  It therefore owns:
  * admission: when may a queued request move from the slot queue into service,
    and what does admitting it cost,
  * service dynamics: when does the next service event fire and what does it
    advance,
  * eviction / preemption, if the model has any,
  * the metric names it exposes and their per-replica values.

A service model never sees wall-clock scaling decisions, billing, or the
controller.  The engine never knows what a "token" is.

IMPLEMENTING A SERVICE MODEL
----------------------------
Provide a `ServiceModel` (metadata + a factory) and a `ReplicaEngine` (the
per-replica physics).  The engine calls, per replica, in this order at every
event time:

    1. if `engine.next_service_time() <= t`:  `engine.service_step(t)`
    2. `engine.on_tick(t)`      # admit from the slot queue, charge blocking
                                # work, rebalance, set the next service clock

`service_step` returns the requests that finished at `t`; the engine uses the
return value only for bookkeeping (`Request.t_done` must already be set by the
model).  `on_tick` is where admission and preemption happen: the model pulls
from `slot.peek()/slot.pop()` and may return work to the queue head with
`slot.push_front()`.

Both hooks are permitted to mutate `Request` runtime fields (`t_admit`,
`t_first_token`, `progress`, `t_done`, `n_preemptions`, and the TBT
accumulators) and nothing else.  Immutable descriptor fields (`demand`, `size`,
`t_arrival`, `attrs`) must not be written: the workload source owns them.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Protocol, Sequence, runtime_checkable

INF = float("inf")


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
@dataclass
class Request:
    """One unit of offered work, in service-model-agnostic terms.

    DESCRIPTOR FIELDS (written once by the workload source, then read-only):
        seq         monotone index within the episode
        t_arrival   arrival time, seconds from episode start
        demand      total service work the request requires.  Units are the
                    service model's: output tokens for `llm_serving`, seconds of
                    service for `request_service`.
        size        admission-time footprint.  Prompt tokens for `llm_serving`;
                    0 for `request_service`.
        cls         workload class label (e.g. "short" / "long"), free text
        attrs       everything model- or corpus-specific (thinking mode, prompt
                    id, function id, ...).  Controllers see only what
                    `ClusterState` chooses to expose from here.

    RUNTIME FIELDS are written by the service model / engine during an episode.
    `demand` is the ground truth of how much work the request needs and is NOT
    observable by a controller unless it declares `uses_privileged_info`.

    MUTABILITY WARNING.  These objects are mutated in place during an episode.
    They must never be reused across episodes; that is enforced structurally by
    `kubegym.core.workload.WorkloadSource`, which constructs fresh instances per
    episode and refuses to hand back an object it has already issued.
    """

    seq: int
    t_arrival: float
    demand: float
    size: float = 0.0
    cls: str = ""
    attrs: Dict[str, Any] = field(default_factory=dict)

    # -- runtime -------------------------------------------------------
    t_admit: Optional[float] = None
    t_first_token: Optional[float] = None
    t_done: Optional[float] = None
    progress: float = 0.0
    replica: Optional[int] = None
    n_preemptions: int = 0
    max_tbt: float = 0.0
    sum_tbt: float = 0.0
    n_tbt: int = 0
    _t_last_token: Optional[float] = None
    #: stamped by the workload source; see `WorkloadSource.build`
    _origin: Optional[tuple] = None

    # -- derived (properties, matching the testbed's Request API) -------
    @property
    def done(self) -> bool:
        return self.t_done is not None

    @property
    def ttft(self) -> Optional[float]:
        """Time to first service output.  `None` until the first event."""
        return None if self.t_first_token is None else self.t_first_token - self.t_arrival

    @property
    def mean_tbt(self) -> Optional[float]:
        return None if self.n_tbt == 0 else self.sum_tbt / self.n_tbt

    @property
    def latency(self) -> Optional[float]:
        return None if self.t_done is None else self.t_done - self.t_arrival

    @property
    def queue_delay(self) -> Optional[float]:
        return None if self.t_admit is None else self.t_admit - self.t_arrival

    # -- token-model sugar --------------------------------------------
    # Valid only when the active service model measures work in tokens
    # (`llm_serving`).  Provided so ported LLM-serving code reads naturally.
    @property
    def generated(self) -> int:
        return int(self.progress)

    @property
    def prompt_tokens(self) -> int:
        return int(self.size)

    @property
    def output_tokens_true(self) -> int:
        return int(self.demand)

    @property
    def remaining(self) -> float:
        return max(0.0, self.demand - self.progress)


# ---------------------------------------------------------------------------
# Slot: the engine-owned waiting queue for one replica
# ---------------------------------------------------------------------------
class Slot(deque):
    """The waiting queue of one replica: a FIFO of requests awaiting admission.

    Storage is engine-owned; the service model is the only thing that moves
    requests out of it.  A preempting model returns its victim with
    `push_front`, which is what makes recompute-style preemption (victim keeps
    its generated prefix and goes to the HEAD of the queue) expressible without
    the core knowing what preemption means.

    It subclasses `collections.deque` on purpose: the admission check
    (`if slot: ... slot[0]`) runs once per replica per simulated event, so it
    must be a C-level operation rather than a Python method call.  Service
    models may read `slot[0]` and call `slot.popleft()` directly.

    `pop()` is deliberately disabled -- on a deque it removes from the RIGHT,
    which would silently serve the queue LIFO.  Use `take()`.
    """

    def __init__(self, rid: int):
        super().__init__()
        self.rid = rid

    def peek(self) -> Optional[Request]:
        return self[0] if self else None

    def take(self) -> Request:
        """FIFO removal: the request at the head of the queue."""
        return self.popleft()

    def pop(self):                                     # type: ignore[override]
        raise TypeError("Slot.pop() is disabled because deque.pop() removes from the RIGHT, "
                        "which would serve the queue LIFO. Use Slot.take() / popleft().")

    def push_back(self, r: Request) -> None:
        r.replica = self.rid
        self.append(r)

    def push_front(self, r: Request) -> None:
        r.replica = self.rid
        self.appendleft(r)

    def drain_into(self, other: "Slot") -> int:
        """Move everything to another slot (used if a slot is force-released)."""
        n = 0
        while self:
            other.push_back(self.popleft())
            n += 1
        return n


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class ReplicaEngine(Protocol):
    """Physics of ONE replica.  Created by `ServiceModel.make_replica`."""

    rid: int
    slot: Slot
    #: The authoritative service clock: wall time of this replica's next service
    #: event, or `inf`.  The engine's event loop reads this ATTRIBUTE directly
    #: (it is the hot path, called once per simulated event), so an
    #: implementation must keep it in sync with `next_service_time()`.
    #: `BaseReplicaEngine` does this for you.
    next_step_t: float
    #: In-service requests.  Also read directly by the event loop (hot path);
    #: `n_running` is its length.  `BaseReplicaEngine` maintains it.
    running: List[Request]

    @property
    def n_running(self) -> int:
        """Requests currently in service on this replica.

        Read in the hot loop as a guard on `next_step_t`: a replica with no
        running work must never contribute a service event, or a stale clock
        would stall the loop.
        """

    def next_service_time(self) -> float:
        """Accessor for `next_step_t`, or `inf` if none.

        Must be `inf` whenever `n_running == 0` and no timed work is pending.
        The engine takes the minimum of this over serving replicas to choose the
        next event time, so returning a time in the past will spin the loop.
        """

    def service_step(self, t: float) -> List[Request]:
        """Advance service at time `t`; return requests that finished at `t`.

        The model must set `t_done` on each returned request and remove it from
        its own running set.  It must also reschedule `next_service_time()`.
        """

    def on_tick(self, t: float) -> None:
        """Admit from `self.slot`, charge blocking work, rebalance, reschedule.

        Called on every event time after arrivals and service steps, for every
        replica that is serving (including draining ones, which finish their
        work but are refused new admissions by `accepting`).
        """

    def metrics(self) -> Dict[str, float]:
        """This replica's contribution to each exposed metric name."""

    def inflight(self) -> List[Dict[str, Any]]:
        """Per-request router-side observations (no ground-truth `demand`)."""

    def has_work(self) -> bool:
        """True while this replica still holds running or queued work."""


@runtime_checkable
class ServiceModel(Protocol):
    """Metadata plus a per-replica physics factory."""

    name: str

    def config_fields(self) -> Sequence[str]:
        """Config field names this model consumes, for provenance subsetting."""

    def metric_names(self) -> Sequence[str]:
        """Metric names a controller may read from `ClusterState`."""

    def retired_metric_names(self) -> Dict[str, str]:
        """name -> reason.  `ClusterState.get` RAISES on these.

        This is how a verified-absent metric stays impossible to depend on: a
        controller coded against a retired name fails loudly instead of reading
        a permanent zero.  Never remove an entry from this map to make a
        controller run.
        """

    def make_replica(self, rid: int, slot: Slot) -> ReplicaEngine:
        """Build the physics for one replica."""

    def reset(self, seed: int) -> None:
        """Re-seed any model-owned randomness at the start of an episode."""

    def provenance(self) -> Dict[str, Any]:
        """Model-level provenance: what is measured, what is a placeholder."""


# ---------------------------------------------------------------------------
# Base class with the boring parts done
# ---------------------------------------------------------------------------
class BaseReplicaEngine:
    """Optional base for `ReplicaEngine` implementations.

    Handles the running list, the service clock, TBT accounting and the
    accepting/draining flag.  Subclasses implement `_can_admit`,
    `_admission_cost`, `_advance` and `_service_interval`.
    """

    def __init__(self, rid: int, slot: Slot):
        self.rid = rid
        self.slot = slot
        self.running: List[Request] = []
        self.accepting = True          # cleared by the pool when draining
        self.next_step_t = INF
        self.n_preemptions = 0
        self.n_success = 0
        self.work_units = 0.0          # cumulative service work delivered

    # -- introspection -------------------------------------------------
    @property
    def n_running(self) -> int:
        return len(self.running)

    @property
    def n_waiting(self) -> int:
        return len(self.slot)

    def next_service_time(self) -> float:
        return self.next_step_t

    def has_work(self) -> bool:
        return bool(self.running) or bool(self.slot)

    # -- TBT bookkeeping shared by every model -------------------------
    def _mark_output(self, r: Request, t: float) -> None:
        """Record one increment of visible progress for `r` at time `t`."""
        if r.t_first_token is None:
            r.t_first_token = t
            r._t_last_token = t
        else:
            gap = t - (r._t_last_token if r._t_last_token is not None else t)
            r.max_tbt = max(r.max_tbt, gap)
            r.sum_tbt += gap
            r.n_tbt += 1
            r._t_last_token = t

    def inflight(self) -> List[Dict[str, Any]]:
        out = []
        for r in list(self.running) + list(self.slot):
            out.append({"seq": r.seq, "tokens_emitted": int(r.progress),
                        "t_admit": r.t_admit, "prompt_tokens": int(r.size),
                        "observed_mode": r.attrs.get("observed_mode", ""),
                        "cls_observed": r.attrs.get("observed_cls", "")})
        return out
