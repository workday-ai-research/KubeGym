"""Replica lifecycle and billing: the part of the environment that costs money.

A replica passes through BOOTING -> SERVING -> DRAINING -> RELEASED.

  BOOTING   `replica_cold_start_s` after the scale-up decision.  It accepts no
            traffic and exposes NO metrics (a booting pod has no metrics
            endpoint), so a controller cannot see its progress -- only that it
            asked for it.  It is billed from the moment it is requested.
  SERVING   accepts dispatch, runs the service model's physics.
  DRAINING  refuses new work, finishes what it holds, and KEEPS BILLING.  This
            is what makes thrash expensive and is the reason scale-down is not
            free.
  RELEASED  removed from the pool once it holds no running or queued work.

BRING-UP IS SEQUENTIAL BY DEFAULT
---------------------------------
On the source testbed, starting several vLLM engines concurrently fails KV
sizing; bringing them up one at a time is the verified path.  `ReplicaPool`
therefore serialises boots by default (`replica_bringup="sequential"`): the i-th
replica requested at time t becomes ready at
`max(t, last_pending_ready) + cold_start`.

The simulator this package was ported from booted replicas in PARALLEL (every
replica requested at t became ready at t + cold_start).  That is an infidelity
relative to the verified real-cluster path, and it makes scale-up look faster
than it is -- i.e. OPTIMISTIC.  `replica_bringup="parallel"` reproduces it
exactly and is used by the parity harness.  The two policies are identical
whenever scale-up adds one replica at a time, which is the common case for
every controller in the suite.

BILLING
-------
`replica_seconds` accrues `dt * (#booting + #serving + #draining)`.  Teardown
time is NOT billed by default, matching the source testbed, which loaded
`replica_teardown_s` but never charged it; at the measured 0.54 s (weak
evidence, n=2) this understates cost by well under 1 % of a scale-down cycle,
and the direction is OPTIMISTIC.  Set `bill_teardown=True` to charge it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from .service import INF, ReplicaEngine, ServiceModel, Slot

BOOTING, SERVING, DRAINING, RELEASED = "booting", "serving", "draining", "released"


@dataclass
class Replica:
    """A replica record.  Physics live in `engine`; the queue lives in `slot`."""

    rid: int
    t_requested: float
    t_ready: float
    slot: Slot
    engine: ReplicaEngine
    draining: bool = False
    t_released: Optional[float] = None

    @property
    def alive(self) -> bool:
        return self.t_released is None

    def phase(self, t: float) -> str:
        if self.t_released is not None:
            return RELEASED
        if self.draining:
            return DRAINING
        return SERVING if self.t_ready <= t + 1e-9 else BOOTING

    def serving(self, t: float) -> bool:
        """Ready to run its physics (draining replicas still finish work)."""
        return self.t_released is None and self.t_ready <= t + 1e-9

    def accepting(self, t: float) -> bool:
        """Eligible for new dispatch."""
        return self.serving(t) and not self.draining


class ReplicaPool:
    """Owns replica identity, lifecycle timing, and GPU-second billing."""

    def __init__(self, service: ServiceModel, *, cold_start_s: float, teardown_s: float = 0.0,
                 k_min: int = 1, k_max: int = 4, bringup: str = "sequential",
                 allow_hypothetical: bool = False, bill_teardown: bool = False):
        if bringup not in ("sequential", "parallel"):
            raise ValueError(f"replica_bringup must be 'sequential' or 'parallel', got {bringup!r}")
        self.service = service
        self.cold_start_s = float(cold_start_s)
        self.teardown_s = float(teardown_s)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.bringup = bringup
        self.allow_hypothetical = bool(allow_hypothetical)
        self.bill_teardown = bool(bill_teardown)

        self.t = 0.0
        self.replicas: List[Replica] = []
        self.retired: List[Replica] = []
        self.target = 0
        self.hypothetical = False
        self.replica_seconds = 0.0
        self.teardown_seconds = 0.0
        self.events: List[Dict[str, Any]] = []
        self._next_rid = 0
        self._t_last_bill = 0.0

    # -- construction / reset ------------------------------------------
    def reset(self, k0: int) -> None:
        self.t = 0.0
        self.replicas = []
        self.retired = []
        self.target = 0
        self.hypothetical = False
        self.replica_seconds = 0.0
        self.teardown_seconds = 0.0
        self.events = []
        self._next_rid = 0
        self._t_last_bill = 0.0
        for _ in range(max(int(k0), self.k_min)):
            self.start(ready_now=True)
        self.target = len(self.replicas)

    # -- lifecycle -----------------------------------------------------
    def start(self, ready_now: bool = False) -> Replica:
        rid = self._next_rid
        self._next_rid += 1
        if ready_now:
            t_ready = self.t
        elif self.bringup == "parallel":
            t_ready = self.t + self.cold_start_s
        else:
            pending = [r.t_ready for r in self.replicas if r.alive and r.t_ready > self.t]
            t_ready = (max(self.t, max(pending)) if pending else self.t) + self.cold_start_s
        slot = Slot(rid)
        rep = Replica(rid=rid, t_requested=self.t, t_ready=t_ready, slot=slot,
                      engine=self.service.make_replica(rid, slot))
        self.replicas.append(rep)
        return rep

    def n_alive(self) -> int:
        """Replicas not draining and not released: what the pool is trying to hold."""
        return len([r for r in self.replicas if r.alive and not r.draining])

    def set_target(self, k: int, t: Optional[float] = None) -> None:
        """Request `k` replicas.  Cold start and drain semantics apply.

        Clamped to `[k_min, k_max]`.  A request above `k_max` is clamped unless
        `allow_hypothetical`, which sets `hypothetical=True` so every result row
        can be marked as exceeding the physical testbed.
        """
        k = int(max(self.k_min, k))
        if k > self.k_max:
            if not self.allow_hypothetical:
                k = self.k_max
            else:
                self.hypothetical = True
        if k == self.target:
            return
        t = self.t if t is None else t
        old = self.target
        if k > self.target:
            for _ in range(k - self.n_alive()):
                # Reviving a draining replica is cheaper than a cold start and
                # matches a real orchestrator cancelling an in-flight scale-down.
                rev = next((r for r in self.replicas if r.alive and r.draining), None)
                if rev is not None:
                    rev.draining = False
                    rev.engine.accepting = True
                else:
                    self.start()
        else:
            # Drain the NEWEST non-draining replicas: they hold the least work.
            cand = [r for r in self.replicas if r.alive and not r.draining]
            cand.sort(key=lambda r: (-r.t_ready, -r.rid))
            for r in cand[:max(0, self.n_alive() - k)]:
                r.draining = True
                r.engine.accepting = False
        self.target = k
        self.events.append({"t": t, "from": old, "to": k})

    def reap(self, t: float) -> List[Replica]:
        """Release drained replicas that hold no work.  Returns those released."""
        out = []
        for r in list(self.replicas):
            if r.draining and not r.engine.has_work():
                r.t_released = t
                self.replicas.remove(r)
                self.retired.append(r)
                if self.bill_teardown:
                    self.teardown_seconds += self.teardown_s
                out.append(r)
        return out

    # -- billing -------------------------------------------------------
    def bill(self, t: float) -> None:
        dt = t - self._t_last_bill
        if dt <= 0:
            return
        # Every replica that exists costs, including one still booting and one
        # draining. This is what makes thrash expensive.
        self.replica_seconds += dt * len(self.replicas)
        self._t_last_bill = t

    def total_replica_seconds(self, t: Optional[float] = None) -> float:
        if t is not None:
            self.bill(t)
        return self.replica_seconds + (self.teardown_seconds if self.bill_teardown else 0.0)

    # -- views ---------------------------------------------------------
    def serving(self, t: float) -> List[Replica]:
        return [r for r in self.replicas if r.serving(t)]

    def accepting(self, t: float) -> List[Replica]:
        return [r for r in self.replicas if r.accepting(t)]

    def booting(self, t: float) -> List[Replica]:
        return [r for r in self.replicas if r.alive and r.t_ready > t + 1e-9]

    def draining(self) -> List[Replica]:
        return [r for r in self.replicas if r.alive and r.draining]

    def next_ready_time(self, t: float, eps: float = 1e-9) -> float:
        return min((r.t_ready for r in self.replicas if r.alive and r.t_ready > t + eps),
                   default=INF)

    def has_work(self) -> bool:
        return any(r.engine.has_work() for r in self.replicas)

    def __iter__(self) -> Iterator[Replica]:
        return iter(self.replicas)

    def __len__(self) -> int:
        return len(self.replicas)

    def provenance(self) -> Dict[str, Any]:
        return {"replica_bringup": self.bringup,
                "cold_start_s": self.cold_start_s,
                "teardown_s": self.teardown_s,
                "teardown_billed": self.bill_teardown,
                "k_min": self.k_min, "k_max": self.k_max,
                "hypothetical_replica_count_used": self.hypothetical}
