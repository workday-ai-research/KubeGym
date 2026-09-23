"""Dispatch policies: which ready replica does an arriving request go to.

The router is a *separate* component from the autoscaling controller and is not
what this benchmark evaluates -- but it is a confound, so it is explicit,
named, and recorded in provenance rather than hard-coded.

Policies see only what a real router sees: each candidate replica's outstanding
work (running + queued) and its last-scraped saturation.  They never see a
request's true `demand`.

Registered policies
-------------------
`least_running`
    Fewest running+queued requests, ties by replica id.  This is what the
    ported simulator used.  Marked as a PLACEHOLDER for the deployed router's
    policy in `llm_serving_l40s.json`.
`least_outstanding_kv_rr`
    Fewest outstanding, then lowest saturation, then round-robin.  This matches
    the dispatch rule in the deployed router of the source project
    (`router.py:Router.pick`), where `outstanding` is the router's own in-flight
    count and `kv_usage` is the last sampled KV fraction.  Use this when the
    router policy must match that deployment.
`round_robin`
    Ignores load; useful as a deliberately weak reference.
`random`
    Uniform over candidates, from the episode's seeded RNG.

Adding one: write `pick(candidates, req, rng) -> Replica` and call
`register_policy(name, factory)`.  `candidates` is a non-empty list of
accepting replicas (never booting, never draining).
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Protocol

POLICY_NAMES = ("least_running", "least_outstanding_kv_rr", "round_robin", "random")


class DispatchPolicy(Protocol):
    name: str

    def reset(self, rng: random.Random) -> None: ...

    def pick(self, candidates: List[Any], req: Any) -> Any:
        """Choose one replica from a non-empty candidate list."""


class LeastRunning:
    """Fewest running+waiting, ties by replica id (the ported simulator's rule)."""

    name = "least_running"

    def reset(self, rng: random.Random) -> None:
        pass

    def pick(self, candidates: List[Any], req: Any) -> Any:
        return min(candidates,
                   key=lambda r: (r.engine.n_running + r.engine.n_waiting, r.rid))


class LeastOutstandingKvRoundRobin:
    """Fewest outstanding, then lowest saturation, then round-robin.

    Mirrors the deployed router's `pick()` in the source project.  Saturation is
    read from the replica's current metrics, which on a real router is the last
    *sampled* value; the simulator supplies it with zero staleness, which is
    OPTIMISTIC about the router's information (see INTERFACE.md).
    """

    name = "least_outstanding_kv_rr"

    def __init__(self) -> None:
        self._rr = 0

    def reset(self, rng: random.Random) -> None:
        self._rr = 0

    def pick(self, candidates: List[Any], req: Any) -> Any:
        def key(r):
            m = r.engine.metrics()
            sat = 0.0
            for k, v in m.items():
                if k.endswith("usage_perc"):
                    sat = float(v)
                    break
            return (r.engine.n_running + r.engine.n_waiting, sat, (r.rid - self._rr) % 1000)

        best = min(candidates, key=key)
        self._rr = (best.rid + 1) % 1000
        return best


class RoundRobin:
    name = "round_robin"

    def __init__(self) -> None:
        self._i = 0

    def reset(self, rng: random.Random) -> None:
        self._i = 0

    def pick(self, candidates: List[Any], req: Any) -> Any:
        c = sorted(candidates, key=lambda r: r.rid)
        out = c[self._i % len(c)]
        self._i += 1
        return out


class RandomDispatch:
    name = "random"

    def __init__(self) -> None:
        self._rng = random.Random(0)

    def reset(self, rng: random.Random) -> None:
        self._rng = random.Random(rng.randrange(1 << 30))

    def pick(self, candidates: List[Any], req: Any) -> Any:
        return self._rng.choice(sorted(candidates, key=lambda r: r.rid))


_REGISTRY: Dict[str, Callable[[], Any]] = {
    "least_running": LeastRunning,
    "least_outstanding_kv_rr": LeastOutstandingKvRoundRobin,
    "round_robin": RoundRobin,
    "random": RandomDispatch,
}


def register_policy(name: str, factory: Callable[[], Any]) -> None:
    if name in _REGISTRY:
        raise KeyError(f"dispatch policy {name!r} already registered")
    _REGISTRY[name] = factory


def make_policy(name: str) -> Any:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown dispatch policy {name!r}; have {sorted(_REGISTRY)}") from None
