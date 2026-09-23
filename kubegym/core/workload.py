"""Workload sources: fresh requests, every episode, by construction.

THE BUG THIS INTERFACE EXISTS TO PREVENT
----------------------------------------
In the testbed this package was ported from, `load_trace()` returned a list of
mutable `Request` dataclasses and the simulator mutated them **in place**.
Running a second episode over the same list silently produced a no-op: every
request was already `done`, so wall time collapsed (5.5 s -> 0.02 s on a
one-hour trace) and every metric was garbage.  Nothing raised.  For an RL
benchmark, where `reset()` is called thousands of times, that failure mode is
fatal and completely silent.

The fix is structural, not documentary:

  * A `WorkloadSource` does not own or return a request *list*.  It owns
    immutable **records** and constructs brand-new `Request` objects on every
    `build()` call.
  * `build()` is `final` in effect: subclasses implement `records()`, which must
    return immutable payloads, and the base class does the construction.
  * Every constructed request is stamped with `(source id, episode, build
    counter)`.  `build()` refuses to return an object carrying a stamp it has
    already issued, so a subclass that caches and re-yields request objects
    fails loudly on its second episode instead of producing a silent no-op.
  * `Simulator.reset()` only ever obtains requests by calling `build()`.  There
    is no API that accepts a caller-supplied request list, so the mistake cannot
    be made from outside either.

Determinism contract
--------------------
`build(episode, seed)` must be a pure function of `(episode, seed)` and the
source's immutable records: identical arguments produce requests with identical
descriptor fields in identical order.  All sampling (e.g. drawing an output
length) must come from a `random.Random` seeded from those arguments -- never
from module-level `random` or an unseeded `numpy` global.

Where sampling happens matters.  Drawing per-request work at `build()` time (not
inside the event loop) makes controller comparisons **paired**: for a given
`(episode, seed)`, every controller faces exactly the same realisation of the
workload, so controller-vs-controller contrasts carry no sampling variance from
the work distribution.  This is inherited from the testbed and preserved.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .service import Request


class WorkloadReuseError(RuntimeError):
    """A workload source tried to hand back a request object it already issued."""


class WorkloadSource:
    """Base class for every workload source.

    Subclasses implement:

        records(episode)      -> Sequence of immutable per-request payloads
        make_request(rec, i, rng) -> Request        (optional; default below)
        manifest()            -> dict describing the corpus/generator

    and MUST NOT override `build`.
    """

    #: human-readable name used in result rows
    name: str = "workload"

    def __init__(self) -> None:
        self._build_count = 0
        self._issued: set = set()

    # -- subclass surface ----------------------------------------------
    def n_episodes(self) -> Optional[int]:
        """Number of distinct episodes, or None for unbounded/parametric."""
        return None

    def records(self, episode: int) -> Sequence[Any]:
        raise NotImplementedError

    def horizon_s(self, episode: int) -> Optional[float]:
        """Intended episode horizon in seconds, if the source defines one."""
        return None

    def manifest(self) -> Dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__}

    def begin_build(self, episode: int, seed: int, rng: random.Random) -> None:
        """Optional hook: set up per-episode samplers before records are mapped.

        Called by `build()` exactly once per episode, before any
        `make_request` call.  Use it to construct additional seeded RNGs (e.g. a
        separate stream for a class-predictor error model) so that each stream is
        a pure function of `(episode, seed)`.
        """

    def make_request(self, rec: Any, i: int, rng: random.Random) -> Request:
        """Construct ONE fresh request from an immutable record.

        Default implementation accepts a mapping with keys
        `t_arrival`, `demand`, and optionally `size`, `cls`, `attrs`.
        """
        if not isinstance(rec, dict):
            raise TypeError(f"{type(self).__name__}.make_request needs a mapping record, "
                            f"got {type(rec).__name__}; override make_request")
        return Request(seq=int(rec.get("seq", i)), t_arrival=float(rec["t_arrival"]),
                       demand=float(rec["demand"]), size=float(rec.get("size", 0.0)),
                       cls=str(rec.get("cls", "")), attrs=dict(rec.get("attrs", {})))

    # -- the only way to get requests ----------------------------------
    def build(self, episode: int = 0, seed: int = 0) -> List[Request]:
        """Return a list of FRESH `Request` objects for one episode.

        Never returns objects from a previous call.  Sorted by arrival time and
        re-sequenced so `seq` is the arrival rank.
        """
        self._build_count += 1
        rng = random.Random((int(seed) * 1_000_003) ^ (int(episode) * 2_654_435_761) ^ 0x5EED)
        stamp = (id(self), int(episode), self._build_count)
        self.begin_build(int(episode), int(seed), rng)
        out: List[Request] = []
        for i, rec in enumerate(self.records(episode)):
            r = self.make_request(rec, i, rng)
            if r._origin is not None:
                raise WorkloadReuseError(
                    f"{type(self).__name__}.make_request returned a request already stamped "
                    f"{r._origin!r}. Workload sources must CONSTRUCT a new Request per episode; "
                    "reusing objects across episodes silently produces no-op episodes because "
                    "the engine mutates requests in place.")
            r._origin = stamp
            out.append(r)
        out.sort(key=lambda x: (x.t_arrival, x.seq))
        key = (int(episode), self._build_count)
        if key in self._issued:                      # defensive; cannot normally happen
            raise WorkloadReuseError(f"duplicate build stamp {key!r}")
        self._issued.add(key)
        return out

    # -- helpers -------------------------------------------------------
    def episode_provenance(self, episode: int, seed: int) -> Dict[str, Any]:
        return {"workload": self.name, "workload_class": type(self).__name__,
                "episode": int(episode), "workload_seed": int(seed)}

    def __repr__(self) -> str:
        n = self.n_episodes()
        return f"{type(self).__name__}(name={self.name!r}, n_episodes={n})"


class RecordListSource(WorkloadSource):
    """A source over an in-memory list of immutable record dicts.

    Useful for tests and for programmatic traces.  The records themselves are
    never handed out; `build()` copies each one into a new `Request`.
    """

    def __init__(self, records: Sequence[Dict[str, Any]], *, name: str = "records",
                 horizon_s: Optional[float] = None,
                 manifest: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.name = name
        self._records = [dict(r) for r in records]
        self._horizon = horizon_s
        self._manifest = dict(manifest or {})

    def n_episodes(self) -> Optional[int]:
        return 1

    def records(self, episode: int) -> Sequence[Any]:
        return self._records

    def horizon_s(self, episode: int) -> Optional[float]:
        return self._horizon

    def manifest(self) -> Dict[str, Any]:
        m = {"name": self.name, "class": type(self).__name__, "n_records": len(self._records)}
        m.update(self._manifest)
        return m


class JsonlTraceSource(WorkloadSource):
    """A source over one or more JSONL trace files, one record per line.

    Each file is one episode; `episode` indexes into the file list.  Lines are
    read once at construction and kept as immutable dicts.  Mapping a raw trace
    record onto `Request` fields is the *service model's* business, so pass a
    `record_to_request` callable (see `kubegym.models.llm_serving.LLMTraceSource`
    for the LLM-serving mapping) or subclass and override `make_request`.
    """

    def __init__(self, paths: Sequence[str], *, name: str = "jsonl",
                 record_to_request=None, horizon_s: Optional[float] = None):
        super().__init__()
        self.name = name
        self.paths = [str(p) for p in paths]
        self._record_to_request = record_to_request
        self._horizon = horizon_s
        self._cache: Dict[int, List[Dict[str, Any]]] = {}

    def n_episodes(self) -> Optional[int]:
        return len(self.paths)

    def _load(self, episode: int) -> List[Dict[str, Any]]:
        if episode not in self._cache:
            path = self.paths[episode % len(self.paths)]
            recs = []
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        recs.append(json.loads(line))
            self._cache[episode] = recs
        return self._cache[episode]

    def records(self, episode: int) -> Sequence[Any]:
        return self._load(episode)

    def horizon_s(self, episode: int) -> Optional[float]:
        return self._horizon

    def make_request(self, rec: Any, i: int, rng: random.Random) -> Request:
        if self._record_to_request is not None:
            return self._record_to_request(rec, i, rng)
        return super().make_request(rec, i, rng)

    def manifest(self) -> Dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__,
                "paths": [os.path.basename(p) for p in self.paths],
                "n_episodes": len(self.paths)}
