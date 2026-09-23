#!/usr/bin/env python3
"""Bridge: register corpus-backed Gymnasium tasks.

WHY THIS FILE EXISTS
--------------------
The corpus (`kubegym.workloads`) and the Gym layer (`kubegym.gym`) were built
independently against `INTERFACE.md`, and neither imports the other: the Gym
layer exposes `register_workload_task` as an extension point, and the corpus
ships 767 trace specs, but nothing connected the two.  Without this module the
headline corpus is unreachable through the Gym API -- every registered task
would be one of the four short reference-trace tasks.  This module is the
join, and it is deliberately the *only* place where the two subpackages meet.

WHAT A CORPUS TASK IS
---------------------
One task per (family, split).  `episode` indexes the trace pool of that group,
so `n_episodes` is the number of traces in it.  All corpus traces are one
simulated hour and carry per-request `service_time_s`, so they drive the
`request_service` model.

DETERMINISM, AND A DOCUMENTED DEPARTURE FROM THE BUILTIN TASKS
--------------------------------------------------------------
A corpus trace's realisation is fixed by the seed recorded in its manifest
entry -- that is what makes its sha256 checkable and what
`kubegym.workloads.cli verify` re-checks for all 767 traces.  So for a corpus
task, `build(episode, seed)` is a pure function of `episode` ALONE: the env's
`seed` selects nothing and changes nothing about the offered load.

That is a real difference from the builtin tasks, where `seed` redraws the work
realisation, and it has two consequences a benchmark user must know:

  * `reset(seed=a)` and `reset(seed=b)` on the same episode give IDENTICAL
    rollouts.  This is trace replay: the trace is the datum, not a sample from
    a distribution we are free to redraw.
  * Stochastic variation across an RL training run therefore comes from episode
    selection (and from the policy), not from work resampling.  Report
    dispersion across EPISODES and across policy seeds, not across work seeds.

Controller comparisons remain paired in the sense the benchmark needs: for a
given episode every controller faces a byte-identical arrival and service
sequence.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from ..core.workload import WorkloadSource
from ..models.request_service import RequestServiceSource
from .corpus import load_manifest, source_from_entry
from .azure2019 import AzureExtract

DEFAULT_CORPUS_ENV_VAR = "KUBEGYM_CORPUS_DIR"

#: Families in the shipped corpus, in the order the paper reports them.
FAMILIES = ("azure_replay", "constant", "variable", "burst", "diurnal")
SPLITS = ("train", "dev", "test")

_TITLE = {"azure_replay": "AzureReplay", "constant": "Constant", "variable": "Variable",
          "burst": "Burst", "diurnal": "Diurnal"}


def find_corpus_dir(corpus_dir: Optional[str] = None) -> str:
    """Locate the corpus directory, or raise with an actionable message."""
    cand = corpus_dir or os.environ.get(DEFAULT_CORPUS_ENV_VAR)
    if not cand:
        raise RuntimeError(
            "corpus directory not given. Pass corpus_dir=..., or set the environment "
            f"variable {DEFAULT_CORPUS_ENV_VAR} to the directory containing "
            "corpus_manifest.json, azure2019_selection.npz and azure2019_selection.json.")
    cand = os.path.abspath(os.path.expanduser(cand))
    man = os.path.join(cand, "corpus_manifest.json")
    if not os.path.isfile(man):
        raise FileNotFoundError(f"no corpus_manifest.json under {cand!r}")
    return cand


class CorpusTraceSource(WorkloadSource):
    """A `WorkloadSource` over one (family, split) group of the corpus.

    Records are realised from each entry's SPEC -- not read from a shipped
    file -- so a task works whether or not that trace was materialised.  The
    realisation is checksummed against the manifest entry at construction of
    each episode's records, so a silent drift in the generator surfaces as a
    mismatch rather than as different numbers.
    """

    def __init__(self, entries: Sequence[Dict[str, Any]], extract: Optional[AzureExtract],
                 *, name: str, verify_digest: bool = True):
        super().__init__()          # installs the core's per-episode freshness guard
        if not entries:
            raise ValueError("CorpusTraceSource needs at least one manifest entry")
        self._entries = list(entries)
        self._extract = extract
        self._name = name
        self._verify = bool(verify_digest)
        self._cache: Dict[int, List[Dict[str, Any]]] = {}
        h = {float(e["horizon_s"]) for e in self._entries}
        if len(h) != 1:
            raise ValueError(f"{name}: mixed horizons {sorted(h)}; a task needs one horizon")
        self._horizon_s = h.pop()

    # -- WorkloadSource surface ------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    def n_episodes(self) -> int:
        return len(self._entries)

    def horizon_s(self, episode: int) -> float:
        return self._horizon_s

    def records(self, episode: int) -> Sequence[Any]:
        i = int(episode) % len(self._entries)
        if i not in self._cache:
            self._cache[i] = self._realise(i)
        return self._cache[i]

    def make_request(self, rec, i, rng):
        # Identical mapping to RequestServiceSource: `service_time_s` is used
        # verbatim, so no distribution is consulted and nothing is drawn.
        return RequestServiceSource.make_request(self, rec, i, rng)  # type: ignore[arg-type]

    def manifest(self) -> Dict[str, Any]:
        return {
            "kind": "kubegym_corpus_group",
            "name": self._name,
            "n_episodes": len(self._entries),
            "horizon_s": self._horizon_s,
            "family": self._entries[0].get("family"),
            "split": self._entries[0].get("split"),
            "trace_ids": [e["trace_id"] for e in self._entries],
            "trace_sha256": [e["sha256"] for e in self._entries],
            "digest_verified": self._verify,
            "calibrated": False,
            "is_measured_data": False,
            "note": ("Arrival times and service times are a documented RECONSTRUCTION from "
                     "per-minute counts and duration percentiles, not measured per-request "
                     "data. See WORKLOADS.md and each entry's `reconstruction` block."),
        }

    def episode_provenance(self, episode: int) -> Dict[str, Any]:
        e = self._entries[int(episode) % len(self._entries)]
        keep = ("trace_id", "family", "split", "kind", "regime", "app_hash", "day",
                "segment_start_minute", "seed", "n_requests", "sha256", "horizon_s",
                "dominant_trigger", "volume_tercile", "reconstruction")
        return {k: e[k] for k in keep if k in e}

    # -- internals -------------------------------------------------------------
    def _realise(self, i: int) -> List[Dict[str, Any]]:
        from .corpus import trace_digest
        e = self._entries[i]
        src = source_from_entry(e, self._extract)
        arr = src.arrays(0, seed=e["seed"])
        t, s = arr["t_arrival"], arr["service_time_s"]
        if self._verify:
            got = trace_digest(t, s)
            if got != e["sha256"]:
                raise RuntimeError(
                    f"corpus trace {e['trace_id']!r} does not reproduce its manifest digest "
                    f"(got {got[:16]}..., expected {e['sha256'][:16]}...). The generator or the "
                    f"extract has drifted from the manifest; do not use these numbers.")
        return [{"seq": int(k), "t_arrival": float(t[k]), "service_time_s": float(s[k]),
                 "cls": str(e.get("regime", ""))} for k in range(len(t))]


def _group(manifest: Dict[str, Any], family: str, split: str) -> List[Dict[str, Any]]:
    g = [e for e in manifest["traces"] if e.get("family") == family and e.get("split") == split]
    return sorted(g, key=lambda e: e["trace_id"])


def register_corpus_tasks(corpus_dir: Optional[str] = None, *,
                          families: Sequence[str] = FAMILIES,
                          splits: Sequence[str] = SPLITS,
                          override: bool = False,
                          verify_digest: bool = True,
                          max_episodes: Optional[int] = None) -> List[str]:
    """Register one Gym task per (family, split) of the shipped corpus.

    Returns the list of registered Gymnasium ids.  Task names are
    `Corpus-<Family>-<Split>`, e.g. `Corpus-Burst-Test`, and each is registered
    in both action modes by the Gym layer's own convention.

    `max_episodes` caps the trace pool per task (useful for smoke tests); the
    cap is recorded in the task notes so a capped run is never mistaken for the
    full pool.
    """
    from ..gym.tasks import register_workload_task

    root = find_corpus_dir(corpus_dir)
    manifest = load_manifest(os.path.join(root, "corpus_manifest.json"))

    extract = None
    npz = os.path.join(root, "azure2019_selection.npz")
    js = os.path.join(root, "azure2019_selection.json")
    if os.path.isfile(npz) and os.path.isfile(js):
        extract = AzureExtract.load(npz, js)

    ids: List[str] = []
    for family in families:
        for split in splits:
            entries = _group(manifest, family, split)
            if not entries:
                continue
            if family == "azure_replay" and extract is None:
                # Cannot realise a replay trace without the extract; skip loudly
                # rather than register a task that raises on reset().
                continue
            if max_episodes is not None:
                entries = entries[: int(max_episodes)]
            name = f"Corpus-{_TITLE.get(family, family)}-{split.capitalize()}"
            horizon = float(entries[0]["horizon_s"])
            notes = [
                f"Corpus family {family!r}, split {split!r}: {len(entries)} traces of "
                f"{horizon:.0f} s each, drawn from the shipped corpus manifest.",
                "Arrival and service times are a RECONSTRUCTION from Azure Functions 2019 "
                "per-minute counts and duration percentiles (or a parametrized generator), "
                "not measured per-request data.",
                "seed does NOT redraw the work: a corpus trace's realisation is fixed by its "
                "manifest seed so its checksum stays verifiable. reset(seed=a) and "
                "reset(seed=b) on the same episode give identical rollouts. Report dispersion "
                "across episodes and policy seeds, not across work seeds.",
                "Each episode's records are checksummed against the manifest entry on first "
                "use; a generator drift raises instead of changing the numbers."
                if verify_digest else
                "Digest verification DISABLED for this registration.",
            ]
            if max_episodes is not None:
                notes.append(f"TRACE POOL CAPPED at max_episodes={max_episodes} -- not the "
                             f"full {len(_group(manifest, family, split))}-trace pool.")

            def factory(_entries=entries, _name=name, _extract=extract):
                return CorpusTraceSource(_entries, _extract, name=_name,
                                         verify_digest=verify_digest)

            ids.extend(register_workload_task(
                name=name,
                workload_factory=factory,
                n_episodes=len(entries),
                horizon_s=horizon,
                service_model="request_service",
                workload_name=f"corpus:{family}:{split}",
                workload_description=(
                    f"{len(entries)} reconstructed 1 h traces, family {family}, split {split}. "
                    f"{manifest.get('split_rule', '')}"[:600]),
                notes=tuple(notes),
                override=override,
            ))
    if not ids:
        raise RuntimeError(f"no corpus tasks registered from {root!r}; checked families "
                           f"{list(families)} and splits {list(splits)}")
    return ids
