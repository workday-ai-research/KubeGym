"""Episode independence and determinism.

THE REGRESSION THIS FILE GUARDS
-------------------------------
In the source testbed, `load_trace()` returned mutable `Request` objects that the
simulator mutated in place.  A second episode over the same list was a silent
no-op: everything was already `done`, so wall time collapsed from 5.5 s to
0.02 s and every metric was garbage, with nothing raised.  For an RL benchmark
that calls `reset()` thousands of times this is fatal and invisible.

`test_two_consecutive_episodes_are_identical_and_nontrivial` is the direct
regression test: the metrics must be identical across two consecutive episodes
with the same seed AND must be non-trivial, so a re-introduced no-op fails on the
non-triviality assertions instead of passing on the equality ones.
"""
from __future__ import annotations

import pytest

from kubegym import RecordListSource, Simulator, WorkloadReuseError
from kubegym.core.workload import WorkloadSource
from kubegym.models.llm_serving import LLMServingModel, LLMTraceSource

HORIZON = 300.0


def _sim(llm_cfg, length_model, trace, k0=2):
    model = LLMServingModel(llm_cfg, length_model=length_model)
    source = LLMTraceSource([trace], length_model=length_model)
    return Simulator(model, source, llm_cfg, k0=k0, replica_bringup="parallel")


def _run(sim, seed, episode=0):
    sim.reset(episode=episode, seed=seed)
    for _ in range(int(HORIZON / sim.control_interval_s)):
        sim.advance_control_interval()
    sim.drain_all(20000.0)
    return sim.episode_summary(HORIZON)


def _core(row):
    return {k: row[k] for k in ("n_requests", "n_done", "work_done", "work_demanded",
                                "preemptions_total", "replica_seconds", "t_end",
                                "mean_latency_s", "p95_latency_s")}


def test_two_consecutive_episodes_are_identical_and_nontrivial(llm_cfg, length_model, trace_s1):
    sim = _sim(llm_cfg, length_model, trace_s1)
    a = _run(sim, seed=1)
    b = _run(sim, seed=1)
    assert _core(a) == _core(b), "same seed must reproduce the episode exactly"

    # non-triviality: a silent no-op episode would satisfy the equality above
    assert a["n_requests"] == 225
    assert a["n_done"] == a["n_requests"] > 0
    assert a["work_done"] == a["work_demanded"] > 10_000
    assert a["replica_seconds"] > 100.0
    assert a["t_end"] > HORIZON
    assert a["mean_latency_s"] > 0.0


def test_a_third_episode_with_a_different_seed_differs(llm_cfg, length_model, trace_s1):
    sim = _sim(llm_cfg, length_model, trace_s1)
    a = _run(sim, seed=1)
    c = _run(sim, seed=2)
    assert a["n_requests"] == c["n_requests"]
    assert a["work_demanded"] != c["work_demanded"], (
        "a different seed must draw a different work realisation")


def test_requests_are_fresh_objects_each_episode(llm_cfg, length_model, trace_s1):
    sim = _sim(llm_cfg, length_model, trace_s1)
    sim.reset(episode=0, seed=1)
    first = sim.requests
    ids_first = {id(r) for r in first}
    for _ in range(4):
        sim.advance_control_interval()
    assert any(r.progress > 0 for r in first), "first episode must actually run"
    sim.reset(episode=0, seed=1)
    second = sim.requests
    assert first is not second
    assert not ({id(r) for r in second} & ids_first) or all(
        r.progress == 0 and r.t_done is None for r in second), (
        "a rebuilt episode must start from untouched requests")
    assert all(r.progress == 0 and r.t_done is None and r.t_admit is None for r in second)


def test_workload_source_that_recycles_objects_is_rejected():
    """A source that caches and re-yields Request objects must fail loudly."""

    class Cheating(WorkloadSource):
        name = "cheating"

        def __init__(self):
            super().__init__()
            self._made = None

        def records(self, episode):
            return [{"t_arrival": 1.0, "demand": 5.0}]

        def make_request(self, rec, i, rng):
            if self._made is None:
                self._made = super().make_request(rec, i, rng)
            return self._made          # <-- the bug: same object every episode

    src = Cheating()
    src.build(episode=0, seed=0)
    with pytest.raises(WorkloadReuseError):
        src.build(episode=0, seed=0)


def test_build_is_a_pure_function_of_episode_and_seed():
    recs = [{"seq": i, "t_arrival": float(i), "demand": 3.0} for i in range(10)]
    src = RecordListSource(recs)
    a = src.build(episode=0, seed=7)
    b = src.build(episode=0, seed=7)
    assert [(r.seq, r.t_arrival, r.demand) for r in a] == [(r.seq, r.t_arrival, r.demand)
                                                           for r in b]
    assert all(x is not y for x, y in zip(a, b))


def test_reset_refuses_completed_requests():
    class Prefinished(WorkloadSource):
        name = "prefinished"

        def records(self, episode):
            return [{"t_arrival": 0.0, "demand": 1.0}]

        def make_request(self, rec, i, rng):
            r = super().make_request(rec, i, rng)
            r.t_done = 0.5              # pretend it already finished
            return r

    from kubegym import ProvenancedConfig, config_path
    from kubegym.models.request_service import RequestServiceModel
    cfg = ProvenancedConfig.load(config_path("request_service_default.json"))
    sim = Simulator(RequestServiceModel(cfg), Prefinished(), cfg, k0=1)
    with pytest.raises(RuntimeError, match="silent no-op"):
        sim.reset(episode=0, seed=0)
