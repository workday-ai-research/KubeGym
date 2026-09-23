"""Parity with the source testbed's `sim_cluster.py`.

Same trace, same seed, same open-loop scale schedule through both
implementations; per-request completion times, first-token times,
replica-seconds, generated tokens and preemption counts must match EXACTLY.

`kubegym` is run with `replica_bringup="parallel"` here because that is what the
source simulator does.  The shipped default is `"sequential"` (the verified
real-cluster path); the two differ only when a scale-up step adds more than one
replica at once, and `test_replica_pool.py` covers both.

SKIPS (loudly) without `KUBEGYM_SOURCE_TESTBED` set -- see conftest.py.
`parity_report.md` records the numbers from the executed run.
"""
from __future__ import annotations

import pytest

from kubegym import Simulator
from kubegym.models.llm_serving import LLMServingModel, LLMTraceSource

HORIZON_S = 900.0
CONTROL_S = 15.0
DRAIN_CAP_S = 20_000.0

SCHEDULES = {
    # cold start, multi-step scale-up, drain, revival, scale-down under load
    "thrash": {0: 1, 4: 3, 12: 4, 24: 1, 30: 2, 40: 4, 50: 1},
    # one replica throughout: fills KV and exercises recompute preemption
    "starve": {0: 1},
}


def _run_original(mod, trace_path, seed, cfg, lm, schedule):
    trace = mod.load_trace(trace_path, lm, seed=seed)
    sim = mod.SimCluster(trace, cfg, k0=1, seed=seed)
    for i in range(int(round(HORIZON_S / CONTROL_S))):
        if i in schedule:
            sim.set_target_replicas(schedule[i], sim.t)
        sim.step_to(sim.t + CONTROL_S)
        sim.scrape(sim.t)
    sim.drain_all(DRAIN_CAP_S)
    return trace, sim.total_replica_seconds(sim.t), sim.t


def _run_kubegym(trace_path, seed, cfg, lm, schedule):
    model = LLMServingModel(cfg, length_model=lm)
    src = LLMTraceSource([trace_path], length_model=lm)
    sim = Simulator(model, src, cfg, k0=1, replica_bringup="parallel")
    sim.reset(episode=0, seed=seed)
    for i in range(int(round(HORIZON_S / CONTROL_S))):
        if i in schedule:
            sim.set_target_replicas(schedule[i], sim.t)
        sim.advance_control_interval()
    sim.drain_all(DRAIN_CAP_S)
    return sim.requests, sim.total_replica_seconds(sim.t), sim.t


@pytest.mark.parametrize("trace_name", ["S1_dev", "S3_dev"])
@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("schedule_name", sorted(SCHEDULES))
def test_bit_exact_parity(source_testbed, data_dir, llm_cfg, length_model,
                          trace_name, seed, schedule_name):
    import os

    import sim_cluster as ORIG
    from length_model import load_length_model as orig_load_lm

    trace_path = os.path.join(data_dir, f"{trace_name}.jsonl")
    orig_cfg = ORIG.load_config(os.path.join(source_testbed, "cluster_config.json"))
    schedule = SCHEDULES[schedule_name]

    a, rs_a, t_a = _run_original(ORIG, trace_path, seed, orig_cfg,
                                 orig_load_lm("measured_lengths.json"), schedule)
    b, rs_b, t_b = _run_kubegym(trace_path, seed, llm_cfg, length_model, schedule)

    assert len(a) == len(b)
    ka = {r.seq: r for r in a}
    kb = {r.seq: r for r in b}
    assert sorted(ka) == sorted(kb)

    # the drawn work realisation must be identical, or nothing else is comparable
    assert [ka[s].output_tokens_true for s in sorted(ka)] == \
           [kb[s].output_tokens_true for s in sorted(kb)]

    for s in sorted(ka):
        x, y = ka[s], kb[s]
        assert x.t_done == y.t_done, f"seq {s}: completion time differs"
        assert x.t_first_token == y.t_first_token, f"seq {s}: first-token time differs"
        assert x.generated == y.generated
        assert x.n_preemptions == y.n_preemptions
    assert sum(r.n_preemptions for r in a) == sum(r.n_preemptions for r in b)
    assert rs_a == rs_b, "replica-seconds must match exactly"
    assert t_a == t_b

    if schedule_name == "starve":
        assert sum(r.n_preemptions for r in a) > 0, (
            "the starve schedule must actually exercise preemption")


def test_step_time_model_reproduces_the_measured_throughput(llm_cfg):
    """The three fitted points, from the config's own provenance block."""
    st = llm_cfg.f("decode_step_time_s")
    fit = llm_cfg.field("decode_step_time_s").extra["fit_points"]
    for b, meas in zip(fit["batch"], fit["tok_per_s"]):
        pred = b / (st["a0_s"] + st["a1_s_per_seq"] * b)
        assert abs(pred - meas) / meas < 0.02
