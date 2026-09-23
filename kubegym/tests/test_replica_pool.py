"""Replica lifecycle: cold start, sequential bring-up, drain billing, clamps."""
from __future__ import annotations

import pytest

from kubegym import Simulator
from kubegym.models.llm_serving import LLMServingModel, LLMTraceSource

COLD = 31.9


def _sim(llm_cfg, length_model, trace, *, k0=2, bringup="sequential"):
    model = LLMServingModel(llm_cfg, length_model=length_model)
    src = LLMTraceSource([trace], length_model=length_model)
    sim = Simulator(model, src, llm_cfg, k0=k0, replica_bringup=bringup)
    sim.reset(episode=0, seed=1)
    return sim


def test_cold_start_delays_readiness(llm_cfg, length_model, trace_s1):
    sim = _sim(llm_cfg, length_model, trace_s1)
    sim.step_to(10.0)
    sim.set_target_replicas(3)
    sim.step_to(10.0 + COLD - 1.0)
    assert sim.scrape().n_ready == 2
    sim.step_to(10.0 + COLD + 1.0)
    assert sim.scrape().n_ready == 3


def test_sequential_bringup_serialises_boots(llm_cfg, length_model, trace_s1):
    """Concurrent vLLM engine init fails KV sizing on the testbed; one at a time
    is the verified path, so two added replicas must not become ready together."""
    sim = _sim(llm_cfg, length_model, trace_s1, k0=1, bringup="sequential")
    sim.set_target_replicas(3)                     # +2 replicas at t=0
    t_ready = sorted(r.t_ready for r in sim.pool.replicas)
    assert t_ready == pytest.approx([0.0, COLD, 2 * COLD])
    sim.step_to(COLD + 1.0)
    assert sim.scrape().n_ready == 2, "only the first boot has completed"
    sim.step_to(2 * COLD + 1.0)
    assert sim.scrape().n_ready == 3


def test_parallel_bringup_reproduces_the_source_simulator(llm_cfg, length_model, trace_s1):
    sim = _sim(llm_cfg, length_model, trace_s1, k0=1, bringup="parallel")
    sim.set_target_replicas(3)
    t_ready = sorted(r.t_ready for r in sim.pool.replicas)
    assert t_ready == pytest.approx([0.0, COLD, COLD])


def test_single_replica_scale_up_is_identical_under_both_policies(llm_cfg, length_model,
                                                                 trace_s1):
    """The policies differ only for multi-replica steps, which is why parity with
    the source simulator is unaffected on one-at-a-time controllers."""
    out = []
    for bringup in ("sequential", "parallel"):
        sim = _sim(llm_cfg, length_model, trace_s1, k0=1, bringup=bringup)
        sim.set_target_replicas(2)
        out.append(sorted(r.t_ready for r in sim.pool.replicas))
    assert out[0] == pytest.approx(out[1])


def test_draining_replicas_keep_billing(llm_cfg, length_model, trace_s1):
    sim = _sim(llm_cfg, length_model, trace_s1, k0=4)
    sim.step_to(50.0)
    a = sim.total_replica_seconds(50.0)
    sim.set_target_replicas(1)
    sim.step_to(60.0)
    b = sim.total_replica_seconds(60.0)
    assert b - a > 10.0, "a draining replica still costs GPU-seconds"
    st = sim.scrape()
    assert st.n_draining > 0
    assert st.n_replicas_effective > st.n_ready


def test_booting_replicas_are_billed(llm_cfg, length_model, trace_s1):
    sim = _sim(llm_cfg, length_model, trace_s1, k0=1)
    sim.step_to(10.0)
    a = sim.total_replica_seconds(10.0)
    sim.set_target_replicas(3)
    sim.step_to(20.0)
    b = sim.total_replica_seconds(20.0)
    assert b - a == pytest.approx(30.0, abs=1e-6), (
        "3 replicas exist during boot, so 10 s costs 30 replica-seconds")


def test_target_is_clamped_to_the_measured_ceiling(llm_cfg, length_model, trace_s1):
    sim = _sim(llm_cfg, length_model, trace_s1, k0=1)
    sim.set_target_replicas(99)
    assert sim.target == 4 == llm_cfg.f("n_replicas_max")
    assert sim.pool.hypothetical is False
    sim.set_target_replicas(0)
    assert sim.target == 1 == llm_cfg.f("n_replicas_min")


def test_hypothetical_replica_counts_are_flagged(llm_cfg, length_model, trace_s1):
    model = LLMServingModel(llm_cfg, length_model=length_model)
    src = LLMTraceSource([trace_s1], length_model=length_model)
    sim = Simulator(model, src, llm_cfg, k0=1, allow_hypothetical_replicas=True)
    sim.reset(episode=0, seed=1)
    sim.set_target_replicas(8)
    assert sim.target == 8
    assert sim.pool.hypothetical is True
    assert sim.episode_summary()["hypothetical_replica_count_used"] is True


def test_scale_down_drains_the_newest_replicas(llm_cfg, length_model, trace_s1):
    sim = _sim(llm_cfg, length_model, trace_s1, k0=4)
    sim.step_to(30.0)
    sim.set_target_replicas(2)
    draining = sorted(r.rid for r in sim.pool.draining())
    assert draining == [2, 3], "the newest replicas hold the least work"


def test_scale_up_revives_a_draining_replica(llm_cfg, length_model, trace_s1):
    sim = _sim(llm_cfg, length_model, trace_s1, k0=2)
    sim.step_to(30.0)
    sim.set_target_replicas(1)
    assert len(sim.pool.draining()) == 1
    sim.set_target_replicas(2)
    assert sim.pool.draining() == [], "cancelling a scale-down must not pay a cold start"
    assert all(r.t_ready <= 30.0 for r in sim.pool.replicas)
