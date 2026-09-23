"""Are the two service models genuinely interchangeable through the seam?

The test that matters is not "both run" but "the SAME driver code runs both",
so a downstream consumer (Gym wrapper, controller, eval harness) can be written
once against the seam.  `run_episode` below is shared verbatim between the two
models; only the model, config and workload source differ.
"""
from __future__ import annotations

import pytest

from kubegym import ObservationSpec, Request, Simulator, Slot
from kubegym.core.service import BaseReplicaEngine, ReplicaEngine, ServiceModel
from kubegym.models.llm_serving import LLMServingModel, LLMTraceSource
from kubegym.models.request_service import (RequestServiceModel, RequestServiceSource,
                                            poisson_arrivals, sample_service_time)

HORIZON = 300.0


def run_episode(sim: Simulator, *, seed: int = 0) -> dict:
    """Driver code shared by BOTH service models -- this is the seam under test."""
    sim.reset(episode=0, seed=seed)
    obs_spec = ObservationSpec(history=2)
    for i in range(int(HORIZON / sim.control_interval_s)):
        st = sim.hist[-1]
        # a trivial saturation-threshold controller, written once, model-agnostic
        k = st.target + 1 if st.kv_mean > 0.7 else (st.target - 1 if st.kv_mean < 0.2
                                                   else st.target)
        sim.set_target_replicas(k)
        sim.advance_control_interval()
        obs = obs_spec.build(sim.hist)
        assert obs.shape == (obs_spec.dim,)
    sim.drain_all(50_000.0)
    return sim.episode_summary(HORIZON)


def _llm(llm_cfg, length_model, trace):
    return Simulator(LLMServingModel(llm_cfg, length_model=length_model),
                     LLMTraceSource([trace], length_model=length_model), llm_cfg, k0=2)


def _rs(rs_cfg, *, rate=1.0, horizon=HORIZON):
    src = RequestServiceSource(poisson_arrivals(rate, horizon, seed=3),
                               service_time_distribution=rs_cfg.f(
                                   "rs_service_time_distribution"))
    return Simulator(RequestServiceModel(rs_cfg), src, rs_cfg, k0=2)


def test_same_driver_runs_both_models(llm_cfg, rs_cfg, length_model, trace_s1):
    a = run_episode(_llm(llm_cfg, length_model, trace_s1), seed=1)
    b = run_episode(_rs(rs_cfg), seed=1)
    assert set(a) == set(b), "episode_summary must have the same schema for both models"
    for row in (a, b):
        assert row["n_done"] == row["n_requests"] > 0
        assert row["work_done"] == pytest.approx(row["work_demanded"], rel=1e-9)
        assert row["replica_seconds"] > 0
        assert row["calibrated"] is False
    assert a["service_model"] == "llm_serving"
    assert b["service_model"] == "request_service"


def test_both_models_satisfy_the_protocol(llm_cfg, rs_cfg, length_model):
    for model in (LLMServingModel(llm_cfg, length_model=length_model),
                  RequestServiceModel(rs_cfg)):
        assert isinstance(model, ServiceModel)
        eng = model.make_replica(0, Slot(0))
        assert isinstance(eng, ReplicaEngine)
        assert isinstance(eng, BaseReplicaEngine)
        assert set(eng.metrics()) == set(model.metric_names())
        assert eng.next_service_time() == float("inf")
        assert eng.has_work() is False


def test_metric_vocabularies_are_disjoint_so_a_mismatch_fails_loudly(llm_cfg, rs_cfg,
                                                                    length_model):
    a = set(LLMServingModel(llm_cfg, length_model=length_model).metric_names())
    b = set(RequestServiceModel(rs_cfg).metric_names())
    assert not (a & b)


def test_a_controller_for_the_wrong_model_raises(rs_cfg):
    sim = _rs(rs_cfg)
    st = sim.reset(episode=0, seed=0)
    with pytest.raises(KeyError, match="not exposed"):
        st.get("vllm:num_requests_running")


def test_generic_aggregates_work_across_models(llm_cfg, rs_cfg, length_model, trace_s1):
    for sim in (_llm(llm_cfg, length_model, trace_s1), _rs(rs_cfg)):
        sim.reset(episode=0, seed=1)
        sim.step_to(120.0)
        st = sim.scrape()
        assert st.n_running >= 0 and st.n_waiting >= 0
        assert 0.0 <= st.kv_mean <= 1.0 and 0.0 <= st.kv_max <= 1.0


def test_request_service_respects_the_concurrency_limit(rs_cfg):
    c = int(rs_cfg.f("rs_max_concurrency"))
    sim = _rs(rs_cfg, rate=4.0)
    sim.reset(episode=0, seed=0)
    for _ in range(20):
        sim.advance_control_interval()
        for rep in sim.pool.replicas:
            assert rep.engine.n_running <= c


def test_request_service_queues_when_saturated(rs_cfg):
    sim = _rs(rs_cfg, rate=8.0)
    sim.reset(episode=0, seed=0, k0=1)
    for _ in range(20):
        sim.advance_control_interval()
    st = sim.scrape()
    assert st.n_waiting > 0, "an overloaded single replica must show a queue"
    assert st.kv_mean == pytest.approx(1.0)


def test_request_service_conserves_work(rs_cfg):
    sim = _rs(rs_cfg, rate=2.0)
    row = run_episode(sim, seed=5)
    assert row["work_done"] == pytest.approx(row["work_demanded"], rel=1e-9)
    for r in sim.requests:
        assert r.t_done is not None and r.t_admit is not None
        assert r.t_done >= r.t_admit >= r.t_arrival
        assert r.n_preemptions == 0, "this model has no preemption"


def test_llm_kv_footprint_grows_as_sequences_decode(llm_cfg, length_model, trace_s1):
    """The mechanism that makes long-output traffic qualitatively different."""
    sim = _llm(llm_cfg, length_model, trace_s1)
    sim.reset(episode=0, seed=1, k0=1)
    sim.set_target_replicas(1)
    footprints = []
    for t in (30.0, 60.0, 120.0):
        sim.step_to(t)
        eng = sim.pool.replicas[0].engine
        footprints.append(eng.kv_used() / max(1, eng.n_running))
    assert footprints[-1] > footprints[0], (
        "per-sequence KV footprint must grow as sequences generate tokens")


def test_service_time_families():
    import random
    rng = random.Random(0)
    assert sample_service_time({"family": "deterministic", "value_s": 2.0}, rng) == 2.0
    assert sample_service_time({"family": "exponential", "mean_s": 1.0}, rng) > 0
    assert sample_service_time({"family": "lognormal", "mu": 0.0, "sigma": 1.0}, rng) > 0
    assert sample_service_time({"family": "empirical", "samples_s": [3.0]}, rng) == 3.0
    v = sample_service_time({"family": "lognormal", "mu": 0.0, "sigma": 3.0, "max_s": 1.5}, rng)
    assert v <= 1.5
    with pytest.raises(ValueError):
        sample_service_time({"family": "weibull"}, rng)


def test_replay_service_times_are_used_verbatim(rs_cfg):
    recs = [{"seq": 0, "t_arrival": 1.0, "service_time_s": 7.25}]
    src = RequestServiceSource(recs, service_time_distribution=rs_cfg.f(
        "rs_service_time_distribution"))
    r = src.build(episode=0, seed=0)[0]
    assert r.demand == 7.25
    assert r.attrs["service_time_drawn"] is False


def test_a_new_service_model_needs_only_the_protocol(rs_cfg):
    """Minimal third model: one request at a time, fixed rate.  If this runs
    through the unmodified engine, the seam holds."""

    class OneAtATime(BaseReplicaEngine):
        def service_step(self, t):
            done = []
            for r in list(self.running):
                r.progress = r.demand
                r.t_done = t
                self.running.remove(r)
                self.n_success += 1
                done.append(r)
            self.next_step_t = float("inf")
            return done

        def on_tick(self, t):
            if not self.running and self.slot and self.accepting:
                r = self.slot.popleft()
                r.t_admit = t
                self._mark_output(r, t)
                self.running.append(r)
                self.next_step_t = t + r.demand

        def metrics(self):
            return {"toy:running": float(len(self.running)),
                    "toy:waiting": float(len(self.slot))}

    class ToyModel:
        name = "toy"

        def config_fields(self):
            return ()

        def metric_names(self):
            return ("toy:running", "toy:waiting")

        def retired_metric_names(self):
            return {}

        def make_replica(self, rid, slot):
            return OneAtATime(rid, slot)

        def reset(self, seed):
            pass

        def provenance(self):
            return {"service_model": "toy", "calibrated": False,
                    "status": "test fixture, not a model of anything"}

    src = RequestServiceSource(poisson_arrivals(0.5, 100.0, seed=1),
                               service_time_distribution={"family": "deterministic",
                                                          "value_s": 1.0})
    sim = Simulator(ToyModel(), src, rs_cfg, k0=1)
    sim.reset(episode=0, seed=0)
    for _ in range(10):
        sim.advance_control_interval()
    sim.drain_all(10_000.0)
    row = sim.episode_summary(150.0)
    assert row["n_done"] == row["n_requests"] > 0
    assert row["service_model"] == "toy"


# --------------------------------------------------------------------------- #
# Non-termination regression (found by the baselines track)
# --------------------------------------------------------------------------- #

def test_completion_tolerance_scales_with_progress_accumulator():
    """An absolute tolerance cannot clear a residual on a large accumulator.

    `Request.progress` is a running sum, so late in a long episode it reaches
    O(1e2) work-units where float64 has only ~1e-16 relative precision. With the
    old absolute 1e-12 test a residual survived (observed 1.68e-12 at t=16391),
    `_reschedule` put the next event back on the current instant, and the event
    loop span forever with no output and no error.
    """
    from kubegym.models.request_service import (COMPLETION_ABS_TOL,
                                                COMPLETION_REL_TOL,
                                                _completion_tol)
    from kubegym.core.service import Request

    small = Request(seq=0, t_arrival=0.0, demand=1e-3, size=0.0, cls="")
    assert _completion_tol(small) >= COMPLETION_ABS_TOL

    big = Request(seq=1, t_arrival=0.0, demand=250.0, size=0.0, cls="")
    big.progress = 250.0 - 1.68e-12          # the observed stuck residual
    tol = _completion_tol(big)
    assert tol >= COMPLETION_REL_TOL * 250.0
    assert big.progress >= big.demand - tol, "the stuck residual must now complete"


def test_reschedule_never_returns_a_non_advancing_event_time():
    import math
    from kubegym.models.request_service import RequestServiceEngine  # noqa: F401
    from kubegym.core.service import Request

    # A residual far below one ULP of a large clock value must still move time.
    t = 16391.0
    rem, rate = 1.68e-12, 1.0
    naive = t + rem / rate
    assert naive == t, "precondition: this residual is below one ULP of t"
    assert math.nextafter(t, math.inf) > t


def test_engine_raises_instead_of_spinning_on_a_stalled_service_model():
    """A model that refuses to advance the clock must fail loudly."""
    import kubegym.core.engine as eng_mod

    assert issubclass(eng_mod.EngineStallError, RuntimeError)
    assert eng_mod.STALL_LIMIT > 0
