"""The metric-name guard.

`vllm:gpu_cache_usage_perc` does not exist on vLLM 0.25.1 (verified absent from
all 66 `vllm:*` families on the live server).  A controller coded against it
reads KV pressure as a permanent zero: a silently broken baseline.  Reading it
must RAISE, never return zero or a default -- whatever service model is active.
"""
from __future__ import annotations

import pytest

from kubegym import ClusterState, RetiredMetricError, Simulator
from kubegym.core.state import RETIRED_METRIC_NAMES
from kubegym.models.llm_serving import METRIC_NAMES, LLMServingModel, LLMTraceSource
from kubegym.models.request_service import (RequestServiceModel, RequestServiceSource,
                                            poisson_arrivals)


def _llm_state(llm_cfg, length_model, trace):
    model = LLMServingModel(llm_cfg, length_model=length_model)
    sim = Simulator(model, LLMTraceSource([trace], length_model=length_model), llm_cfg, k0=2)
    st = sim.reset(episode=0, seed=1)
    sim.step_to(60.0)
    return sim, sim.scrape()


def test_retired_metric_raises(llm_cfg, length_model, trace_s1):
    _, st = _llm_state(llm_cfg, length_model, trace_s1)
    with pytest.raises(KeyError) as e:
        st.get("vllm:gpu_cache_usage_perc")
    msg = str(e.value)
    assert "does not exist on vLLM 0.25.1" in msg
    assert "vllm:kv_cache_usage_perc" in msg, "the error must name the correct metric"
    assert isinstance(e.value, RetiredMetricError)
    with pytest.raises(KeyError):
        st.get("vllm:cpu_cache_usage_perc")


def test_retired_metric_raises_for_every_service_model(rs_cfg):
    """The guard is a floor in the core, not a per-model courtesy."""
    model = RequestServiceModel(rs_cfg)
    src = RequestServiceSource(poisson_arrivals(0.5, 60.0, seed=0),
                               service_time_distribution=rs_cfg.f(
                                   "rs_service_time_distribution"))
    sim = Simulator(model, src, rs_cfg, k0=1)
    st = sim.reset(episode=0, seed=0)
    with pytest.raises(KeyError, match="does not exist on vLLM 0.25.1"):
        st.get("vllm:gpu_cache_usage_perc")


def test_has_is_false_for_retired_names(llm_cfg, length_model, trace_s1):
    _, st = _llm_state(llm_cfg, length_model, trace_s1)
    assert st.has("vllm:kv_cache_usage_perc") is True
    assert st.has("vllm:gpu_cache_usage_perc") is False, (
        "has() must not report a retired name as available")


def test_unknown_metric_raises_and_lists_what_exists(llm_cfg, length_model, trace_s1):
    _, st = _llm_state(llm_cfg, length_model, trace_s1)
    with pytest.raises(KeyError) as e:
        st.get("vllm:invented_metric")
    assert "not exposed" in str(e.value)
    assert "vllm:kv_cache_usage_perc" in str(e.value)


def test_all_verified_metrics_present_one_entry_per_ready_replica(llm_cfg, length_model,
                                                                 trace_s1):
    _, st = _llm_state(llm_cfg, length_model, trace_s1)
    assert st.n_ready == 2
    for name in METRIC_NAMES:
        assert len(st.get(name)) == st.n_ready, name


def test_booting_replica_exposes_no_metrics(llm_cfg, length_model, trace_s1):
    sim, _ = _llm_state(llm_cfg, length_model, trace_s1)
    sim.set_target_replicas(4)
    st = sim.scrape()
    assert st.n_booting == 2
    for name in METRIC_NAMES:
        assert len(st.get(name)) == st.n_ready, (
            "a booting replica has no metrics endpoint and must contribute no entry")
    assert st.n_replicas_effective == st.n_ready + st.n_booting + st.n_draining


def test_retired_map_is_not_empty():
    assert "vllm:gpu_cache_usage_perc" in RETIRED_METRIC_NAMES, (
        "the always-on retired-metric floor must not be emptied")


def test_state_constructed_directly_still_guards():
    st = ClusterState(t=0.0, metrics={"vllm:kv_cache_usage_perc": [0.1]}, n_ready=1,
                      n_booting=0, n_draining=0, target=1)
    with pytest.raises(KeyError):
        st.get("vllm:gpu_cache_usage_perc")
