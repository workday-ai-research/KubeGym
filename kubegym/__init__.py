"""kubegym -- a reproducible benchmark core for RL in cloud resource management.

This package is the *environment core*: a service-model-agnostic discrete-event
simulator, a provenance-tracked configuration system, and two service models.
The Gymnasium wrapper, the workload corpus and the RL baselines are separate
layers built on the interfaces documented in INTERFACE.md; nothing here imports
gymnasium, torch or stable-baselines3, and the core depends only on the standard
library plus numpy.

Quick start
-----------
    from kubegym import (ProvenancedConfig, Simulator, config_path,
                         LLMServingModel, LLMTraceSource)
    from kubegym.models.length_model import load_length_model

    cfg = ProvenancedConfig.load(config_path("llm_serving_l40s.json"))
    lm = load_length_model("measured_lengths.json")     # placeholder if absent
    model = LLMServingModel(cfg, length_model=lm)
    src = LLMTraceSource(["scenarios/S1_dev.jsonl"], length_model=lm)
    sim = Simulator(model, src, cfg, k0=2)

    sim.reset(episode=0, seed=1)
    sim.set_target_replicas(3)
    state = sim.advance_control_interval()      # -> ClusterState
    sim.drain_all(t_cap=3600.0)
    row = sim.episode_summary()                # provenance-stamped

THE CLAIMS GATE
---------------
`cfg.calibrated` is False whenever any field marked `required_for_claims` is a
placeholder, and `Simulator.episode_summary()` stamps that flag onto every
result row.  Both shipped configs are `calibrated: false`.  No number produced
by this package with those configs may be reported as an empirical result.
"""
from __future__ import annotations

import os

__version__ = "0.1.0"

from .core.engine import Simulator
from .core.replica import Replica, ReplicaPool
from .core.router import make_policy, register_policy
from .core.service import BaseReplicaEngine, ReplicaEngine, Request, ServiceModel, Slot
from .core.state import (RETIRED_METRIC_NAMES, ClusterState, ObservationSpec,
                         RetiredMetricError)
from .core.workload import (JsonlTraceSource, RecordListSource, WorkloadReuseError,
                            WorkloadSource)
from .models.llm_serving import LLMServingModel, LLMTraceSource, make_llm_serving
from .models.request_service import (RequestServiceModel, RequestServiceSource,
                                     poisson_arrivals)
from .provenance import Field, ProvenanceError, ProvenancedConfig, load_config

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def config_path(name: str) -> str:
    """Absolute path to a shipped config, by file name."""
    p = os.path.join(CONFIG_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"no shipped config {name!r}; have {sorted(os.listdir(CONFIG_DIR))}")
    return p


__all__ = [
    "__version__",
    # config / provenance
    "ProvenancedConfig", "Field", "ProvenanceError", "load_config", "config_path", "CONFIG_DIR",
    # core
    "Simulator", "ReplicaPool", "Replica", "ClusterState", "ObservationSpec",
    "RetiredMetricError", "RETIRED_METRIC_NAMES",
    "Request", "Slot", "ServiceModel", "ReplicaEngine", "BaseReplicaEngine",
    "WorkloadSource", "RecordListSource", "JsonlTraceSource", "WorkloadReuseError",
    "make_policy", "register_policy",
    # models
    "LLMServingModel", "LLMTraceSource", "make_llm_serving",
    "RequestServiceModel", "RequestServiceSource", "poisson_arrivals",
]
