"""The task registry: id scheme, split separation, and the extension point.

A benchmark's registry is where comparability is either preserved or quietly
lost.  What this file pins:

  * split work seeds are DISJOINT for every registered task, so a `test`
    episode cannot be a training episode under another name;
  * `list_tasks()` carries the calibration flag, so nobody picks a task without
    seeing that its physics are uncalibrated;
  * re-registering a name is refused, so two runs of one id are one task;
  * `register_task` / `register_workload_task` work for a caller outside this
    package -- the contract the workload-corpus track builds against -- and the
    resulting env is a fully functional `KubeGymEnv`.
"""
from __future__ import annotations

import gymnasium
import pytest

import kubegym.gym as kgym
from kubegym import ProvenancedConfig, config_path
from kubegym.gym.tasks import ACTION_MODES, DEFAULT_SPLIT_WORK_SEEDS, TaskSpec

ALL_IDS = kgym.task_ids()
ALL_TASKS = kgym.task_names()


# ---------------------------------------------------------------------------
# the id scheme
# ---------------------------------------------------------------------------
def test_builtin_tasks_are_registered_in_both_action_modes():
    assert len(ALL_TASKS) == 4, ALL_TASKS
    assert len(ALL_IDS) == 2 * len(ALL_TASKS)
    assert set(ALL_TASKS) == {"LLMServing-S1", "LLMServing-S3", "LLMServing-Mixed",
                              "RequestService-Poisson"}


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_id_scheme(gym_id):
    assert gym_id.startswith("KubeGym/")
    assert gym_id.endswith("-v0")
    task, mode = gym_id[len("KubeGym/"):-len("-v0")].rsplit("-", 1)
    assert mode in set(ACTION_MODES.values())
    assert task in ALL_TASKS
    assert kgym.get_task(gym_id) is kgym.get_task(task)
    assert gym_id in gymnasium.registry


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_every_registered_id_constructs_and_runs(gym_id):
    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["task_id"] == gym_id
    env.step(0)
    env.close()


# ---------------------------------------------------------------------------
# split separation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task", ALL_TASKS)
def test_split_work_seeds_are_disjoint(task):
    spec = kgym.get_task(task)
    splits = spec.splits()
    assert set(splits) == {"train", "dev", "test"}
    seen = {}
    for s in splits:
        seeds = spec.split_work_seeds[s]
        assert len(seeds) == len(set(seeds)), f"{task}/{s} repeats a work seed"
        for w in seeds:
            assert w not in seen, (
                f"{task}: work seed {w} is in both {seen.get(w)!r} and {s!r}; a test episode "
                "would be a training episode under another name")
            seen[w] = s
    assert len(spec.pairs("train")) == len(spec.split_work_seeds["train"]) * spec.n_episodes


def test_the_spec_refuses_overlapping_splits_at_construction():
    ref = kgym.get_task("LLMServing-S1")
    bad = {"train": (1, 2, 3), "test": (3, 4)}
    with pytest.raises(ValueError, match="Splits must be disjoint"):
        TaskSpec(name="Bad-Overlap", service_model_name=ref.service_model_name,
                 config_file=ref.config_file, workload_name="x", workload_description="x",
                 workload_factory=ref.workload_factory,
                 service_model_factory=ref.service_model_factory,
                 n_episodes=1, horizon_s=300.0, k0=1, obs_features=ref.obs_features,
                 obs_history=ref.obs_history, obs_norm=ref.obs_norm, weights=ref.weights,
                 slo_profile=ref.slo_profile, split_work_seeds=bad)


def test_default_split_seed_ranges_are_disjoint():
    pools = [set(v) for v in DEFAULT_SPLIT_WORK_SEEDS.values()]
    for i, a in enumerate(pools):
        for b in pools[i + 1:]:
            assert not (a & b)


# ---------------------------------------------------------------------------
# listing
# ---------------------------------------------------------------------------
def test_list_tasks_carries_the_calibration_flag():
    rows = kgym.list_tasks()
    assert len(rows) == len(ALL_IDS)
    for r in rows:
        assert r["calibrated"] is False, (
            "both shipped configs are calibrated=false; a True here means the config changed")
        assert r["uncalibrated_required_fields"], "the blocking fields must be named"
        assert r["obs_dim"] > 0 and r["n_control_steps"] > 0
        assert r["n_replicas_min"] <= r["n_replicas_max"]
        assert set(r["weights"]) == {"slo_per_violating_request",
                                     "resource_per_replica_second",
                                     "churn_per_replica_changed"}
        assert r["splits"] == ["dev", "test", "train"]
    assert {r["id"] for r in rows} == set(ALL_IDS)


def test_list_tasks_weights_match_the_env_that_gets_built():
    for r in kgym.list_tasks():
        env = gymnasium.make(r["id"], disable_env_checker=True).unwrapped
        assert env.cost_model.weights.to_dict() == r["weights"]
        assert env.observation_space.shape[0] == r["obs_dim"]
        assert env.n_control_steps == r["n_control_steps"]
        assert env.k_max == r["n_replicas_max"]


@pytest.mark.parametrize("task", ALL_TASKS)
def test_control_steps_divide_the_horizon_exactly(task):
    spec = kgym.get_task(task)
    n = spec.n_control_steps()
    assert n * spec.control_interval_s() == pytest.approx(spec.horizon_s)


def test_a_horizon_that_does_not_divide_is_refused():
    ref = kgym.get_task("LLMServing-S1")
    spec = TaskSpec(name="Bad-Horizon", service_model_name=ref.service_model_name,
                    config_file=ref.config_file, workload_name="x", workload_description="x",
                    workload_factory=ref.workload_factory,
                    service_model_factory=ref.service_model_factory,
                    n_episodes=1, horizon_s=307.0, k0=1, obs_features=ref.obs_features,
                    obs_history=ref.obs_history, obs_norm=ref.obs_norm, weights=ref.weights,
                    slo_profile=ref.slo_profile)
    with pytest.raises(ValueError, match="not an integer multiple"):
        spec.n_control_steps()


def test_delta_set_must_contain_a_hold_action():
    ref = kgym.get_task("LLMServing-S1")
    with pytest.raises(ValueError, match="does not contain 0"):
        TaskSpec(name="Bad-Delta", service_model_name=ref.service_model_name,
                 config_file=ref.config_file, workload_name="x", workload_description="x",
                 workload_factory=ref.workload_factory,
                 service_model_factory=ref.service_model_factory,
                 n_episodes=1, horizon_s=300.0, k0=1, obs_features=ref.obs_features,
                 obs_history=ref.obs_history, obs_norm=ref.obs_norm, weights=ref.weights,
                 slo_profile=ref.slo_profile, delta_set=(-1, 1))


# ---------------------------------------------------------------------------
# the extension point
# ---------------------------------------------------------------------------
def _external_workload_factory():
    """What a downstream track's factory looks like: a core source, fresh per env."""
    def factory():
        from kubegym.models.request_service import RequestServiceSource, poisson_arrivals
        cfg = ProvenancedConfig.load(config_path("request_service_default.json"))
        return RequestServiceSource(
            poisson_arrivals(rate_rps=2.0, horizon_s=150.0, seed=99, cls="ext"),
            service_time_distribution=cfg.f("rs_service_time_distribution"),
            name="external_test_workload", horizon_s=150.0,
            manifest={"registered_by": "test_gym_registry"})
    return factory


def test_register_workload_task_is_a_working_extension_point():
    """The corpus track registers tasks without this package importing theirs."""
    name = "ExternalTest-Poisson"
    ids = kgym.register_workload_task(
        name=name, workload_factory=_external_workload_factory(),
        n_episodes=1, horizon_s=150.0, service_model="request_service",
        workload_name="ext_poisson_2rps", workload_description="registered by the test suite",
        notes=("Synthetic task registered by kubegym/tests/test_gym_registry.py.",),
        override=True)
    try:
        assert sorted(ids) == [f"KubeGym/{name}-Abs-v0", f"KubeGym/{name}-Delta-v0"]
        spec = kgym.get_task(name)
        # conventions inherited from the reference spec, and recorded as inherited
        ref = kgym.get_task("RequestService-Poisson")
        assert spec.obs_features == ref.obs_features
        assert spec.weights == ref.weights
        assert spec.slo_profile == ref.slo_profile
        assert any("register_workload_task" in n for n in spec.notes)
        assert spec.n_control_steps() == 10

        env = gymnasium.make(ids[0], disable_env_checker=True).unwrapped
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert info["calibrated"] is False
        n = 0
        while True:
            _, _, term, trunc, info = env.step(0)
            n += 1
            if term or trunc:
                break
        assert n == 10
        assert info["episode_summary"]["n_done"] > 0
        assert env.workload.manifest()["registered_by"] == "test_gym_registry"
        assert name in kgym.task_names()
        assert {r["id"] for r in kgym.list_tasks()} >= set(ids)
    finally:
        _unregister(name, ids)


def test_register_task_refuses_to_silently_replace_a_task():
    spec = kgym.get_task("LLMServing-S1")
    with pytest.raises(ValueError, match="already registered"):
        kgym.register_task(spec)


def test_register_task_rejects_an_unknown_action_mode():
    spec = kgym.get_task("LLMServing-S1")
    with pytest.raises(KeyError, match="unknown action modes"):
        kgym.register_task(spec, action_modes=("delta", "continuous"), override=True)


def test_register_workload_task_needs_a_known_service_model():
    with pytest.raises(KeyError, match="no reference spec"):
        kgym.register_workload_task(name="X", workload_factory=lambda: None,
                                    n_episodes=1, horizon_s=15.0,
                                    service_model="does_not_exist")


def test_register_builtin_tasks_is_idempotent():
    before = set(kgym.task_ids())
    again = kgym.register_builtin_tasks()
    assert set(kgym.task_ids()) == before
    assert set(again) == before


def _unregister(name, ids):
    """Undo a test registration so the registry is not polluted for other tests."""
    from kubegym.gym import tasks as t
    t._TASKS.pop(name, None)
    for gid in ids:
        t._IDS.pop(gid, None)
        gymnasium.registry.pop(gid, None)
