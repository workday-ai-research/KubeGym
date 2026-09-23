"""Gymnasium API conformance for every registered task and both action modes.

What this file pins:

  * `gymnasium.utils.env_checker.check_env` passes, with NO warnings, for every
    registered id.  Warnings are asserted absent rather than filtered: the
    benchmark's selling point is auditability, so a warning must surface in
    `gym_tests_report.md` rather than be suppressed here.
  * The action space respects `n_replicas_min` / `n_replicas_max` in both modes,
    and the measured ceiling of 4 replicas cannot be exceeded without explicitly
    opting into the core's `allow_hypothetical_replicas`.
  * The claims-gate flag reaches `info` on every step, and the episode-level
    cost row is passed through `ProvenancedConfig.stamp`.
"""
from __future__ import annotations

import warnings

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import kubegym.gym as kgym

ALL_IDS = kgym.task_ids()
ALL_TASKS = kgym.task_names()


def make_raw(gym_id: str, **kwargs):
    """The unwrapped env, so `check_env` sees our class and not OrderEnforcing."""
    return gymnasium.make(gym_id, disable_env_checker=True, **kwargs).unwrapped


# ---------------------------------------------------------------------------
# conformance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_check_env_conformance(gym_id):
    env = make_raw(gym_id)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env, skip_render_check=False)
    msgs = [str(w.message) for w in caught]
    assert msgs == [], (
        f"check_env emitted warnings for {gym_id}: {msgs}. Do NOT filter them here -- fix the "
        "env or record them in gym_tests_report.md.")


@pytest.mark.parametrize("task", ALL_TASKS)
def test_both_action_modes_registered(task):
    spec = kgym.get_task(task)
    ids = spec.gym_ids()
    assert set(ids) == {"delta", "absolute"}
    for mode, gid in ids.items():
        assert gid in ALL_IDS
        assert kgym.task_action_mode(gid) == mode


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_reset_and_step_signatures(gym_id):
    env = make_raw(gym_id)
    obs, info = env.reset(seed=3)
    assert env.observation_space.contains(obs), (obs.dtype, obs.min(), obs.max())
    assert isinstance(info, dict)
    out = env.step(env.action_space.sample())
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_episode_length_is_the_task_horizon(gym_id):
    env = make_raw(gym_id)
    env.reset(seed=0)
    n = 0
    while True:
        _, _, term, trunc, info = env.step(0)
        n += 1
        if term or trunc:
            break
        assert n < 1000, "episode did not end"
    assert n == env.n_control_steps == int(round(env.horizon_s / env.control_interval_s))
    # The horizon is intrinsic to the task, and the post-horizon drain closed the
    # episode out, so this is termination and not a time-limit truncation.
    assert term is (not info["post_horizon_drain"]["truncated_drain"])
    assert trunc is info["post_horizon_drain"]["truncated_drain"]


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_step_after_episode_end_raises(gym_id):
    env = make_raw(gym_id)
    env.reset(seed=0)
    for _ in range(env.n_control_steps):
        env.step(0)
    with pytest.raises(RuntimeError, match="after the episode ended"):
        env.step(0)


def test_no_render_mode_is_offered():
    with pytest.raises(ValueError, match="no render modes"):
        kgym.KubeGymEnv("LLMServing-S1", render_mode="human")
    env = kgym.KubeGymEnv("LLMServing-S1")
    assert env.metadata["render_modes"] == []
    assert env.render() is None


# ---------------------------------------------------------------------------
# action spaces and the replica ceiling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task", ALL_TASKS)
def test_absolute_action_space_spans_exactly_the_replica_range(task):
    env = kgym.KubeGymEnv(task, action_mode="absolute")
    assert env.action_space.n == env.k_max - env.k_min + 1
    env.reset(seed=0)
    for a in range(env.action_space.n):
        assert env._action_to_target(a) == env.k_min + a
    # No action can request anything outside the configured range.
    targets = [env._action_to_target(a) for a in range(env.action_space.n)]
    assert min(targets) == env.k_min and max(targets) == env.k_max


@pytest.mark.parametrize("task", ALL_TASKS)
def test_delta_action_space_is_the_symmetric_increment_set(task):
    spec = kgym.get_task(task)
    env = kgym.KubeGymEnv(task, action_mode="delta")
    assert env.action_space.n == len(spec.delta_set)
    assert 0 in spec.delta_set, "a hold action must exist"
    assert sorted(spec.delta_set) == sorted(-d for d in spec.delta_set), "increments symmetric"


@pytest.mark.parametrize("task", ALL_TASKS)
def test_delta_action_clamps_at_the_replica_ceiling(task):
    """Pushing +max delta repeatedly must stop at n_replicas_max, not past it."""
    env = kgym.KubeGymEnv(task, action_mode="delta")
    up = int(np.argmax(env.delta_set))            # the largest positive increment
    _, info = env.reset(seed=0)
    clamped_seen = False
    for _ in range(env.n_control_steps):
        _, _, term, trunc, info = env.step(up)
        assert info["target"] <= env.k_max_physical, info
        assert info["hypothetical_replica_count_used"] is False
        if info["action_clamped"]:
            clamped_seen = True
        if term or trunc:
            break
    assert info["target"] == env.k_max_physical
    assert clamped_seen, (
        "target never hit the ceiling, so clamping was not exercised; the episode is too short "
        "or the delta set too small for this assertion to mean anything")


@pytest.mark.parametrize("task", ALL_TASKS)
def test_delta_action_clamps_at_the_replica_floor(task):
    env = kgym.KubeGymEnv(task, action_mode="delta")
    down = int(np.argmin(env.delta_set))
    env.reset(seed=0)
    for _ in range(env.n_control_steps):
        _, _, term, trunc, info = env.step(down)
        assert info["target"] >= env.k_min
        if term or trunc:
            break
    assert info["target"] == env.k_min


def test_exceeding_the_measured_ceiling_needs_an_explicit_opt_in():
    """`n_replicas_max=4` is a MEASURED physical limit; it must not be crossed silently."""
    env = kgym.KubeGymEnv("LLMServing-S1", action_mode="absolute")
    assert env.k_max == env.k_max_physical == 4
    assert env.action_space.n == 4                       # k_min=1 .. k_max=4
    assert env.allow_hypothetical_replicas is False

    with pytest.warns(RuntimeWarning, match="tainted"):
        hyp = kgym.KubeGymEnv("LLMServing-S1", action_mode="absolute",
                              allow_hypothetical_replicas=True,
                              hypothetical_replicas_max=6)
    assert hyp.k_max == 6 and hyp.k_max_physical == 4
    _, info = hyp.reset(seed=0)
    assert info["hypothetical_replicas_allowed"] is True
    _, _, _, _, info = hyp.step(hyp.action_space.n - 1)   # request 6 replicas
    assert info["target"] == 6
    assert info["hypothetical_replica_count_used"] is True, (
        "the core must taint the pool when a target exceeds the measured ceiling")


def test_hypothetical_ceiling_must_actually_be_above_the_physical_one():
    with pytest.raises(ValueError, match="not above n_replicas_max"):
        kgym.KubeGymEnv("LLMServing-S1", allow_hypothetical_replicas=True,
                        hypothetical_replicas_max=4)


def test_invalid_action_is_rejected():
    env = kgym.KubeGymEnv("LLMServing-S1", action_mode="absolute")
    env.reset(seed=0)
    with pytest.raises(ValueError, match="not in Discrete"):
        env.step(env.action_space.n + 3)


# ---------------------------------------------------------------------------
# the calibration flag, on every row
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_calibration_flag_reaches_info_on_every_step(gym_id):
    env = make_raw(gym_id)
    _, info = env.reset(seed=1)
    assert info["calibrated"] is False, (
        "both shipped configs are calibrated=false; if this ever becomes True the config "
        "changed and every claim in the paper must be revisited")
    assert info["uncalibrated_required_fields"], "the blocking fields must be named, not just flagged"
    n = 0
    while True:
        _, _, term, trunc, info = env.step(0)
        n += 1
        assert info["calibrated"] is env.cfg.calibrated
        assert set(info["uncalibrated_required_fields"]) == set(env.cfg.uncalibrated_required)
        if term or trunc:
            break
    assert n == env.n_control_steps


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_episode_end_info_is_provenance_stamped(gym_id):
    env = make_raw(gym_id)
    env.reset(seed=2)
    for _ in range(env.n_control_steps):
        _, _, term, trunc, info = env.step(0)
    # the core's own stamped summary
    summary = info["episode_summary"]
    assert summary["calibrated"] is False
    assert "uncalibrated_required_fields" in summary
    # our episode cost row, stamped through the same code path
    row = info["episode_cost"]
    assert row["calibrated"] is False
    assert row["config"] == "llm_serving_l40s.json" or row["config"].endswith(".json")
    assert row["uncalibrated_required_fields"]
    assert row["return"] == pytest.approx(-row["cost_total"])


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_info_reconstructs_the_episode(gym_id):
    """`info` must carry enough to re-run and to audit the step."""
    env = make_raw(gym_id)
    _, info0 = env.reset(seed=5)
    identity = {"task", "task_id", "action_mode", "split", "episode_index", "work_seed", "config"}
    assert identity <= set(info0)
    _, _, _, _, info = env.step(0)
    required = identity | {
        "t", "step", "n_control_steps", "action", "target_before", "target_requested",
        "target", "action_clamped", "n_ready", "n_booting", "n_draining",
        "n_replicas_effective", "arrivals_since_last", "replica_seconds_step",
        "replica_seconds_total", "cost", "cost_cumulative", "slo", "calibrated",
        "uncalibrated_required_fields", "hypothetical_replica_count_used",
    }
    missing = required - set(info)
    assert not missing, f"info is missing {sorted(missing)}"
    # re-running from the identity fields alone must reproduce the episode
    env2 = kgym.KubeGymEnv(info["task"], action_mode=info["action_mode"], split=info["split"])
    env2.reset(options={"episode": info["episode_index"], "work_seed": info["work_seed"]})
    _, _, _, _, info2 = env2.step(0)
    assert info2["cost"]["total"] == pytest.approx(info["cost"]["total"])
    assert info2["t"] == pytest.approx(info["t"])


def test_make_via_gymnasium_applies_no_time_limit_wrapper():
    """A TimeLimit on top of our horizon would let episode length disagree with the task."""
    env = gymnasium.make("KubeGym/LLMServing-S1-Delta-v0")
    assert not any(type(w).__name__ == "TimeLimit"
                   for w in _wrapper_chain(env)), _wrapper_chain(env)
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.unwrapped.observation_space.shape[0],)
    assert info["task_id"] == "KubeGym/LLMServing-S1-Delta-v0"


def _wrapper_chain(env):
    chain = []
    while hasattr(env, "env"):
        chain.append(env)
        env = env.env
    return chain


def test_split_must_exist():
    with pytest.raises(KeyError, match="has no split"):
        kgym.KubeGymEnv("LLMServing-S1", split="validation")


def test_action_mode_must_be_known():
    with pytest.raises(ValueError, match="action_mode must be one of"):
        kgym.KubeGymEnv("LLMServing-S1", action_mode="continuous")
