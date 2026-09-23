"""Determinism and episode independence, THROUGH THE GYM API.

`kubegym/tests/test_episode.py` already pins this at the core level.  This file
re-pins it one layer up, because the failure mode the core's `WorkloadSource`
guard exists to prevent is specifically an RL failure mode: `reset()` called
thousands of times, requests mutated in place, a second episode over the same
objects silently becoming a no-op with every metric garbage and nothing raised.
A wrapper can reintroduce it by caching a request list, holding a `Simulator`
across resets in the wrong way, or sharing a workload source between envs.

So every determinism assertion here also has a NON-TRIVIALITY assertion beside
it.  Byte-identical rollouts are worthless as evidence if both rollouts are
empty: the no-op bug produces perfectly reproducible garbage.
"""
from __future__ import annotations

import gymnasium
import numpy as np
import pytest

import kubegym.gym as kgym
from kubegym.core.state import RETIRED_METRIC_NAMES, RetiredMetricError

ALL_IDS = kgym.task_ids()
ALL_TASKS = kgym.task_names()


def rollout(env, seed, actions=None, options=None):
    """A full episode; returns everything a byte-comparison needs."""
    obs0, info0 = env.reset(seed=seed, options=options)
    obs = [obs0.copy()]
    rewards, targets, costs = [], [], []
    i = 0
    while True:
        a = (actions[i % len(actions)] if actions is not None
             else i % env.action_space.n)
        o, r, term, trunc, info = env.step(a)
        obs.append(o.copy())
        rewards.append(r)
        targets.append(info["target"])
        costs.append({k: info["cost"][k] for k in ("slo", "resource", "churn", "total")})
        i += 1
        if term or trunc:
            break
    return {"obs": np.stack(obs), "rewards": rewards, "targets": targets, "costs": costs,
            "summary": info["episode_summary"], "episode_cost": info["episode_cost"],
            "identity": (info["episode_index"], info["work_seed"], info["split"]),
            "n_steps": i}


def assert_non_trivial(roll, gym_id=""):
    """The guard that makes the determinism assertions mean something.

    A no-op episode is byte-identical to itself. These are the quantities that
    collapse when requests are reused: nothing arrives, nothing completes,
    nothing is billed, wall time falls off a cliff.
    """
    s = roll["summary"]
    assert s["n_requests"] > 0, f"{gym_id}: no requests were built"
    assert s["n_done"] > 0, f"{gym_id}: no request completed -- reused-request no-op?"
    assert s["work_done"] > 0, f"{gym_id}: no service work was delivered"
    assert s["replica_seconds"] > 0, f"{gym_id}: nothing was billed"
    assert roll["n_steps"] > 1
    assert any(r != 0.0 for r in roll["rewards"]), f"{gym_id}: every reward was zero"
    assert float(np.abs(roll["obs"]).sum()) > 0.0, f"{gym_id}: the observation is all zeros"


# ---------------------------------------------------------------------------
# seed determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_same_seed_gives_byte_identical_rollouts(gym_id):
    actions = [0, 1, 0, 2, 1, 0]
    e1 = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    e2 = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    r1 = rollout(e1, 11, actions)
    r2 = rollout(e2, 11, actions)
    assert_non_trivial(r1, gym_id)
    assert r1["obs"].tobytes() == r2["obs"].tobytes(), "observations diverged"
    assert r1["rewards"] == r2["rewards"], "rewards diverged"
    assert r1["targets"] == r2["targets"]
    assert r1["costs"] == r2["costs"]
    assert r1["identity"] == r2["identity"]
    for k in ("n_requests", "n_done", "work_done", "replica_seconds", "preemptions_total",
              "mean_latency_s", "p95_latency_s", "scale_events"):
        assert r1["summary"][k] == r2["summary"][k], k


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_same_seed_on_one_env_instance_replays_identically(gym_id):
    """Reusing ONE env is the case the reused-request bug would break."""
    actions = [1, 1, 0, 0, 2]
    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    first = rollout(env, 7, actions)
    _ = rollout(env, 8, actions)             # a different episode in between
    again = rollout(env, 7, actions)
    assert_non_trivial(first, gym_id)
    assert_non_trivial(again, gym_id)
    assert first["obs"].tobytes() == again["obs"].tobytes()
    assert first["rewards"] == again["rewards"]
    assert first["summary"]["work_done"] == again["summary"]["work_done"]


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_different_seeds_give_different_episodes(gym_id):
    """Determinism is only interesting if the seed does something."""
    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    actions = [0]
    rolls = [rollout(env, s, actions) for s in (0, 1, 2, 3)]
    for r in rolls:
        assert_non_trivial(r, gym_id)
    works = {r["summary"]["work_done"] for r in rolls}
    assert len(works) > 1, (
        f"{gym_id}: four seeds produced identical offered work; the work seed is not reaching "
        "the workload source")
    idents = {r["identity"] for r in rolls}
    assert len(idents) == 4


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_many_resets_do_not_degrade(gym_id):
    """Fifty episodes, and the fiftieth is as substantive as the first.

    This is the direct RL-scale version of the core's reuse guard: the source
    testbed's bug showed up as wall time collapsing and metrics going to zero
    after the first episode.
    """
    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    first = rollout(env, 0, [0])
    for s in range(1, 49):
        env.reset(seed=s)
        for _ in range(3):
            env.step(0)
    last = rollout(env, 0, [0])
    assert_non_trivial(last, gym_id)
    assert first["obs"].tobytes() == last["obs"].tobytes()
    assert last["summary"]["n_done"] == first["summary"]["n_done"]


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_two_envs_do_not_share_workload_state(gym_id):
    """Each env builds its own source; interleaving two must not couple them."""
    e1 = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    e2 = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    assert e1.workload is not e2.workload
    solo = rollout(e1, 2, [1])
    e1.reset(seed=2)
    e2.reset(seed=9)
    interleaved_obs = []
    for _ in range(e1.n_control_steps):
        o1, _, t1, u1, _ = e1.step(1)
        e2.step(0)
        interleaved_obs.append(o1.copy())
        if t1 or u1:
            break
    assert np.stack(interleaved_obs).tobytes() == solo["obs"][1:].tobytes()


# ---------------------------------------------------------------------------
# episode / seed selection semantics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_reset_seed_maps_deterministically_into_the_split(gym_id):
    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    pairs = env.task_spec.pairs(env.split)
    for s in (0, 1, 5, 63, 1000):
        _, info = env.reset(seed=s)
        assert (info["episode_index"], info["work_seed"]) == pairs[s % len(pairs)]


def test_reset_options_select_an_explicit_episode_and_seed():
    env = kgym.KubeGymEnv("LLMServing-Mixed", split="dev")
    ws = kgym.get_task("LLMServing-Mixed").split_work_seeds["dev"][0]
    _, info = env.reset(options={"episode": 1, "work_seed": ws})
    assert info["episode_index"] == 1 and info["work_seed"] == ws
    with pytest.raises(ValueError, match="both 'episode' and 'work_seed'"):
        env.reset(options={"episode": 0})
    with pytest.raises(KeyError, match="unknown reset options"):
        env.reset(options={"epsiode": 0, "work_seed": ws})


def test_a_foreign_split_seed_is_refused():
    """A test seed must not be usable from a training env by accident."""
    spec = kgym.get_task("LLMServing-S1")
    env = kgym.KubeGymEnv("LLMServing-S1", split="train")
    test_seed = spec.split_work_seeds["test"][0]
    with pytest.raises(ValueError, match="not in split 'train'"):
        env.reset(options={"episode": 0, "work_seed": test_seed})


@pytest.mark.parametrize("task", ALL_TASKS)
def test_splits_select_disjoint_work_realisations(task):
    """Distinct splits must not produce the same episode."""
    spec = kgym.get_task(task)
    seen = {}
    for split in spec.splits():
        env = kgym.KubeGymEnv(task, split=split)
        for s in range(4):
            _, info = env.reset(seed=s)
            key = (info["episode_index"], info["work_seed"])
            assert key not in seen or seen[key] == split, (
                f"{key} reachable from both {seen.get(key)!r} and {split!r}")
            seen[key] = split


def test_episode_index_selects_a_different_trace_on_the_mixed_task():
    env = kgym.KubeGymEnv("LLMServing-Mixed")
    ws = kgym.get_task("LLMServing-Mixed").split_work_seeds["train"][0]
    r0 = rollout(env, None, [0], options={"episode": 0, "work_seed": ws})
    r1 = rollout(env, None, [0], options={"episode": 1, "work_seed": ws})
    assert_non_trivial(r0)
    assert_non_trivial(r1)
    # S1_dev has 225 arrivals, S3_dev has 221 (kubegym/tests/data/README.md)
    assert r0["summary"]["n_requests"] != r1["summary"]["n_requests"]
    assert {r0["summary"]["n_requests"], r1["summary"]["n_requests"]} == {225, 221}


# ---------------------------------------------------------------------------
# HPA-relevant semantics and the metric-name guard, seen through the Gym API
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_current_replicas_semantics_match_the_hpa_definition(gym_id):
    """`n_replicas_effective` is what the Kubernetes HPA calls `currentReplicas`.

    Every replica being billed -- ready plus booting plus draining -- as
    documented in INTERFACE.md section 6. An HPA baseline computing a ratio
    against only the READY count would under-provision during a boot and
    over-provision during a drain, so this identity is what the baseline track
    will build on. No controller is asserted here; this track ships none.
    """
    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    env.reset(seed=0)
    # Hold at the ceiling long enough for replicas to finish booting and take
    # work, THEN scale to the floor, so both the booting and the draining term
    # of the identity get a chance to be non-zero.
    n_up = env.n_control_steps // 2
    up, down = env.action_space.n - 1, 0
    saw_boot = saw_drain = False
    for i in range(env.n_control_steps):
        _, _, term, trunc, info = env.step(up if i < n_up else down)
        st = env.sim.hist[-1]
        assert info["n_replicas_effective"] == (
            info["n_ready"] + info["n_booting"] + info["n_draining"])
        assert st.n_replicas_effective == info["n_replicas_effective"]
        assert info["n_replicas_effective"] >= info["n_ready"]
        saw_boot = saw_boot or info["n_booting"] > 0
        saw_drain = saw_drain or info["n_draining"] > 0
        if term or trunc:
            break
    # Whether the booting term is EVER non-zero is a property of the physics, not
    # of the wrapper: a boot is only visible to a controller if it outlasts the
    # scrape cadence. Both regimes are asserted, so the task that has no visible
    # actuation lag is pinned as such rather than passing silently.
    lagged = env.cfg.f("replica_cold_start_s") > env.control_interval_s
    if lagged:
        assert saw_boot, (
            f"{gym_id}: cold start {env.cfg.f('replica_cold_start_s')} s exceeds the "
            f"{env.control_interval_s} s control interval, so a boot must be observable")
    else:
        assert not saw_boot, (
            f"{gym_id}: cold start {env.cfg.f('replica_cold_start_s')} s is shorter than the "
            f"{env.control_interval_s} s control interval, so a boot completes inside one step "
            "and n_booting is expected to be identically zero. If this now fails the constant "
            "changed -- update the env card, which documents these elements as dead.")
        # No claim is made about `saw_drain` in this regime. Boot invisibility is
        # deterministic (the cold start is a constant below the cadence), but a
        # drained replica stays visible for as long as it still holds work, and
        # the placeholder service-time lognormal has a heavy enough tail
        # (max_s = 600 s) that this occasionally outlasts a 15 s interval. It was
        # measured at 1 step in 580 across all constant and hold-then-drop
        # policies, so an assertion either way would be a coin flip.


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_a_drain_is_observable_when_a_ready_replica_holding_work_is_removed(gym_id):
    """The draining half of `currentReplicas`, isolated.

    A replica still BOOTING when scale-down arrives holds no work, so the core
    reaps it in the same event and it never appears as draining. Draining is
    observable only when a ready replica with work in flight is removed -- which
    is the case that makes thrash expensive, so it is worth pinning separately.
    """
    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    if env.cfg.f("replica_cold_start_s") <= env.control_interval_s:
        pytest.skip(f"{gym_id}: boots and drains both complete inside one control interval; "
                    "pinned by test_current_replicas_semantics_match_the_hpa_definition")
    env.reset(seed=0)
    up, down = env.action_space.n - 1, 0
    n_up = max(3, int(env.cfg.f("replica_cold_start_s") // env.control_interval_s) + 2)
    saw_ready_gt1 = saw_drain = False
    for i in range(env.n_control_steps):
        _, _, term, trunc, info = env.step(up if i < n_up else down)
        saw_ready_gt1 = saw_ready_gt1 or info["n_ready"] > 1
        saw_drain = saw_drain or info["n_draining"] > 0
        if term or trunc:
            break
    assert saw_ready_gt1, "no replica ever became ready; the drain case cannot arise"
    assert saw_drain, "a ready replica was removed but never appeared as draining"


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_control_cadence_is_the_hpa_sync_period(gym_id):
    """15 s: the Kubernetes HPA default sync period, per the config's own source string."""
    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    assert env.control_interval_s == 15.0
    assert "sync period" in env.cfg.source("control_interval_s")
    env.reset(seed=0)
    t_prev = env.sim.t
    for _ in range(3):
        _, _, _, _, info = env.step(0)
        assert info["t"] - t_prev == pytest.approx(env.control_interval_s)
        t_prev = info["t"]


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_retired_metric_names_still_raise_through_the_gym_api(gym_id):
    """The load-bearing guard must survive the wrapper.

    `vllm:gpu_cache_usage_perc` does not exist on vLLM 0.25.1. A controller or
    reward coded against it would read a permanent zero. Reading it from a state
    the env produced must raise, not return a default.
    """
    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    env.reset(seed=0)
    env.step(0)
    st = env.sim.hist[-1]
    for name, reason in RETIRED_METRIC_NAMES.items():
        assert st.has(name) is False
        with pytest.raises(RetiredMetricError):
            st.get(name)
    with pytest.raises(KeyError, match="not exposed"):
        st.get("vllm:this_metric_was_never_real")


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_the_env_never_enables_hypothetical_replicas_by_default(gym_id):
    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    env.reset(seed=0)
    for _ in range(env.n_control_steps):
        _, _, term, trunc, info = env.step(env.action_space.n - 1)
        assert info["hypothetical_replica_count_used"] is False
        assert info["target"] <= info["n_replicas_max_physical"]
        if term or trunc:
            break
