"""Observation semantics: element order, bounds, and no privileged information.

The fairness guarantee of the benchmark (INTERFACE.md section 6) is that an RL
policy can never see more than a hand-written controller can.  The core enforces
half of it -- every entry of `kubegym.core.state.FEATURES` is a function of a
`ClusterState` alone.  The wrapper can still break it, in two ways this file
closes:

  * by adding an element that is not a `FEATURES` key and not something the
    core's own `decide(hist, t)` controller interface receives;
  * by normalising with a quantity derived from the episode, which is
    information from the future dressed up as a scaled metric.

`test_observation_is_invariant_to_mutating_request_ground_truth` is the direct
check: it takes a real scrape history, rewrites `demand` (the ground-truth total
work) on every request in the episode, rebuilds the observation from the same
history, and asserts it does not move.  If the observation ever held a reference
into a live `Request` instead of a snapshot, that test fails.
"""
from __future__ import annotations

import numpy as np
import pytest

import kubegym.gym as kgym
from kubegym.core.state import FEATURES, ObservationSpec
from kubegym.gym.obs import FEATURE_SCALE_KIND, ObsNorm, normalise

ALL_TASKS = kgym.task_names()

#: Keys the core's `router_inflight` is allowed to expose.  None of them is a
#: function of a request's total work: `tokens_emitted` is what the router has
#: already forwarded, `prompt_tokens` is the visible prompt size, `observed_mode`
#: / `cls_observed` are the labels the router itself saw on the request.
ROUTER_INFLIGHT_WHITELIST = {"seq", "tokens_emitted", "t_admit", "prompt_tokens",
                             "observed_mode", "cls_observed"}


def rollout(env, n_steps, action=0):
    env.reset(seed=0)
    obs = []
    for _ in range(n_steps):
        o, _, term, trunc, _ = env.step(action)
        obs.append(o)
        if term or trunc:
            break
    return obs


# ---------------------------------------------------------------------------
# element order and layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task", ALL_TASKS)
def test_element_order_matches_observation_spec_names(task):
    """The divisor vector must line up with `ObservationSpec.build` element order.

    `ObsNorm.divisor_vector` tiles the per-feature divisors `history` times,
    which is only correct if `ObservationSpec` stacks history oldest-first with
    features as the INNER loop.  This asserts that layout against the core's own
    `names()` rather than trusting the comment in obs.py.
    """
    spec = kgym.get_task(task)
    ospec = ObservationSpec(features=tuple(spec.obs_features), history=spec.obs_history)
    names = ospec.names()
    divisors = spec.obs_norm.divisor_vector(spec.obs_features, spec.obs_history)
    assert len(names) == len(divisors) == ospec.dim
    for i, nm in enumerate(names):
        feature = nm.split("[")[0]
        assert divisors[i] == pytest.approx(spec.obs_norm.divisor(feature)), (
            f"element {i} named {nm!r} is divided by {divisors[i]}, but feature {feature!r} "
            f"declares {spec.obs_norm.divisor(feature)}")


@pytest.mark.parametrize("task", ALL_TASKS)
def test_observation_layout_is_documented_element_by_element(task):
    env = kgym.KubeGymEnv(task)
    rows = env.observation_description()
    assert len(rows) == env.observation_space.shape[0] == len(env.obs_names)
    assert [r["name"] for r in rows] == env.obs_names
    assert rows[-1]["name"] == "episode_progress"
    for r in rows[:-1]:
        assert r["feature"] in FEATURES
        assert r["divisor"] > 0


@pytest.mark.parametrize("task", ALL_TASKS)
def test_every_feature_is_a_core_feature(task):
    """No wrapper-invented features: the fairness guarantee is inherited, not re-argued."""
    spec = kgym.get_task(task)
    assert set(spec.obs_features) <= set(FEATURES), (
        set(spec.obs_features) - set(FEATURES))
    assert set(spec.obs_features) <= set(FEATURE_SCALE_KIND), (
        "a feature without a declared normalisation would enter the observation unscaled")


@pytest.mark.parametrize("task", ALL_TASKS)
def test_target_is_observable_so_delta_actions_are_well_posed(task):
    """`delta` acts on the current target, so the policy must be able to see it."""
    assert "target" in kgym.get_task(task).obs_features


# ---------------------------------------------------------------------------
# bounds and clipping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task", ALL_TASKS)
def test_observation_stays_inside_the_declared_box(task):
    env = kgym.KubeGymEnv(task)
    for mode in ("delta", "absolute"):
        e = kgym.KubeGymEnv(task, action_mode=mode)
        for a in range(e.action_space.n):
            for o in rollout(e, e.n_control_steps, action=a):
                assert e.observation_space.contains(o), (o.min(), o.max())
    assert env.observation_space.low.min() == 0.0


@pytest.mark.parametrize("task", ALL_TASKS)
def test_no_clipping_saturation_on_the_shipped_fixtures(task):
    """Clipping is lossy; assert it never actually bites on the shipped tasks.

    If this ever fails, the observation has started saturating and the policy has
    stopped seeing the queue grow -- raise `clip` or the relevant scale and say
    so in the env card, do not delete the test.
    """
    clip = kgym.get_task(task).obs_norm.clip
    worst = 0.0
    for mode in ("delta", "absolute"):
        env = kgym.KubeGymEnv(task, action_mode=mode)
        for a in range(env.action_space.n):
            for o in rollout(env, env.n_control_steps, action=a):
                worst = max(worst, float(np.max(o)))
    assert worst < clip, f"observation reached the clip ceiling {clip} (max seen {worst})"
    assert worst > 0.0


@pytest.mark.parametrize("task", ALL_TASKS)
def test_episode_progress_element_runs_zero_to_one(task):
    env = kgym.KubeGymEnv(task)
    obs, _ = env.reset(seed=0)
    assert obs[-1] == pytest.approx(0.0)
    seen = [float(obs[-1])]
    for i in range(env.n_control_steps):
        o, _, term, trunc, _ = env.step(0)
        seen.append(float(o[-1]))
        if term or trunc:
            break
    assert seen[-1] == pytest.approx(1.0)
    assert seen == sorted(seen), "episode progress must be monotone"


# ---------------------------------------------------------------------------
# no privileged information
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task", ALL_TASKS)
def test_observation_is_invariant_to_mutating_request_ground_truth(task):
    """Rewrite every request's ground-truth work; the observation must not move.

    `demand` is the ground truth a controller is not allowed to see (a controller
    that reads it must declare `uses_privileged_info` and is excluded from
    Pareto fronts by default).  A `ClusterState` is a snapshot of computed
    floats, so if the wrapper builds its observation only from `sim.hist` the
    mutation below is invisible.  If the wrapper ever kept a live reference into
    a `Request`, this fails.
    """
    env = kgym.KubeGymEnv(task)
    env.reset(seed=0)
    for _ in range(env.n_control_steps // 2):
        env.step(1)
    hist = env.sim.hist
    before = env.observe_from_history(hist, env._step_i)

    touched = 0
    for r in env.sim.requests:
        r.demand = float(r.demand) * 7.3 + 11.0     # ground truth, wildly rewritten
        touched += 1
    assert touched > 0
    after = env.observe_from_history(hist, env._step_i)
    assert np.array_equal(before, after), (
        "the observation changed when request ground-truth `demand` was rewritten, so it is "
        "not a pure function of the scrape history")
    assert before.tobytes() == after.tobytes()


@pytest.mark.parametrize("task", ALL_TASKS)
def test_router_inflight_exposes_no_ground_truth(task):
    """The one per-request channel a controller may read must stay work-free."""
    env = kgym.KubeGymEnv(task)
    env.reset(seed=0)
    seen_any = False
    for _ in range(env.n_control_steps):
        _, _, term, trunc, _ = env.step(3 if env.action_space.n > 3 else 0)
        for st in env.sim.hist:
            for entry in st.router_inflight:
                seen_any = True
                extra = set(entry) - ROUTER_INFLIGHT_WHITELIST
                assert not extra, (
                    f"router_inflight gained field(s) {sorted(extra)}; check they carry no "
                    "information about a request's total work before widening the whitelist")
                assert "demand" not in entry and "output_tokens_true" not in entry
                assert "remaining" not in entry
        if term or trunc:
            break
    assert seen_any, "no in-flight requests were ever observed; the test proved nothing"


@pytest.mark.parametrize("task", ALL_TASKS)
def test_normalisation_uses_no_episode_statistics(task):
    """Divisors must be fixed before the episode runs, not derived from it.

    Two independent checks. First, the divisor vector is byte-identical before
    and after a rollout. Second -- the one that would catch an adaptive
    normaliser that recomputed scales per step -- re-deriving the observation at
    step `i` from the FIRST `i+1` scrapes of a longer episode reproduces the
    observation that was emitted at step `i`, which cannot hold if the scaling
    depended on anything later.
    """
    env = kgym.KubeGymEnv(task)
    d_before = env._divisors.copy()
    env.reset(seed=0)
    emitted = []
    for _ in range(env.n_control_steps):
        o, _, term, trunc, _ = env.step(0)
        emitted.append(o.copy())
        if term or trunc:
            break
    assert np.array_equal(d_before, env._divisors)

    hist = list(env.sim.hist)
    for i, o in enumerate(emitted, start=1):
        # hist[0] is the t=0 scrape, so step i corresponds to hist[:i+1]
        recon = env.observe_from_history(hist[:i + 1], i)
        assert np.array_equal(recon, o), f"observation at step {i} depends on later scrapes"


@pytest.mark.parametrize("task", ALL_TASKS)
def test_booting_replicas_contribute_no_metrics(task):
    """A booting replica has no metrics endpoint on a real cluster, nor here.

    The observation may show that a boot was *requested* (`n_booting`, `target`),
    which is the controller's own actuation state, but never a metric from it.
    """
    env = kgym.KubeGymEnv(task, action_mode="absolute")
    env.reset(seed=0)
    saw_booting = False
    for _ in range(env.n_control_steps):
        _, _, term, trunc, info = env.step(env.action_space.n - 1)   # scale to the ceiling
        st = env.sim.hist[-1]
        # The invariant holds on every step, boot in progress or not: the metric
        # lists cover READY replicas only.
        for name in env.service_model.metric_names():
            assert len(st.get(name)) == st.n_ready, (
                f"{name} has {len(st.get(name))} entries with n_ready={st.n_ready} and "
                f"n_booting={st.n_booting}: a booting replica leaked a metric")
        saw_booting = saw_booting or st.n_booting > 0
        if term or trunc:
            break
    if env.cfg.f("replica_cold_start_s") > env.control_interval_s:
        assert saw_booting, "no boot was observed; the boot-in-progress case did not run"
    else:
        assert not saw_booting, (
            f"{task}: cold start is shorter than the control interval, so n_booting is expected "
            "to be identically zero -- see test_actuation_lag_visibility")


@pytest.mark.parametrize("task", ALL_TASKS)
def test_actuation_lag_visibility(task):
    """Whether the policy can SEE a scale-up in flight, per task, as data.

    On the LLM tasks the measured 31.9 s cold start spans 2.13 control
    intervals, so `n_booting` is a live signal. On `RequestService-Poisson` the
    PLACEHOLDER 2 s cold start completes inside one 15 s interval, so the four
    `n_booting[t-k]` elements of the observation are identically zero and the
    actuation lag is invisible: that task is effectively a no-lag control
    problem. It is a property of a placeholder constant, so it is pinned here
    and stated in the env card rather than left for a reader to discover from a
    flat learning curve.
    """
    env = kgym.KubeGymEnv(task, action_mode="absolute")
    idx = [i for i, nm in enumerate(env.obs_names) if nm.startswith("n_booting")]
    assert len(idx) == kgym.get_task(task).obs_history
    nonzero = 0
    n_steps = 0
    for a in range(env.action_space.n):
        env.reset(seed=0)
        while True:
            o, _, term, trunc, _ = env.step(a)
            n_steps += 1
            nonzero += int(np.any(o[idx] > 0))
            if term or trunc:
                break
    lagged = env.cfg.f("replica_cold_start_s") > env.control_interval_s
    if lagged:
        assert nonzero > 0, f"{task}: n_booting never left zero despite a super-interval boot"
    else:
        assert nonzero == 0, (
            f"{task}: n_booting was non-zero on {nonzero}/{n_steps} steps, but the cold start "
            f"({env.cfg.f('replica_cold_start_s')} s) is shorter than the control interval "
            f"({env.control_interval_s} s). The env card documents these elements as dead; "
            "update it.")


# ---------------------------------------------------------------------------
# the normaliser itself
# ---------------------------------------------------------------------------
def test_undeclared_feature_has_no_default_divisor():
    norm = ObsNorm(replica_scale=4.0, concurrency_scale=32.0, per_replica_concurrency=8.0,
                   arrivals_scale=16.0)
    with pytest.raises(KeyError, match="no normalisation declared"):
        norm.divisor("some_new_feature")


def test_nonpositive_scale_is_rejected():
    for bad in ({"replica_scale": 0.0}, {"arrivals_scale": -1.0},
                {"concurrency_scale": float("inf")}):
        kw = dict(replica_scale=4.0, concurrency_scale=32.0, per_replica_concurrency=8.0,
                  arrivals_scale=16.0)
        kw.update(bad)
        with pytest.raises(ValueError, match="must be finite and > 0"):
            ObsNorm(**kw)


def test_normalise_clips_both_ends():
    raw = np.asarray([-5.0, 0.0, 1.0, 1e6], dtype=np.float32)
    div = np.asarray([1.0, 1.0, 2.0, 1.0], dtype=np.float32)
    out = normalise(raw, div, clip=10.0)
    assert out.dtype == np.float32
    assert list(out) == [0.0, 0.0, 0.5, 10.0]


def test_all_core_features_have_a_declared_scale():
    """A new core feature must be given a scale before a task can use it."""
    missing = sorted(set(FEATURES) - set(FEATURE_SCALE_KIND))
    assert not missing, (
        f"kubegym.core.state.FEATURES gained {missing} with no entry in "
        "kubegym.gym.obs.FEATURE_SCALE_KIND")
