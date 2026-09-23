"""Cost-model semantics: decomposition, monotonicity, and what the reward reads.

Pinned here:

  * the reported terms sum to the reported total, and the total is exactly
    `-reward`, on every step of every task and both action modes;
  * the episode cost row equals the sum over steps;
  * monotonicity in each term -- more SLO violation must never improve reward,
    more replica-seconds must never improve reward, more churn must never
    improve reward, all else equal;
  * `test_cost_model_source_reads_no_ground_truth` greps `kubegym/gym/cost.py`
    for the ground-truth attribute names, so the claim that the objective is
    implementable on a real cluster from router + orchestrator telemetry is a
    mechanical check and not a promise in a docstring.
"""
from __future__ import annotations

import inspect

import pytest

import kubegym.gym as kgym
from kubegym.gym import cost as cost_mod
from kubegym.gym.cost import TERMS, CostModel, CostWeights

ALL_IDS = kgym.task_ids()
ALL_TASKS = kgym.task_names()


# ---------------------------------------------------------------------------
# decomposition
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_terms_sum_to_total_and_total_is_minus_reward(gym_id):
    import gymnasium
    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    env.reset(seed=4)
    n = 0
    while True:
        _, reward, term, trunc, info = env.step(n % env.action_space.n)
        n += 1
        c = info["cost"]
        assert sum(c[k] for k in TERMS) == pytest.approx(c["total"], rel=1e-12, abs=1e-12)
        assert reward == pytest.approx(-c["total"], rel=1e-12, abs=1e-12)
        assert all(c[k] >= 0.0 for k in TERMS), c
        if term or trunc:
            break
    row = info["episode_cost"]
    for k in TERMS:
        assert row[f"cost_{k}"] == pytest.approx(info["cost_cumulative"][k])
    assert row["cost_total"] == pytest.approx(sum(row[f"cost_{k}"] for k in TERMS))


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_episode_return_equals_the_sum_of_step_rewards(gym_id):
    import gymnasium
    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    env.reset(seed=6)
    total = 0.0
    while True:
        _, reward, term, trunc, info = env.step(0)
        total += reward
        if term or trunc:
            break
    assert info["episode_cost"]["return"] == pytest.approx(total, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_raw_counts_accompany_every_weighted_term(gym_id):
    """A row must record what happened as well as what it was charged."""
    import gymnasium
    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)
    c = info["cost"]
    w = env.cost_model.weights
    assert c["slo"] == pytest.approx(w.slo * c["raw_slo_violations"])
    assert c["resource"] == pytest.approx(w.resource * c["raw_replica_seconds"])
    assert c["churn"] == pytest.approx(w.churn * c["raw_churn_replicas"])
    assert c["raw_replica_seconds"] == pytest.approx(info["replica_seconds_step"])


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_resource_term_is_the_replica_seconds_actually_billed(gym_id):
    """Cross-check the cost model's resource term against the core's own billing."""
    import gymnasium
    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    env.reset(seed=0)
    billed = 0.0
    while True:
        _, _, term, trunc, info = env.step(1 % env.action_space.n)
        billed += info["cost"]["raw_replica_seconds"]
        if term or trunc:
            break
    drain = info["post_horizon_drain"]
    total = billed + drain["replica_seconds"]
    assert total == pytest.approx(info["episode_summary"]["replica_seconds"], rel=1e-9)


# ---------------------------------------------------------------------------
# monotonicity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task", ALL_TASKS)
def test_cost_is_monotone_in_every_term(task):
    """All else equal, more of any bad thing must not improve the reward."""
    spec = kgym.get_task(task)
    cm = CostModel(spec.load_config(), spec.weights, slo_profile=spec.slo_profile)
    base = dict(slo_violations=3, replica_seconds=60.0, churn_replicas=1,
                slo_violations_drain=1, replica_seconds_drain=10.0, unfinished_requests=0)
    t0 = cm.decompose(**base)["total"]
    for field, bump in (("slo_violations", 1), ("replica_seconds", 15.0),
                        ("churn_replicas", 1), ("slo_violations_drain", 1),
                        ("replica_seconds_drain", 5.0), ("unfinished_requests", 1)):
        worse = dict(base)
        worse[field] = worse[field] + bump
        t1 = cm.decompose(**worse)["total"]
        assert t1 >= t0, f"increasing {field} lowered the cost"
        assert -t1 <= -t0, f"increasing {field} improved the reward"
    # and strictly worse wherever the weight is non-zero
    if spec.weights.slo > 0:
        assert cm.decompose(**{**base, "slo_violations": 4})["total"] > t0
    if spec.weights.resource > 0:
        assert cm.decompose(**{**base, "replica_seconds": 75.0})["total"] > t0
    if spec.weights.churn > 0:
        assert cm.decompose(**{**base, "churn_replicas": 2})["total"] > t0


@pytest.mark.parametrize("task", ALL_TASKS)
def test_cost_is_zero_only_when_nothing_happened(task):
    spec = kgym.get_task(task)
    cm = CostModel(spec.load_config(), spec.weights, slo_profile=spec.slo_profile)
    assert cm.decompose()["total"] == 0.0
    assert cm.decompose(replica_seconds=1.0)["total"] > 0.0


def test_negative_weights_are_refused():
    spec = kgym.get_task("LLMServing-S1")
    for bad in ({"slo": -1.0}, {"resource": -0.1}, {"churn": -5.0}):
        kw = {"slo": 0.1, "resource": 0.01, "churn": 0.05}
        kw.update(bad)
        with pytest.raises(ValueError, match="must be >= 0"):
            CostWeights(**kw)


def test_zero_weight_makes_a_term_inert_but_still_reported():
    spec = kgym.get_task("LLMServing-S1")
    cm = CostModel(spec.load_config(), CostWeights(slo=0.0, resource=1.0, churn=0.0),
                   slo_profile=spec.slo_profile)
    d = cm.decompose(slo_violations=99, replica_seconds=2.0, churn_replicas=7)
    assert d["slo"] == 0.0 and d["churn"] == 0.0
    assert d["raw_slo_violations"] == 99.0 and d["raw_churn_replicas"] == 7.0
    assert d["total"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# what the reward is allowed to read
# ---------------------------------------------------------------------------
def test_cost_model_source_reads_no_ground_truth():
    """The objective must be computable on the real cluster, so it reads no `demand`.

    A source-level check on purpose: it catches the mistake at the moment
    someone writes `r.demand` into a new SLO rule, which a behavioural test
    would only catch if that rule happened to fire.
    """
    src = inspect.getsource(cost_mod)
    body = "\n".join(line for line in src.splitlines()
                     if not line.lstrip().startswith("#"))
    # Strip the module docstring, which names these attributes in order to
    # promise it does not read them.
    body = body.split('"""', 2)[-1]
    for forbidden in (".demand", ".remaining", ".output_tokens_true", ".progress"):
        assert forbidden not in body, (
            f"kubegym/gym/cost.py reads {forbidden}, which is ground-truth work a real "
            "deployment cannot observe. The reward would stop being implementable on the "
            "cluster the physics were measured on.")
    assert ".ttft" in body or "ttft" in body
    assert cost_mod.CostModel(
        kgym.get_task("LLMServing-S1").load_config(),
        kgym.get_task("LLMServing-S1").weights,
        slo_profile="llm").provenance()["reads_ground_truth_demand"] is False


def test_slo_profile_must_supply_its_keys():
    spec = kgym.get_task("LLMServing-S1")
    cfg = spec.load_config()
    with pytest.raises(KeyError, match="missing"):
        CostModel(cfg, spec.weights, slo_profile="request_service")
    with pytest.raises(KeyError, match="unknown slo_profile"):
        CostModel(cfg, spec.weights, slo_profile="made_up")


def test_slo_targets_come_from_the_config_and_carry_their_flag():
    spec = kgym.get_task("LLMServing-S1")
    cfg = spec.load_config()
    cm = CostModel(cfg, spec.weights, slo_profile="llm")
    assert cm.slo == {k: float(v) for k, v in cfg.f("slo").items()}
    p = cm.provenance()
    assert p["slo_targets_calibrated"] is False, (
        "the config's `slo` field is a labelled placeholder; if this flips, re-read the config")
    assert p["weights_calibrated"] is False
    assert "DESIGN CHOICE" in p["weights_note"]


# ---------------------------------------------------------------------------
# scoring windows
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_each_request_is_scored_at_most_once(gym_id):
    import gymnasium
    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    env.reset(seed=0)
    scored_total = 0
    while True:
        _, _, term, trunc, info = env.step(0)
        scored_total += info["slo"]["reasons_step"]["n_scored"]
        if term or trunc:
            break
    d = info["post_horizon_drain"]
    n_req = info["episode_summary"]["n_requests"]
    assert len(env._scored) <= n_req
    assert scored_total + d["slo_reasons"]["n_scored"] + d["unfinished_requests"] <= n_req
    # every request is accounted for exactly once by the end of the episode
    assert len(env._scored) + d["unfinished_requests"] == n_req


def test_pending_and_completed_scoring_do_not_double_count():
    """A request scored as a pending hard violation is not scored again on completion."""
    spec = kgym.get_task("RequestService-Poisson")
    cm = CostModel(spec.load_config(), spec.weights, slo_profile=spec.slo_profile)

    class R:
        def __init__(self):
            self.seq = 1
            self.t_arrival = 0.0
            self.t_done = None
            self.t_first_token = None
            self.t_admit = None
        latency = None
        queue_delay = None
        mean_tbt = None
        ttft = None

    r = R()
    scored = set()
    n, newly, _ = cm.score_window([r], 0.0, cm.hard_s + 1.0, scored)
    assert n == 1 and newly == [1]
    scored.update(newly)
    r.t_done = cm.hard_s + 2.0
    n2, newly2, _ = cm.score_window([r], cm.hard_s + 1.0, cm.hard_s + 3.0, scored)
    assert n2 == 0 and newly2 == []


@pytest.mark.parametrize("gym_id", ALL_IDS)
def test_drain_terms_are_charged_only_at_the_final_step(gym_id):
    import gymnasium
    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    env.reset(seed=0)
    for i in range(env.n_control_steps):
        _, _, term, trunc, info = env.step(0)
        last = (i == env.n_control_steps - 1)
        c = info["cost"]
        if not last:
            assert c["slo_drain"] == 0.0 and c["resource_drain"] == 0.0
            assert c["slo_unfinished"] == 0.0
            assert "post_horizon_drain" not in info
        else:
            assert info["post_horizon_drain"]["charged_to_reward"] is True
            assert c["resource_drain"] > 0.0, (
                "the drain billed nothing, so charging it is untested here")


def test_unpriced_tail_is_measurably_cheaper_when_the_drain_is_not_charged():
    """The reason the canonical tasks charge the drain: not charging it is gameable."""
    import gymnasium

    def run(include):
        env = kgym.KubeGymEnv("LLMServing-S1", action_mode="absolute",
                              include_drain_in_reward=include)
        env.reset(seed=0)
        tot = 0.0
        while True:
            _, r, term, trunc, info = env.step(0)     # hold at n_replicas_min
            tot += r
            if term or trunc:
                break
        return tot, info

    charged, info_c = run(True)
    free, info_f = run(False)
    assert free > charged, "dropping the drain cost must make the same policy look better"
    assert info_f["post_horizon_drain"]["charged_to_reward"] is False
    assert info_f["post_horizon_drain"]["replica_seconds"] > 0.0
    assert (free - charged) == pytest.approx(
        info_c["cost"]["resource_drain"] + info_c["cost"]["slo_drain"]
        + info_c["cost"]["slo_unfinished"], rel=1e-9)
