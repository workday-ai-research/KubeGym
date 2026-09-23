"""Dev-only tuning of the analytic controllers under an enforced step budget.

THE RULES, ALL ENFORCED IN CODE
-------------------------------
1. **The test split is never opened.**  `runner.assert_not_test` is called before
   any env is built, and it raises rather than warns.
2. **One budget currency for every method**: environment steps taken on the
   train and dev splits.  `runner.BudgetLedger` charges every step and raises
   past the cap.  The cap is the same integer for every method and every family
   (`CAP_ENV_STEPS`), so the parity claim is checkable by reading `spent` against
   `cap_env_steps` in `tuning_ledger.json` -- it is not asserted in prose.
3. **One objective for every method**: `runner.objective`, the environment's own
   mean episode cost.  No method gets a bespoke objective.  In particular the
   `churn` shaping term is included, because a controller tuned for cost+SLO only
   is not optimising what the benchmark scores (`GYM_API.md` section 7.1).
4. **Deterministic selection**: minimum mean cost; ties broken by lower
   replica-seconds, then fewer scale events, then lexicographic parameter JSON.
5. **Subsampling is seeded and blind.**  A grid larger than
   `MAX_CONFIGS_PER_METHOD` is *randomly* sampled at a fixed seed, never
   hand-shortlisted, so no method can be handed the good corners of its space.
6. **Fronts, not points.**  Every trial's full cost decomposition is stored, so
   selection is repeated at three SLO-weight multipliers at zero extra cost.  The
   canonical weighting (multiplier 1.0) is the headline.

THE DEV TUNING SET
------------------
The same episodes for every method, per family, so the comparison is paired.
Where a family's dev split has more traces than `N_DEV_TUNE`, the subset is
stratified by regime (round-robin over regime labels in `trace_id` order) rather
than truncated, because `azure_replay` dev spans five regimes whose difficulty
differs by orders of magnitude and the first 15 trace ids are not a sample of it.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .gym_controllers import REGISTRY, HPAGym
from .obsmap import ObsView
from .runner import (BudgetLedger, FAMILIES, assert_not_test, episode_list,
                     load_manifest, make_env, objective, regime_of, rollout,
                     trace_entries)

#: the shared cap: train+dev environment steps per (method, family)
CAP_ENV_STEPS = 1_200_000
#: dev episodes used for tuning and for RL model selection (identical set)
N_DEV_TUNE = 12
#: cap on sampled configurations where a grid is larger.  The source project's
#: protocol used 60; doubled here so an "undertuned baseline" objection has a
#: number to argue with, and still far inside the shared cap.
MAX_CONFIGS_PER_METHOD = 120
#: seed for blind subsampling of an oversized grid
SUBSAMPLE_SEED = 1234
#: SLO-weight multipliers at which selection is repeated post hoc
W_SLO_MULTIPLIERS = (0.25, 1.0, 4.0)


# ---------------------------------------------------------------------------
# the dev tuning set
# ---------------------------------------------------------------------------
def dev_tuning_episodes(family: str, manifest: Dict[str, Any],
                        n: int = N_DEV_TUNE) -> List[int]:
    entries = trace_entries(manifest, family, "dev")
    if len(entries) <= n:
        return list(range(len(entries)))
    by_regime: Dict[str, List[int]] = {}
    for i, e in enumerate(entries):
        by_regime.setdefault(regime_of(e), []).append(i)
    keys = sorted(by_regime)
    out: List[int] = []
    r = 0
    while len(out) < n:
        progressed = False
        for k in keys:
            if r < len(by_regime[k]):
                out.append(by_regime[k][r])
                progressed = True
                if len(out) == n:
                    break
        if not progressed:
            break
        r += 1
    return sorted(out)


# ---------------------------------------------------------------------------
# grids
# ---------------------------------------------------------------------------
def _product(space: Dict[str, Sequence]) -> List[Dict[str, Any]]:
    keys = sorted(space)
    out: List[Dict[str, Any]] = [{}]
    for k in keys:
        out = [dict(c, **{k: v}) for c in out for v in space[k]]
    return out


def full_grid(method: str, service_model: str = "request_service") -> List[Dict[str, Any]]:
    if method == "hpa":
        return HPAGym.configs(service_model)
    cls = REGISTRY[method]
    return _product(dict(cls.param_space))


def sampled_grid(method: str, cap: int = MAX_CONFIGS_PER_METHOD
                 ) -> Tuple[List[Dict[str, Any]], int, bool]:
    grid = full_grid(method)
    n_full = len(grid)
    if n_full <= cap:
        return grid, n_full, True
    rng = random.Random(SUBSAMPLE_SEED)
    idx = sorted(rng.sample(range(n_full), cap))
    return [grid[i] for i in idx], n_full, False


# ---------------------------------------------------------------------------
# the ridge forecaster, fit on dev only
# ---------------------------------------------------------------------------
def fit_ridge(family: str, episodes: Sequence[int], ledger: BudgetLedger,
              n_lags: int = 4, lam: float = 1.0) -> Dict[str, Any]:
    """One-step-ahead ridge on the controller-visible arrival-rate series.

    Fit from the **dev** tuning episodes only.  The arrival series is exogenous
    -- it does not depend on the controller -- so one static-`k` pass collects
    it, and those steps are charged to the ledger like any other.

    Returns `ar_coef = [c_1 ... c_L, intercept]` in the layout
    `ForecastThresholdGym` expects, plus the in-sample R^2.  If the fit is
    degenerate the caller keeps `ar_coef = None`, the controller falls back to
    Holt and records `ridge_fallback = True`, so an unfitted model can never be
    reported as a fitted one.
    """
    assert_not_test("dev", "ridge fit")
    env = make_env(family, "dev", "absolute")
    view = ObsView(env.unwrapped.task_spec)
    cfg = env.unwrapped.cfg
    ctrl = REGISTRY["static"](view, cfg, action_mode="absolute", k=1)
    series: List[List[float]] = []
    ws = int(env.unwrapped.task_spec.split_work_seeds["dev"][0])
    dt = float(cfg.f("control_interval_s"))
    for ep in episodes:
        obs, _ = env.reset(options={"episode": int(ep), "work_seed": ws})
        ctrl.reset(env)
        s: List[float] = []
        n = 0
        while True:
            sc, t, pg = view.decode(obs)
            s.append(ctrl.arrival_rate(sc, max(dt, 30.0), pg))
            obs, _, term, trunc, _ = env.step(int(ctrl.desired_replicas(obs) - ctrl.k_min))
            n += 1
            if term or trunc:
                break
        ledger.charge(n, "ridge_fit_dev")
        series.append(s)
    env.close()

    X, y = [], []
    for s in series:
        for i in range(n_lags, len(s)):
            X.append(list(reversed(s[i - n_lags:i])) + [1.0])
            y.append(s[i])
    if len(y) < 10 * (n_lags + 1):
        return {"ar_coef": None, "r2": None, "n_obs": len(y),
                "reason": "too few observations for a stable fit"}
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1])
    A[-1, -1] -= lam                                      # do not penalise intercept
    try:
        beta = np.linalg.solve(A, Xa.T @ ya)
    except np.linalg.LinAlgError:
        return {"ar_coef": None, "r2": None, "n_obs": len(y), "reason": "singular normal matrix"}
    pred = Xa @ beta
    ss_res = float(((ya - pred) ** 2).sum())
    ss_tot = float(((ya - ya.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {"ar_coef": [float(b) for b in beta], "r2": r2, "n_obs": len(y),
            "n_lags": n_lags, "ridge_lambda": lam,
            "note": "fit on dev tuning episodes only; layout is [c_1..c_L, intercept]"}


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def _select(trials: List[Dict[str, Any]], w_mult: float = 1.0) -> Dict[str, Any]:
    """Deterministic selection at an SLO-weight multiplier."""
    def key(tr):
        c = tr["mean_terms"]
        # re-weight the SLO family of terms; resource and churn unchanged
        slo = c["cost_slo"] + c["cost_slo_drain"] + c["cost_slo_unfinished"]
        rest = c["cost_resource"] + c["cost_resource_drain"] + c["cost_churn"]
        return (w_mult * slo + rest, c["replica_seconds"], c["scale_events"],
                json.dumps(tr["params"], sort_keys=True))
    return min(trials, key=key)


def sweep_family(family: str, methods: Sequence[str], manifest: Dict[str, Any],
                 cap: int = CAP_ENV_STEPS) -> Dict[str, Any]:
    assert_not_test("dev", "tuning sweep")
    episodes = dev_tuning_episodes(family, manifest)
    entries = trace_entries(manifest, family, "dev")
    env = make_env(family, "dev", "absolute")
    u = env.unwrapped
    view = ObsView(u.task_spec)
    cfg = u.cfg
    ws = int(u.task_spec.split_work_seeds["dev"][0])
    out: Dict[str, Any] = {"family": family, "n_dev_tuning_episodes": len(episodes),
                           "dev_tuning_episodes": episodes,
                           "dev_tuning_regimes": [regime_of(entries[e]) for e in episodes],
                           "methods": {}}
    for method in methods:
        ledger = BudgetLedger(method=method, family=family, cap=cap)
        t0 = time.perf_counter()
        grid, n_full, exhaustive = sampled_grid(method)
        learned: Dict[str, Any] = {}
        if method == "forecast":
            learned = fit_ridge(family, episodes, ledger)
        trials: List[Dict[str, Any]] = []
        for params in grid:
            p = dict(params)
            if method == "forecast" and p.get("forecaster") == "ridge":
                p["ar_coef"] = learned.get("ar_coef")
            ctrl = REGISTRY[method](view, cfg, action_mode="absolute", **p)
            rows = []
            for ep in episodes:
                r = rollout(env, ctrl, ep, ws)
                ledger.charge(r["n_control_steps"], "sweep_dev")
                rows.append(r)
            mean_terms = {k: float(np.mean([r[k] for r in rows])) for k in
                          ("cost_total", "cost_slo", "cost_resource", "cost_churn",
                           "cost_slo_drain", "cost_resource_drain", "cost_slo_unfinished",
                           "replica_seconds", "scale_events", "slo_attainment",
                           "latency_p95_s")}
            trials.append({"params": dict(params), "mean_terms": mean_terms,
                           "objective": objective(rows)})
            ledger.log_trial(dict(params), objective(rows),
                             {"mean_cost_total": mean_terms["cost_total"],
                              "mean_replica_seconds": mean_terms["replica_seconds"],
                              "mean_slo_attainment": mean_terms["slo_attainment"]})
        ledger.wall_s = time.perf_counter() - t0
        sel = {f"w_slo_x{m}": _select(trials, m)["params"] for m in W_SLO_MULTIPLIERS}
        best = _select(trials, 1.0)
        chosen = dict(best["params"])
        if method == "forecast" and chosen.get("forecaster") == "ridge":
            chosen["ar_coef"] = learned.get("ar_coef")
        out["methods"][method] = {
            "selected_params": chosen,
            "selected_objective_dev": best["objective"],
            "selected_mean_terms_dev": best["mean_terms"],
            "selected_by_w_slo_multiplier": sel,
            "n_configs_in_full_grid": n_full,
            "n_configs_evaluated": len(grid),
            "grid_exhaustive": exhaustive,
            "subsample_seed": None if exhaustive else SUBSAMPLE_SEED,
            "uses_privileged_info": REGISTRY[method].uses_privileged_info,
            "is_adaptation": REGISTRY[method].is_adaptation,
            "learned_component": learned or None,
            "ledger": ledger.to_dict(),
            "trials": ledger.trials,
        }
    env.close()
    return out


def run(out_json: str = "out/tuned_params.json",
        methods: Sequence[str] = ("static", "hpa", "queue", "leading", "forecast", "oracle"),
        families: Sequence[str] = FAMILIES) -> Dict[str, Any]:
    manifest = load_manifest()
    res = {"cap_env_steps_per_method_family": CAP_ENV_STEPS,
           "n_dev_tuning_episodes": N_DEV_TUNE,
           "max_configs_per_method": MAX_CONFIGS_PER_METHOD,
           "subsample_seed": SUBSAMPLE_SEED,
           "objective": ("mean episode cost_total on the dev tuning episodes -- the "
                         "environment's own reward, identical for every method"),
           "families": {}}
    for fam in families:
        t0 = time.perf_counter()
        res["families"][fam] = sweep_family(fam, methods, manifest)
        res["families"][fam]["wall_s"] = round(time.perf_counter() - t0, 1)
        print(f"[tune] {fam} done in {time.perf_counter() - t0:.0f} s", flush=True)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    return res


def run_one(family: str, out_json: Optional[str] = None,
            methods: Sequence[str] = ("static", "hpa", "queue", "leading",
                                      "forecast", "oracle")) -> Dict[str, Any]:
    """Tune one family and write `out/tuned_<family>.json` (parallel driver)."""
    manifest = load_manifest()
    t0 = time.perf_counter()
    d = sweep_family(family, methods, manifest)
    d["wall_s"] = round(time.perf_counter() - t0, 1)
    d["cap_env_steps_per_method_family"] = CAP_ENV_STEPS
    d["max_configs_per_method"] = MAX_CONFIGS_PER_METHOD
    d["subsample_seed"] = SUBSAMPLE_SEED
    out_json = out_json or f"out/tuned_{family}.json"
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(d, fh, indent=1, default=str)
    return d


if __name__ == "__main__":
    import sys
    fams = sys.argv[1:] or list(FAMILIES)
    for fam in fams:
        d = run_one(fam)
        for m, md in d["methods"].items():
            print(f"{fam:<13} {m:<9} obj={md['selected_objective_dev']:>10.3f} "
                  f"spent={md['ledger']['spent_env_steps']:>8}/{md['ledger']['cap_env_steps']} "
                  f"cfgs={md['n_configs_evaluated']}/{md['n_configs_in_full_grid']} "
                  f"wall={md['ledger']['wall_s']:.0f}s params={md['selected_params']}",
                  flush=True)
