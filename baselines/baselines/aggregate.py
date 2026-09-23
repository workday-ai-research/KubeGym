"""Aggregate the per-episode test rows into the tables the paper prints.

WHAT DISPERSION MEANS HERE, AND WHAT IT DOES NOT
-----------------------------------------------
Two independent sources of variation exist on the corpus tasks and they are kept
separate throughout:

* **episodes** -- different traces from the same family and split.  Every method
  faces the identical set, so a method-vs-method contrast is *paired*.
* **training seeds** -- only for the learners.  Five per (algorithm, family).

There is deliberately **no work-seed dispersion**: for corpus-backed tasks the
env `seed` does not redraw the offered load, so a work-seed replicate is a
byte-identical rerun.  Pooling it with either of the above would manufacture
degrees of freedom.  The other kind of variance -- where `seed` *does* redraw
work -- is measured separately on the builtin `llm_serving` tasks in
`llm_side.py` and never pooled with these numbers.

RANKING RULE
------------
`pairwise_comparisons.csv` reports, for every ordered pair of methods within a
family, the paired difference in episode cost with a percentile bootstrap CI over
episodes, plus the learner's seed dispersion.  A pair is marked
`within_noise = True` when the 95 % CI of the paired mean difference contains
zero, or when the point difference is smaller than the relevant seed dispersion.
**Pairs marked `within_noise` are not ranked anywhere**, and the summary table
carries the flag rather than an ordering.

HEAVY TAILS
-----------
Corpus episodes span 1 to ~45,000 requests per hour, so an episode mean is
routinely dominated by one or two traces.  Both the mean and the median are
reported, and `top_episode_cost_share` records the fraction of a method's total
family cost contributed by its single worst episode, so a dominated metric is
visible rather than implied.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

METRICS = ["cost_total", "return", "cost_slo", "cost_resource", "cost_churn",
           "cost_slo_drain", "cost_resource_drain", "cost_slo_unfinished",
           "replica_seconds", "slo_attainment", "slo_attainment_feasible_floor",
           "churn_replicas", "scale_events", "latency_mean_s", "latency_p95_s",
           "latency_p99_s", "queue_delay_p95_s", "target_mean", "target_max",
           "target_min",
           "n_obs_saturated_steps", "n_actions_clamped"]

BOOT_N = 10_000
BOOT_SEED = 20260829


def load_rows(pattern: str = "out/eval_*.csv") -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no evaluation rows matching {pattern!r}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df


def tag_tuned_static(df: pd.DataFrame, out_dir: str = "out") -> pd.DataFrame:
    """Record which `static-k` row is the dev-selected `static` baseline."""
    sel: Dict[str, int] = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "tuned_*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        sel[d["family"]] = int(json.loads(
            d["methods"]["static"]["selected_params"].replace("'", '"'))["k"]
            if isinstance(d["methods"]["static"]["selected_params"], str)
            else d["methods"]["static"]["selected_params"]["k"])
    df["dev_selected_static_k"] = df["family"].map(sel)
    return df


# ---------------------------------------------------------------------------
# per-method summary
# ---------------------------------------------------------------------------
def _seed_collapse(g: pd.DataFrame) -> pd.DataFrame:
    """Mean over training seeds, per episode -- removes seed noise from pairing."""
    return g.groupby("episode", as_index=False)[METRICS].mean()


def summarise(df: pd.DataFrame, by_regime: bool = False) -> pd.DataFrame:
    keys = ["family", "method"] + (["regime"] if by_regime else [])
    out: List[Dict[str, Any]] = []
    for k, g in df.groupby(keys, sort=True):
        kd = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        n_seeds = int(g["policy_seed"].nunique())
        has_seeds = n_seeds > 1 or (g["policy_seed"] >= 0).any()
        per_ep = _seed_collapse(g)
        row: Dict[str, Any] = dict(kd)
        row["method_class"] = g["method_class"].iloc[0]
        row["uses_privileged_info"] = bool(g["uses_privileged_info"].iloc[0])
        row["is_adaptation"] = bool(g["is_adaptation"].iloc[0])
        row["n_episodes"] = int(per_ep.shape[0])
        row["n_policy_seeds"] = n_seeds if has_seeds else 0
        row["n_rows"] = int(g.shape[0])
        for m in METRICS:
            v = per_ep[m].to_numpy(dtype=float)
            row[f"{m}_mean"] = float(np.nanmean(v)) if len(v) else np.nan
            row[f"{m}_median"] = float(np.nanmedian(v)) if len(v) else np.nan
            row[f"{m}_sd_episodes"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else np.nan
            row[f"{m}_iqr_episodes"] = (float(np.nanpercentile(v, 75) - np.nanpercentile(v, 25))
                                        if len(v) > 1 else np.nan)
        # seed dispersion: sd across per-seed episode means
        if has_seeds and n_seeds > 1:
            per_seed = g.groupby("policy_seed")[METRICS].mean()
            for m in METRICS:
                row[f"{m}_sd_seeds"] = float(per_seed[m].std(ddof=1))
                row[f"{m}_range_seeds"] = float(per_seed[m].max() - per_seed[m].min())
        else:
            for m in METRICS:
                row[f"{m}_sd_seeds"] = np.nan
                row[f"{m}_range_seeds"] = np.nan
        tot = per_ep["cost_total"].to_numpy(dtype=float)
        row["top_episode_cost_share"] = (float(np.nanmax(tot) / np.nansum(tot))
                                         if np.nansum(tot) > 0 else np.nan)
        row["worst_episode"] = (int(per_ep.loc[np.nanargmax(tot), "episode"])
                                if len(tot) else -1)
        # A controller whose target never moves off the floor is static-k_min in
        # disguise -- the degenerate-configuration failure the source project's
        # tuning protocol caught (an unreachable metric target silently turning
        # HPA into static-1). Flagged rather than left for a reader to infer from
        # two identical objective values.
        row["is_constant_target"] = bool(
            np.allclose(per_ep["target_max"], per_ep["target_min"], atol=1e-9)
            and np.allclose(per_ep["target_max"], per_ep["target_max"].iloc[0], atol=1e-9))
        row["constant_target_value"] = (float(per_ep["target_max"].iloc[0])
                                        if row["is_constant_target"] else np.nan)
        row["mean_frac_episodes_no_scaling"] = float(
            (per_ep["scale_events"] == 0).mean())
        row["calibrated"] = bool(g["calibrated"].iloc[0]) if "calibrated" in g else False
        row["uncalibrated_required_fields"] = (
            g["uncalibrated_required_fields"].iloc[0]
            if "uncalibrated_required_fields" in g else "")
        row["tuning_budget_cap_env_steps"] = int(g["tuning_budget_cap_env_steps"].iloc[0])
        row["tuning_budget_spent_env_steps"] = int(g["tuning_budget_spent_env_steps"].iloc[0])
        row["n_configs_evaluated"] = int(g["n_configs_evaluated"].iloc[0])
        row["params"] = g["params"].iloc[0]
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# paired comparisons
# ---------------------------------------------------------------------------
def _boot_ci(d: np.ndarray, stat, n: int = BOOT_N, seed: int = BOOT_SEED,
             alpha: float = 0.05) -> Tuple[float, float]:
    if len(d) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    s = stat(d[idx], axis=1)
    return (float(np.percentile(s, 100 * alpha / 2)),
            float(np.percentile(s, 100 * (1 - alpha / 2))))


def pairwise(df: pd.DataFrame, methods: Optional[Sequence[str]] = None,
             metric: str = "cost_total") -> pd.DataFrame:
    """Paired differences between every ordered method pair, per family."""
    out: List[Dict[str, Any]] = []
    for family, gf in df.groupby("family", sort=True):
        avail = sorted(gf["method"].unique())
        ms = [m for m in (methods or avail) if m in avail]
        per_ep: Dict[str, pd.Series] = {}
        seed_sd: Dict[str, float] = {}
        for m in ms:
            g = gf[gf["method"] == m]
            per_ep[m] = _seed_collapse(g).set_index("episode")[metric]
            if g["policy_seed"].nunique() > 1:
                seed_sd[m] = float(g.groupby("policy_seed")[metric].mean().std(ddof=1))
            else:
                seed_sd[m] = 0.0
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                common = per_ep[a].index.intersection(per_ep[b].index)
                d = (per_ep[a].loc[common] - per_ep[b].loc[common]).to_numpy(dtype=float)
                d = d[np.isfinite(d)]
                if len(d) < 2:
                    continue
                lo, hi = _boot_ci(d, np.mean)
                mlo, mhi = _boot_ci(d, np.median)
                sd_ep = float(np.std(d, ddof=1))
                noise = max(seed_sd[a], seed_sd[b])
                ci_covers_zero = bool(lo <= 0.0 <= hi)
                below_seed_noise = bool(abs(float(np.mean(d))) < noise)
                out.append({
                    "family": family, "metric": metric, "method_a": a, "method_b": b,
                    "n_paired_episodes": int(len(d)),
                    "mean_diff_a_minus_b": float(np.mean(d)),
                    "median_diff_a_minus_b": float(np.median(d)),
                    "sd_diff_episodes": sd_ep,
                    "ci95_mean_lo": lo, "ci95_mean_hi": hi,
                    "ci95_median_lo": mlo, "ci95_median_hi": mhi,
                    "seed_sd_a": seed_sd[a], "seed_sd_b": seed_sd[b],
                    "ci_covers_zero": ci_covers_zero,
                    "abs_diff_below_seed_sd": below_seed_noise,
                    "within_noise": bool(ci_covers_zero or below_seed_noise),
                    "n_wins_a": int((d < 0).sum()), "n_wins_b": int((d > 0).sum()),
                    "n_ties": int((d == 0).sum()),
                })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# learning curves
# ---------------------------------------------------------------------------
def learning_curves(rl_dir: str = "out/rl") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for f in sorted(glob.glob(os.path.join(rl_dir, "train_*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        for s in d["seeds"]:
            for pt in s["curve"]:
                rows.append({"algo": d["algo"], "family": d["family"],
                             "seed": s["seed"], "steps": pt["steps"],
                             "dev_cost_total": pt["mean_cost_total"],
                             "dev_return": pt["mean_return"],
                             "dev_slo_attainment": pt["mean_slo_attainment"],
                             "dev_replica_seconds": pt["mean_replica_seconds"],
                             "action_mode": d["action_mode_selected"]})
    return pd.DataFrame(rows)


def budget_table(out_dir: str = "out", rl_dir: str = "out/rl") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for f in sorted(glob.glob(os.path.join(out_dir, "tuned_*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        for m, md in d["methods"].items():
            L = md["ledger"]
            rows.append({"family": d["family"], "method": m, "method_class": "analytic",
                         "cap_env_steps": L["cap_env_steps"],
                         "spent_env_steps": L["spent_env_steps"],
                         "pct_of_cap": 100.0 * L["spent_env_steps"] / L["cap_env_steps"],
                         "n_trials": L["n_trials"],
                         "n_configs_in_full_grid": md["n_configs_in_full_grid"],
                         "n_configs_evaluated": md["n_configs_evaluated"],
                         "grid_exhaustive": md["grid_exhaustive"],
                         "wall_s": L["wall_s"], "by_phase": json.dumps(L["by_phase"])})
    for f in sorted(glob.glob(os.path.join(rl_dir, "train_*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        L = d["ledger"]
        rows.append({"family": d["family"], "method": d["algo"], "method_class": "learned",
                     "cap_env_steps": L["cap_env_steps"],
                     "spent_env_steps": L["spent_env_steps"],
                     "pct_of_cap": 100.0 * L["spent_env_steps"] / L["cap_env_steps"],
                     "n_trials": L["n_trials"],
                     "n_configs_in_full_grid": len(d["hp_grid"]),
                     "n_configs_evaluated": len(d["search"]),
                     "grid_exhaustive": True,
                     "wall_s": L["wall_s"], "by_phase": json.dumps(L["by_phase"])})
    return pd.DataFrame(rows)


def run(out_dir: str = "out") -> Dict[str, Any]:
    df = load_rows(os.path.join(out_dir, "eval_*.csv"))
    df = tag_tuned_static(df, out_dir)
    df.to_csv(os.path.join(out_dir, "reference_results.csv"), index=False)

    headline = [m for m in ("static", "hpa", "queue", "leading", "forecast",
                            "ppo", "dqn") if m in set(df["method"])]
    s_fam = summarise(df)
    s_fam["scope"] = "family"
    s_reg = summarise(df, by_regime=True)
    s_reg["scope"] = "family_regime"
    summary = pd.concat([s_fam, s_reg], ignore_index=True)
    summary.to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)

    pw = pairwise(df, methods=headline + ["oracle"])
    pw.to_csv(os.path.join(out_dir, "pairwise_comparisons.csv"), index=False)

    lc = learning_curves()
    if len(lc):
        lc.to_csv(os.path.join(out_dir, "learning_curves.csv"), index=False)
    bt = budget_table(out_dir)
    bt.to_csv(os.path.join(out_dir, "tuning_budget.csv"), index=False)
    return {"n_rows": len(df), "n_summary": len(summary), "n_pairs": len(pw),
            "n_curve_points": len(lc), "headline_methods": headline}


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
