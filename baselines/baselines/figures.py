"""The three shipped figures: learning curves, cost/SLO trade-off, per-family.

One figure per point being made:

1. `learning_curves.png`   -- did the learners learn, and to what level?
2. `cost_slo_tradeoff.png` -- where does each method sit against the trivial
                              Static-k front and against the irreducible SLO floor?
3. `per_family_breakdown.png` -- per family, which separations survive episode
                              dispersion and which do not.

Panel-level colour is threaded: a method keeps one colour in all three figures.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILY_ORDER = ["constant", "diurnal", "variable", "burst", "azure_replay"]
FAMILY_LABEL = {"constant": "Constant", "diurnal": "Diurnal", "variable": "Variable",
                "burst": "Burst", "azure_replay": "Azure replay"}
METHOD_LABEL = {"static": "Static-k (tuned)", "hpa": "Kubernetes HPA",
                "queue": "Queue-concurrency", "leading": "Leading-indicator",
                "forecast": "Forecast-threshold", "ppo": "PPO", "dqn": "DQN",
                "oracle": "Oracle (privileged)"}
DEPLOYABLE = ["static", "hpa", "queue", "leading", "forecast", "ppo", "dqn"]

#: one colour per method, reused across every figure
COLORS = {
    "static": "#8c8c8c", "hpa": "#1f4e79", "queue": "#2e8b7a",
    "leading": "#b8860b", "forecast": "#7b3f9d", "ppo": "#c0392b",
    "dqn": "#e07b39", "oracle": "#000000",
}
MARKERS = {"static": "s", "hpa": "o", "queue": "^", "leading": "v",
           "forecast": "D", "ppo": "P", "dqn": "X", "oracle": "*"}

BOOT_N = 5000
BOOT_SEED = 20260829


def _boot_ci_mean(v: np.ndarray, n: int = BOOT_N, seed: int = BOOT_SEED):
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    s = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# ---------------------------------------------------------------------------
# figure 1: learning curves
# ---------------------------------------------------------------------------
def analytic_curve_reference(out_dir: str = "out", rl_dir: str = "out/rl"
                             ) -> pd.DataFrame:
    """Analytic dev objectives on **the same episodes the curves use**.

    The learning curve is evaluated on `train_rl.N_DEV_CURVE` dev episodes, a
    subset of the 12 used for selection, so a reference line taken from the
    12-episode tuning objective would not be on the same basis as the curve.
    This recomputes the analytic controllers on exactly the curve episodes.
    """
    from .gym_controllers import REGISTRY
    from .obsmap import ObsView
    from .runner import make_env, objective, rollout
    rows: List[Dict[str, Any]] = []
    for fam in FAMILY_ORDER:
        tp = os.path.join(out_dir, f"tuned_{fam}.json")
        if not os.path.isfile(tp):
            continue
        with open(tp) as fh:
            tuned = json.load(fh)
        eps: Optional[Sequence[int]] = None
        for algo in ("ppo", "dqn"):
            p = os.path.join(rl_dir, f"train_{algo}_{fam}.json")
            if os.path.isfile(p):
                with open(p) as fh:
                    eps = json.load(fh)["dev_episodes_curve"]
                break
        if eps is None:
            continue
        env = make_env(fam, "dev", "absolute")
        u = env.unwrapped
        view = ObsView(u.task_spec)
        ws = int(u.task_spec.split_work_seeds["dev"][0])
        for m, md in tuned["methods"].items():
            ctrl = REGISTRY[m](view, u.cfg, action_mode="absolute",
                               **md["selected_params"])
            rs = [rollout(env, ctrl, ep, ws) for ep in eps]
            rows.append({"family": fam, "method": m, "n_episodes": len(eps),
                         "dev_cost_total": objective(rs),
                         "uses_privileged_info": REGISTRY[m].uses_privileged_info})
        env.close()
    return pd.DataFrame(rows)


def fig_learning_curves(lc: pd.DataFrame, ref: pd.DataFrame,
                        path: str = "out/learning_curves.png") -> str:
    fams = [f for f in FAMILY_ORDER if f in set(lc["family"])]
    fig, axes = plt.subplots(1, len(fams), figsize=(2.05 * len(fams), 2.5),
                             squeeze=False)
    axes = axes[0]
    for j, (ax, fam) in enumerate(zip(axes, fams)):
        g = lc[lc["family"] == fam]
        for algo in ("ppo", "dqn"):
            ga = g[g["algo"] == algo]
            if not len(ga):
                continue
            piv = ga.pivot_table(index="steps", columns="seed", values="dev_cost_total")
            x = piv.index.to_numpy() / 1000.0
            mu = piv.mean(axis=1).to_numpy()
            lo = piv.min(axis=1).to_numpy()
            hi = piv.max(axis=1).to_numpy()
            ax.fill_between(x, lo, hi, color=COLORS[algo], alpha=0.18, linewidth=0)
            ax.plot(x, mu, color=COLORS[algo], linewidth=1.5,
                    marker=MARKERS[algo], markersize=2.8,
                    label=f"{METHOD_LABEL[algo]} (n=5 seeds)")
        rg = ref[(ref["family"] == fam) & (~ref["uses_privileged_info"])]
        if len(rg):
            best = rg.loc[rg["dev_cost_total"].idxmin()]
            ax.axhline(best["dev_cost_total"], color=COLORS[best["method"]],
                       linestyle="--", linewidth=1.1)
            st = rg[rg["method"] == "static"]
            if len(st):
                ax.axhline(float(st["dev_cost_total"].iloc[0]), color=COLORS["static"],
                           linestyle=":", linewidth=1.1)
        ax.set_title(FAMILY_LABEL[fam], loc="left")
        ax.set_xlabel("training steps (thousands)")
        if j == 0:
            ax.set_ylabel("dev episode cost")
        ax.margins(0.04)
    return fig, axes, path


# ---------------------------------------------------------------------------
# figure 2: cost / SLO trade-off
# ---------------------------------------------------------------------------
def fig_tradeoff(df: pd.DataFrame, path: str = "out/cost_slo_tradeoff.png"):
    fams = [f for f in FAMILY_ORDER if f in set(df["family"])]
    fig, axes = plt.subplots(1, len(fams), figsize=(2.15 * len(fams), 2.6),
                             squeeze=False)
    axes = axes[0]
    for j, (ax, fam) in enumerate(zip(axes, fams)):
        g = df[df["family"] == fam]
        front = g[g["method_class"] == "static_front"]
        if len(front):
            fr = front.groupby("method").agg(
                rs=("replica_seconds", "mean"), slo=("slo_attainment", "mean"),
                k=("params", lambda s: json.loads(s.iloc[0])["k"])).sort_values("k")
            ax.plot(fr["rs"] / 1000.0, fr["slo"], color="#c9c9c9", linewidth=1.0,
                    marker="o", markersize=2.0, zorder=1)
            for lab, i in (("k=1", 0), ("k=2", 1), ("k=4", 3)):
                if i < len(fr):
                    ax.annotate(lab, (fr["rs"].iloc[i] / 1000.0, fr["slo"].iloc[i]),
                                textcoords="offset points", xytext=(2, -9),
                                color="#8c8c8c", fontsize=6)
        floor = g["slo_attainment_feasible_floor"].mean()
        if np.isfinite(floor):
            ax.axhline(floor, color="#b03a2e", linestyle="--", linewidth=0.9,
                       zorder=0)
        for m in DEPLOYABLE + ["oracle"]:
            gm = g[g["method"] == m]
            if not len(gm):
                continue
            per_ep = gm.groupby("episode")[["replica_seconds", "slo_attainment"]].mean()
            x, y = per_ep["replica_seconds"].mean() / 1000.0, per_ep["slo_attainment"].mean()
            xe = per_ep["replica_seconds"].std(ddof=1) / 1000.0
            ye = per_ep["slo_attainment"].std(ddof=1)
            ax.errorbar(x, y, xerr=xe, yerr=ye, color=COLORS[m], linewidth=0.8,
                        capsize=1.5, alpha=0.85, zorder=3)
            ax.plot([x], [y], marker=MARKERS[m], color=COLORS[m],
                    markerfacecolor="none" if m == "oracle" else COLORS[m],
                    markersize=5.5 if m != "oracle" else 8, zorder=4,
                    label=METHOD_LABEL[m])
        # Zoom to where the methods live. The Static-k front runs out to k=20 at
        # ~75k replica-seconds and is flat in SLO from about k=6 upward, so a
        # full-range panel puts every adaptive method inside one marker width
        # and shows nothing. The truncation is stated on the panel.
        mm = g[g["method"].isin(DEPLOYABLE + ["oracle"])]
        if len(mm):
            per = mm.groupby("method")[["replica_seconds", "slo_attainment"]].mean()
            xhi = float(per["replica_seconds"].max()) / 1000.0
            ylo = float(min(per["slo_attainment"].min(), floor if np.isfinite(floor) else 1.0))
            ax.set_xlim(0.0, xhi * 1.75)
            ax.set_ylim(max(0.0, ylo - 0.10), 1.02)
        ax.set_title(FAMILY_LABEL[fam], loc="left")
        ax.set_xlabel("cost (thousand replica-seconds)")
        if j == 0:
            ax.set_ylabel("SLO attainment")
    return fig, axes, path


# ---------------------------------------------------------------------------
# figure 3: per-family breakdown with a not-distinguishable band
# ---------------------------------------------------------------------------
def fig_per_family(df: pd.DataFrame, path: str = "out/per_family_breakdown.png"):
    fams = [f for f in FAMILY_ORDER if f in set(df["family"])]
    fig, axes = plt.subplots(1, len(fams), figsize=(2.15 * len(fams), 2.7),
                             squeeze=False)
    axes = axes[0]
    for j, (ax, fam) in enumerate(zip(axes, fams)):
        g = df[(df["family"] == fam) & (df["method"].isin(DEPLOYABLE + ["oracle"]))]
        pts: List[Tuple[str, float, float, float]] = []
        for m in DEPLOYABLE + ["oracle"]:
            gm = g[g["method"] == m]
            if not len(gm):
                continue
            v = gm.groupby("episode")["cost_total"].mean().to_numpy(dtype=float)
            lo, hi = _boot_ci_mean(v)
            pts.append((m, float(np.nanmean(v)), lo, hi))
        pts.sort(key=lambda t: t[1])
        dep = [p for p in pts if p[0] != "oracle"]
        if dep:
            b_lo, b_hi = dep[0][2], dep[0][3]
            ax.axvspan(b_lo, b_hi, color="#dfe8f0", zorder=0)
        ys = np.arange(len(pts))[::-1]
        for y, (m, mu, lo, hi) in zip(ys, pts):
            ax.plot([lo, hi], [y, y], color=COLORS[m], linewidth=1.2, zorder=2)
            ax.plot([mu], [y], marker=MARKERS[m], color=COLORS[m],
                    markerfacecolor="none" if m == "oracle" else COLORS[m],
                    markersize=5.0 if m != "oracle" else 7.5, zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels([METHOD_LABEL[m].replace(" (privileged)", "*")
                            .replace(" (tuned)", "") for m, _, _, _ in pts])
        ax.set_title(FAMILY_LABEL[fam], loc="left")
        ax.set_xlabel("mean episode cost")
        ax.margins(0.08)
    return fig, axes, path
