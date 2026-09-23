"""Diagnostic figures for the workload corpus.

Four figures, each with one message:

  `corpus_families.png`     what the four generated families and the five Azure
                            regimes actually look like as rate series.
  `corpus_validation.png`   where the generated families agree with the
                            reconstructed Azure traces and where they do not.
  `corpus_sensitivity.png`  how far the arrival-placement assumption moves the
                            corpus, at the trace level and downstream.
  `corpus_population.png`   the selected 45 applications against the full Azure
                            population, and the split structure.

The functions take already-computed inputs (realised traces, validation results,
sensitivity results) so that a figure never re-runs an analysis, and each returns
the `Figure` so the caller saves it.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .azure2019 import REGIMES, SPLIT_ORDER
from .generator import FAMILIES
from .validation import ACF_LAGS, acf, bin_counts, index_of_dispersion, interarrivals

#: One colour per family / regime, threaded across every figure (rule 4.1).
FAMILY_COLOURS: Dict[str, str] = {
    "constant": "#4C72B0", "variable": "#55A868", "burst": "#C44E52", "diurnal": "#8172B2",
    "azure_replay": "#333333",
}
REGIME_COLOURS: Dict[str, str] = {
    "sparse": "#8C8C8C", "intermittent": "#937860", "steady": "#4C72B0",
    "diurnal": "#8172B2", "bursty": "#C44E52",
}
ARM_COLOURS: Dict[str, str] = {
    "uniform": "#333333", "poisson": "#4C72B0", "deterministic": "#55A868",
    "bursty_batch": "#C44E52",
}


def _pick(realised: Dict[str, Dict[str, Any]], *, family: Optional[str] = None,
          regime: Optional[str] = None, split: str = "train",
          min_requests: int = 1) -> List[Dict[str, Any]]:
    out = []
    for v in realised.values():
        e = v["entry"]
        if e["split"] != split or v["t_arrival"].size < min_requests:
            continue
        if family is not None and e["family"] != family:
            continue
        if regime is not None and e.get("regime") != regime:
            continue
        out.append(v)
    return sorted(out, key=lambda v: v["entry"]["trace_id"])


# ---------------------------------------------------------------------------
# Figure 1: the families
# ---------------------------------------------------------------------------
def figure_families(realised: Dict[str, Dict[str, Any]], *, bin_s: float = 15.0):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.4), sharex=True)
    order: List[Tuple[str, str, str]] = [
        ("family", "constant", "Constant: steady offered rate"),
        ("family", "variable", "Variable: mean-reverting log-rate"),
        ("family", "burst", "Burst: step change in rate"),
        ("family", "diurnal", "Diurnal: sinusoidal rate"),
        ("regime", "steady", "Azure, steady regime"),
        ("regime", "bursty", "Azure, bursty regime"),
        ("regime", "diurnal", "Azure, diurnal regime"),
        ("regime", "intermittent", "Azure, intermittent regime"),
        ("regime", "sparse", "Azure, sparse regime"),
    ]
    for ax, (kind, key, title) in zip(axes.ravel(), order):
        if kind == "family":
            cand = _pick(realised, family=key, min_requests=50)
            colour = FAMILY_COLOURS[key]
        else:
            cand = _pick(realised, regime=key, min_requests=1)
            cand = sorted(cand, key=lambda v: -v["t_arrival"].size)
            colour = REGIME_COLOURS[key]
        if not cand:
            ax.set_axis_off()
            continue
        v = cand[len(cand) // 3 if kind == "family" else 0]
        c = bin_counts(v["t_arrival"], v["horizon_s"], bin_s) / bin_s
        tmid = (np.arange(c.size) + 0.5) * bin_s / 60.0
        ax.plot(tmid, c, lw=0.8, color=colour)
        ax.fill_between(tmid, 0, c, color=colour, alpha=0.18, lw=0)
        ax.set_title(title, fontsize=8, loc="left")
        iod = index_of_dispersion(bin_counts(v["t_arrival"], v["horizon_s"], 60.0))
        ax.text(0.98, 0.93, f"{v['t_arrival'].size:,} req\nIoD$_{{60s}}$="
                            + (f"{iod:.1f}" if iod is not None else "n/a"),
                transform=ax.transAxes, ha="right", va="top", fontsize=6)
        ax.margins(x=0.02)
        ax.set_ylim(0, float(c.max()) * 1.42 if c.max() > 0 else 1.0)
    for ax in axes[-1]:
        ax.set_xlabel("time in episode (min)")
    for ax in axes[:, 0]:
        ax.set_ylabel("arrival rate (req/s)")
    fig.suptitle("Generated families and real Azure regimes: one representative trace each\n"
                 f"({int(bin_s)} s bins, 1 h episode). Note the differing y scales.",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ---------------------------------------------------------------------------
# Figure 2: validation
# ---------------------------------------------------------------------------
def figure_validation(realised: Dict[str, Dict[str, Any]], results: Dict[str, Any], *,
                      split: str = "train"):
    import matplotlib.pyplot as plt

    band = results["rate_match_band_rps"]
    azure = [v for v in realised.values() if v["entry"]["split"] == split
             and v["entry"]["kind"] == "azure_replay" and v["t_arrival"].size >= 2]
    azure_m = [v for v in azure if band[0] <= v["t_arrival"].size / v["horizon_s"] <= band[1]]
    fams = [f for f in FAMILIES]

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4))
    ax = axes[0, 0]
    def pooled_norm(traces):
        parts = []
        for v in traces:
            ia = interarrivals(v["t_arrival"])
            if ia.size and ia.mean() > 0:
                parts.append(ia / ia.mean())
        return np.concatenate(parts) if parts else np.zeros(0)
    for label, traces, colour, style in [
            ("Azure (rate-matched)", azure_m, FAMILY_COLOURS["azure_replay"], "-"),
            *[(f, _pick(realised, family=f, split=split, min_requests=2),
               FAMILY_COLOURS[f], "--") for f in fams]]:
        x = np.sort(pooled_norm(traces))
        if x.size < 10:
            continue
        y = 1.0 - np.arange(x.size) / x.size
        ax.plot(x, y, style, lw=1.4 if style == "-" else 1.0, color=colour, label=label)
    ax.plot(np.logspace(-3, 1.2, 200), np.exp(-np.logspace(-3, 1.2, 200)), ":",
            color="#999999", lw=0.9, label="Exponential (Poisson)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-3, 20); ax.set_ylim(1e-5, 1.2)
    ax.set_xlabel("inter-arrival gap / trace mean gap")
    ax.set_ylabel("P(gap > x)")
    ax.set_title("Inter-arrival shape: generator vs Azure", fontsize=8, loc="left")
    ax.legend(fontsize=6, frameon=False, loc="lower left")

    ax = axes[0, 1]
    xs, labels, colours = [], [], []
    for b in (1.0, 15.0, 60.0):
        for name, traces in [("Azure", azure), *[(f, _pick(realised, family=f, split=split,
                                                            min_requests=2)) for f in fams]]:
            v = [index_of_dispersion(bin_counts(t["t_arrival"], t["horizon_s"], b))
                 for t in traces]
            v = [x for x in v if x is not None and x > 0]
            xs.append(v)
            labels.append(f"{name}\n{int(b)}s")
            colours.append(FAMILY_COLOURS.get(name, FAMILY_COLOURS["azure_replay"]))
    pos = []
    p = 0.0
    for i in range(len(xs)):
        if i and i % 5 == 0:
            p += 0.8
        p += 1.0
        pos.append(p)
    bp = ax.boxplot(xs, positions=pos, widths=0.7, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_linewidth(0.6)
    for k in ("whiskers", "caps", "medians"):
        for a in bp[k]:
            a.set_linewidth(0.7); a.set_color("#333333")
    ax.set_yscale("log")
    ax.axhline(1.0, color="#999999", ls=":", lw=0.9)
    ax.text(pos[-1], 1.05, "Poisson", fontsize=6, color="#666666", ha="right", va="bottom")
    ax.set_xticks([np.mean(pos[i:i + 5]) for i in range(0, len(pos), 5)])
    ax.set_xticklabels(["1 s bins", "15 s bins", "60 s bins"])
    ax.set_ylabel("index of dispersion")
    ax.set_title("Burstiness by timescale (box = traces)", fontsize=8, loc="left")
    handles = [plt.Line2D([], [], color=FAMILY_COLOURS.get(n, "#333333"), lw=4, alpha=0.55,
                          label=("Azure replay" if n == "Azure" else n))
               for n in ["Azure", *fams]]
    ax.legend(handles=handles, fontsize=6, frameon=False, ncol=2, loc="upper left")

    ax = axes[1, 0]
    for name, traces, colour in [("Azure replay", azure, FAMILY_COLOURS["azure_replay"]),
                                 *[(f, _pick(realised, family=f, split=split, min_requests=2),
                                    FAMILY_COLOURS[f]) for f in fams]]:
        acfs = []
        for v in traces:
            a = acf(bin_counts(v["t_arrival"], v["horizon_s"], 60.0))
            row = [a[f"lag{k}"] for k in ACF_LAGS]
            if all(x is not None for x in row):
                acfs.append(row)
        if not acfs:
            continue
        M = np.asarray(acfs)
        ax.plot(ACF_LAGS, np.median(M, axis=0), "-o", ms=2.5, lw=1.0, color=colour, label=name)
    ax.axhline(0.0, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("lag (minutes)")
    ax.set_ylabel("autocorrelation, per-minute count")
    ax.set_title("Rate memory: median ACF over traces", fontsize=8, loc="left")
    ax.legend(fontsize=6, frameon=False)

    ax = axes[1, 1]
    ks_labels = ["interarrival\n(normalised)", "per-15s\ncount", "per-60s\ncount"]
    keys = ["interarrival_normalised", "rate_per_15s_bin", "rate_per_60s_bin"]
    w = 0.2
    for i, f in enumerate(fams):
        fb = results["splits"][split]["families"][f]
        allv = [fb["vs_all_azure"][k]["ks_d"] for k in keys]
        matv = [(fb["vs_rate_matched_azure"] or {}).get(k, {}).get("ks_d") for k in keys]
        x = np.arange(len(keys)) + (i - 1.5) * w
        ax.bar(x, allv, width=w * 0.92, color=FAMILY_COLOURS[f], alpha=0.45, lw=0)
        ax.plot(x, [m if m is not None else np.nan for m in matv], "D", ms=4,
                color=FAMILY_COLOURS[f], label=f)
    ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels(ks_labels)
    ax.set_ylabel("KS statistic $D$ (lower = closer)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Bars: vs all Azure.  Diamonds: rate-matched", fontsize=8, loc="left")
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper left")

    fig.suptitle(f"Generated families vs reconstructed Azure replay, split `{split}`",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


# ---------------------------------------------------------------------------
# Figure 3: sensitivity
# ---------------------------------------------------------------------------
def figure_sensitivity(sensitivity: Dict[str, Any]):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    st = sensitivity["trace_statistics_all_replay_traces"]
    sm = sensitivity["simulated"]
    arms = ["poisson", "deterministic", "bursty_batch"]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.1))

    ax = axes[0]
    keys = ["iod_1s", "iod_15s", "iod_60s"]
    for i, arm in enumerate(arms):
        med, lo, hi = [], [], []
        for k in keys:
            r = st["summary"]["metrics"][k][arm]["paired_ratio_vs_reference"]
            med.append(r["median"]); lo.append(r["p10_p90"][0]); hi.append(r["p10_p90"][1])
        x = np.arange(len(keys)) + (i - 1) * 0.22
        ax.errorbar(x, med, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
                    fmt="o", ms=4, lw=1.0, capsize=2, color=ARM_COLOURS[arm], label=arm)
    ax.axhline(1.0, color=ARM_COLOURS["uniform"], lw=1.0)
    ax.text(2.42, 1.0, "uniform", fontsize=6, va="bottom", ha="right", color="#333333")
    ax.set_yscale("log"); ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(["1 s", "15 s", "60 s"])
    ax.set_xlabel("count bin width")
    ax.set_ylabel("IoD ratio to uniform")
    ax.set_title(f"Trace statistics ({st['n_traces']} traces)", fontsize=8, loc="left")
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.set_xlim(-0.45, 2.9)

    ax = axes[1]
    keys = ["sim_mean_latency_s", "sim_p95_latency_s", "sim_queue_violation_rate"]
    names = ["mean lat.", "p95 lat.", "queue viol."]
    for i, arm in enumerate(arms):
        med, lo, hi = [], [], []
        for k in keys:
            r = sm["summary"]["metrics"][k][arm]["paired_ratio_vs_reference"]
            med.append(r["median"]); lo.append(r["p10_p90"][0]); hi.append(r["p10_p90"][1])
        x = np.arange(len(keys)) + (i - 1) * 0.22
        ax.errorbar(x, med, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
                    fmt="o", ms=4, lw=1.0, capsize=2, color=ARM_COLOURS[arm])
    ax.axhline(1.0, color=ARM_COLOURS["uniform"], lw=1.0)
    ax.set_yscale("log"); ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels(names)
    ax.set_ylabel("ratio to uniform")
    ax.set_title(f"Downstream queueing ({sm['sample']['n_traces']} traces)",
                 fontsize=8, loc="left")

    ax = axes[2]
    regimes = [r for r in REGIMES if r in st["paired_ratio_by_regime"]]
    for i, arm in enumerate(arms):
        v = [st["paired_ratio_by_regime"][r]["iod_1s"][arm] for r in regimes]
        ax.plot(np.arange(len(regimes)) + (i - 1) * 0.12, v, "o", ms=4,
                color=ARM_COLOURS[arm])
    ax.axhline(1.0, color=ARM_COLOURS["uniform"], lw=1.0)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(regimes)))
    ax.set_xticklabels(regimes, rotation=30, ha="right")
    ax.set_ylabel("IoD$_{1s}$ ratio to uniform")
    ax.set_title("By Azure regime", fontsize=8, loc="left")

    fig.suptitle("Sensitivity to arrival-placement assumption A1\n"
                 "paired medians; bars span the 10th-90th percentile over traces",
                 fontsize=9, x=0.01, ha="left")
    for ax in axes:
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _p: f"{v:g}" if v >= 0.1 else f"{v:.2g}"))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


# ---------------------------------------------------------------------------
# Figure 4: population and splits
# ---------------------------------------------------------------------------
def figure_population(extract, manifest: Dict[str, Any], population_frame=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.1))
    pop = extract.meta["population_comparison"]

    ax = axes[0]
    if population_frame is not None:
        pf = population_frame
        ax.scatter(pf["mean_rate_per_minute"].clip(lower=1e-4), pf["zero_minute_fraction"],
                   s=1.5, c="#CCCCCC", lw=0, rasterized=True, label="all 14-day apps "
                   f"(n={len(pf):,})")
    xs = [a.stats["mean_rate_per_minute"] for a in extract.apps]
    ys = [a.stats["zero_minute_fraction"] for a in extract.apps]
    cs = [REGIME_COLOURS[a.regime] for a in extract.apps]
    ax.scatter(xs, ys, s=18, c=cs, edgecolor="white", lw=0.4, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("mean rate (invocations/min, 14 d)")
    ax.set_ylabel("zero-invocation minutes")
    ax.set_title("45 selected applications", fontsize=8, loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=REGIME_COLOURS[r], label=r)
               for r in REGIMES]
    if population_frame is not None:
        handles.append(plt.Line2D([], [], marker="o", ls="", ms=3, color="#CCCCCC",
                                  label="population"))
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="upper right")

    ax = axes[1]
    keys = ["sparse", "intermittent", "steady", "diurnal", "bursty"]
    p = [pop["regime_share"]["population"].get(k, 0.0) for k in keys]
    s = [pop["regime_share"]["selected"].get(k, 0.0) for k in keys]
    y = np.arange(len(keys))
    ax.barh(y + 0.19, p, height=0.36, color="#CCCCCC", lw=0, label="Azure population")
    ax.barh(y - 0.19, s, height=0.36, color=[REGIME_COLOURS[k] for k in keys], lw=0,
            label="corpus (stratified)")
    ax.set_yticks(y); ax.set_yticklabels(keys)
    ax.set_xlabel("share of applications")
    ax.set_title("Corpus oversamples the rare regimes", fontsize=8, loc="left")
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.margins(x=0.10)

    ax = axes[2]
    fams = sorted(manifest["n_traces_by_family"])
    counts = {s: [sum(1 for e in manifest["traces"] if e["split"] == s and e["family"] == f)
                  for f in fams] for s in SPLIT_ORDER}
    x = np.arange(len(fams))
    bottom = np.zeros(len(fams))
    shades = {"train": "#4C72B0", "dev": "#55A868", "test": "#C44E52"}
    for s in SPLIT_ORDER:
        ax.bar(x, counts[s], bottom=bottom, width=0.62, color=shades[s], alpha=0.85, lw=0,
               label=s)
        bottom += np.asarray(counts[s], dtype=float)
    for xi, tot in zip(x, bottom):
        ax.text(xi, tot + 6, f"{int(tot)}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("azure_replay", "azure\nreplay") for f in fams],
                       rotation=0, fontsize=6)
    ax.set_ylabel("traces")
    ax.set_ylim(0, bottom.max() * 1.18)
    ax.set_title(f"{manifest['n_traces']} traces, disjoint splits", fontsize=8, loc="left")
    ax.legend(fontsize=6, frameon=False, loc="upper right")

    fig.suptitle("Application selection, population coverage, and corpus composition",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig
