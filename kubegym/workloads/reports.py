"""Generate the corpus's validation results and the markdown reports.

Two artefacts come out of here:

  * `validation_results.json` -- the full battery of `validation.compare_family`
    outputs, per split and per family, plus the rate-matched variant.
  * `validation_report.md` -- the human-readable table, written from those
    numbers rather than from prose, so the report cannot drift from the data.

The rate-matched variant exists because the synthetic families are specified in
capacity fractions (0.25x to 1.5x, i.e. roughly 1.1 to 6.6 req/s at the
provisional capacity) while the Azure replay traces span from 1 arrival per hour
to 37283, a range of four and a half orders of magnitude.  Comparing raw
inter-arrival times across that gap measures the rate mismatch and nothing else.
So both are reported: the **all-traces** comparison, which shows honestly that
the generator does not cover the Azure rate range, and the **rate-matched**
comparison against Azure traces whose mean rate falls inside the synthetic
range, which isolates the shape of the arrival process.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .azure2019 import SPLIT_ORDER, AzureExtract
from .corpus import realise_entry
from .generator import PROVISIONAL_CAPACITY_RPS
from .validation import compare_family, trace_summary, worst_axes

RATE_MATCH_BAND_RPS: Tuple[float, float] = (0.25 * PROVISIONAL_CAPACITY_RPS,
                                            1.5 * PROVISIONAL_CAPACITY_RPS)


def realise_all(manifest: Dict[str, Any], extract: AzureExtract, *,
                progress_every: int = 200) -> Dict[str, Dict[str, Any]]:
    """Arrays for every trace in the manifest, keyed by trace_id."""
    out: Dict[str, Dict[str, Any]] = {}
    for i, e in enumerate(manifest["traces"]):
        r = realise_entry(e, extract)
        out[e["trace_id"]] = {"t_arrival": r["t_arrival"],
                              "service_time_s": r["service_time_s"],
                              "horizon_s": r["horizon_s"], "entry": e}
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  realised {i + 1}/{len(manifest['traces'])}", flush=True)
    return out


def build_validation(manifest: Dict[str, Any], realised: Dict[str, Dict[str, Any]],
                     ) -> Dict[str, Any]:
    """The full validation battery, per split and family, all-traces and rate-matched."""
    fams = [f for f in manifest["families"] if f != "azure_replay"]
    results: Dict[str, Any] = {
        "schema": "kubegym-workload-validation-1",
        "calibrated": False,
        "what_is_compared": "each synthetic family's pooled arrival statistics against the "
                            "RECONSTRUCTED Azure replay traces of the same split",
        "caveat": "The Azure side is itself a reconstruction. Below one minute the 'real' "
                  "inter-arrival distribution IS arrival-placement assumption A1, so agreement "
                  "there compares two models. At and above one minute the count series is the "
                  "recorded data and the comparison is against data.",
        "rate_match_band_rps": list(RATE_MATCH_BAND_RPS),
        "splits": {},
    }
    for split in SPLIT_ORDER:
        real_all = [v for v in realised.values()
                    if v["entry"]["split"] == split
                    and v["entry"]["kind"] == "azure_replay"
                    and v["t_arrival"].size >= 2]
        lo, hi = RATE_MATCH_BAND_RPS
        real_matched = [v for v in real_all
                        if lo <= v["t_arrival"].size / v["horizon_s"] <= hi]
        block: Dict[str, Any] = {
            "n_azure_traces": len(real_all),
            "n_azure_traces_rate_matched": len(real_matched),
            "azure_mean_rate_rps": {
                "min": float(min(v["t_arrival"].size / v["horizon_s"] for v in real_all)),
                "median": float(np.median([v["t_arrival"].size / v["horizon_s"]
                                           for v in real_all])),
                "max": float(max(v["t_arrival"].size / v["horizon_s"] for v in real_all)),
            },
            "families": {},
        }
        for fam in fams:
            gen = [v for v in realised.values()
                   if v["entry"]["split"] == split and v["entry"]["family"] == fam
                   and v["t_arrival"].size >= 2]
            cmp_all = compare_family(real_all, gen)
            cmp_match = compare_family(real_matched, gen) if real_matched else None
            block["families"][fam] = {
                "n_gen_traces": len(gen),
                "gen_mean_rate_rps": {
                    "min": float(min(v["t_arrival"].size / v["horizon_s"] for v in gen)),
                    "median": float(np.median([v["t_arrival"].size / v["horizon_s"]
                                               for v in gen])),
                    "max": float(max(v["t_arrival"].size / v["horizon_s"] for v in gen)),
                },
                "vs_all_azure": cmp_all,
                "vs_rate_matched_azure": cmp_match,
                "worst_axes_all": worst_axes(cmp_all),
                "worst_axes_rate_matched": worst_axes(cmp_match) if cmp_match else None,
            }
        results["splits"][split] = block
    return results


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _f(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and (abs(x) >= 1e4 or (x != 0 and abs(x) < 1e-3)):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def validation_markdown(results: Dict[str, Any], sensitivity: Dict[str, Any],
                        manifest: Dict[str, Any]) -> str:
    L: List[str] = []
    A = L.append
    A("# KubeGym workload corpus -- distributional validation report")
    A("")
    A("**Status: UNCALIBRATED.** Every number below is computed over a corpus whose Azure half "
      "is a *reconstruction* (per-minute counts and duration percentiles turned into arrival "
      "instants and service times under stated assumptions) and whose rate unit derives from "
      "placeholder constants. Nothing here is a measurement of the Azure Functions workload. "
      "See `WORKLOADS.md` and the `reconstruction_provenance` block of `corpus_manifest.json`.")
    A("")
    A("## 1. What is compared, and against what")
    A("")
    A(results["what_is_compared"] + ".")
    A("")
    A("> " + results["caveat"])
    A("")
    A("Two comparisons are reported for every family:")
    A("")
    A("1. **vs all Azure traces** -- the honest coverage answer. The synthetic families are "
      "specified in fractions of single-replica capacity (0.25x to 1.5x, about "
      f"{RATE_MATCH_BAND_RPS[0]:.2f} to {RATE_MATCH_BAND_RPS[1]:.2f} req/s at the provisional "
      "capacity), while the Azure replay traces span four and a half orders of magnitude of "
      "rate. This comparison therefore *reports the rate mismatch*, and it is large.")
    A("2. **vs rate-matched Azure traces** -- Azure traces whose mean rate falls inside the "
      "synthetic band. This isolates the *shape* of the arrival process from the rate.")
    A("")
    A("Two-sample Kolmogorov-Smirnov statistic `D` (maximum CDF gap, an effect size) and the "
      "1-Wasserstein / earth-mover distance are reported. **The KS p-values are not the point:** "
      "pooled samples run to 10^5-10^6 gaps, where every difference is significant. `D` and the "
      "Wasserstein distance are the numbers to read.")
    A("")

    for split in SPLIT_ORDER:
        b = results["splits"][split]
        A(f"## 2.{SPLIT_ORDER.index(split) + 1} Split `{split}`")
        A("")
        A(f"Azure side: {b['n_azure_traces']} replay traces, mean rate "
          f"{_f(b['azure_mean_rate_rps']['min'], 4)} to "
          f"{_f(b['azure_mean_rate_rps']['max'], 1)} req/s "
          f"(median {_f(b['azure_mean_rate_rps']['median'], 3)}). "
          f"Rate-matched subset: {b['n_azure_traces_rate_matched']} traces.")
        A("")
        for label, key in (("vs ALL Azure traces", "vs_all_azure"),
                           ("vs RATE-MATCHED Azure traces", "vs_rate_matched_azure")):
            A(f"### {label}")
            A("")
            A("| family | n gen | interarrival (raw) D | W (s) | interarrival (normalised) D | "
              "W | per-60s-bin count D | W | IoD@1s gen/real | IoD@60s gen/real | "
              "ACF60 lag1 gen-real |")
            A("|---|---|---|---|---|---|---|---|---|---|---|")
            for fam, fb in b["families"].items():
                c = fb[key]
                if c is None:
                    A(f"| `{fam}` | {fb['n_gen_traces']} | " + " | ".join(["n/a"] * 9) + " |")
                    continue
                ia, ian = c["interarrival_raw_s"], c["interarrival_normalised"]
                rb = c["rate_per_60s_bin"]
                i1, i60 = c["iod_1s"], c["iod_60s"]
                a1 = c["acf60_lag1"]
                def ratio(d):
                    if d["real_median"] in (None, 0) or d["gen_median"] is None:
                        return "n/a"
                    return f"{d['gen_median'] / d['real_median']:.2f}x"
                A(f"| `{fam}` | {fb['n_gen_traces']} | {_f(ia['ks_d'])} | {_f(ia['wasserstein'])} "
                  f"| {_f(ian['ks_d'])} | {_f(ian['wasserstein'])} | {_f(rb['ks_d'])} "
                  f"| {_f(rb['wasserstein'], 2)} | {ratio(i1)} | {ratio(i60)} "
                  f"| {_f(a1['difference_gen_minus_real'])} |")
            A("")

    A("## 3. Where the generator does NOT match the trace")
    A("")
    A("Read this section before using the synthetic families as a stand-in for the real one. "
      "The `rate_per_*_bin` and raw-inter-arrival axes are dominated by the rate-coverage gap: "
      "the generator does not span the Azure rate range, and the all-traces `D` around 0.85 on "
      "the per-60s-bin count distribution *is* that statement. The `iod_*` and `acf60_*` axes "
      "are KS statistics over per-trace scalars, so their sample size is the number of TRACES; "
      "in the rate-matched columns the Azure side is only "
      + ", ".join(f"{results['splits'][sp]['n_azure_traces_rate_matched']} ({sp})"
                  for sp in SPLIT_ORDER)
      + " traces, and a `D` near 1 there reflects that small sample as much as a real mismatch. "
      "Read the ratio columns of section 2 for those axes instead.")
    A("")
    tb = results["splits"]["train"]
    for fam, fb in tb["families"].items():
        A(f"* **`{fam}`** -- worst agreement (by KS D) against all Azure traces: "
          + ", ".join(f"`{k}` D={v:.3f}" for k, v in fb["worst_axes_all"]) + ".")
        if fb["worst_axes_rate_matched"]:
            A(f"  Rate-matched: "
              + ", ".join(f"`{k}` D={v:.3f}" for k, v in fb["worst_axes_rate_matched"]) + ".")
    A("")

    A("## 4. Sensitivity of the corpus to the arrival-placement assumption (A1)")
    A("")
    st = sensitivity["trace_statistics_all_replay_traces"]
    sm = sensitivity["simulated"]
    A(f"Trace statistics over **all {st['n_traces']}** replay traces with at least "
      f"{st['min_requests']} arrivals; downstream queueing over a stratified subsample of "
      f"**{sm['sample']['n_traces']}** traces run through the `request_service` model at a fixed "
      "static replica count (held identical across arms so the comparison is paired).")
    A("")
    A("Paired median ratio against the `uniform` reference arm, with the 10th-90th percentile "
      "band over traces:")
    A("")
    A("| measure | scope | poisson | deterministic | bursty_batch |")
    A("|---|---|---|---|---|")
    def row(summary, key, scope):
        m = summary["metrics"].get(key)
        if not m:
            return
        cells = []
        for arm in ("poisson", "deterministic", "bursty_batch"):
            r = m[arm].get("paired_ratio_vs_reference")
            cells.append(f"{r['median']:.3f}x [{r['p10_p90'][0]:.2f}, {r['p10_p90'][1]:.2f}]"
                         if r else "n/a")
        A(f"| `{key}` | {scope} | " + " | ".join(cells) + " |")
    for k in ("iod_1s", "iod_15s", "iod_60s", "peak_to_mean_15s"):
        row(st["summary"], k, f"all {st['n_traces']} traces")
    for k in ("sim_mean_latency_s", "sim_p95_latency_s", "sim_queue_violation_rate"):
        row(sm["summary"], k, f"{sm['sample']['n_traces']}-trace subsample")
    A("")
    A("**The two pools are not interchangeable.** The trace-statistic rows are pooled over every "
      f"replay trace ({st['n_traces']}); the queueing rows are the "
      f"{sm['sample']['n_traces']}-trace simulated subsample, which skews denser because it "
      f"excludes traces below {sm['sample']['min_requests']} arrivals, and therefore has different "
      "burstiness magnitudes. Where both pools report the same statistic they are shown side by "
      "side below; never carry a magnitude across the two.")
    A("")
    A("| statistic | all-traces pool | simulated subsample |")
    A("|---|---|---|")
    for k in ("iod_1s", "iod_15s"):
        rs = st["summary"]["metrics"].get(k, {}).get("bursty_batch", {}).get(
            "paired_ratio_vs_reference")
        rm = sm["summary"]["metrics"].get(k, {}).get("bursty_batch", {}).get(
            "paired_ratio_vs_reference")
        if rs and rm:
            A(f"| `{k}` `bursty_batch`/`uniform` | {rs['median']:.3f}x (n={rs['n_pairs']}) "
              f"| {rm['median']:.3f}x (n={rm['n_pairs']}) |")
    A("")
    A("**Result.** " + sensitivity["verdict"])
    A("")
    A("The `bursty_batch` arm is the intended stress case for this question: it is the only arm "
      "that changes sub-minute structure by more than Poisson noise, and it is also the arm with "
      "the least support in the data -- nothing in the Azure trace says whether serverless "
      "invocations arrive in batches within a minute.")
    A("")
    A("Three structural facts make this table readable. (i) `uniform`, `deterministic` and "
      "`bursty_batch` all preserve the recorded per-minute count exactly, so their 60 s index of "
      "dispersion is identical by construction -- the 1.000x entries are an identity, not a "
      "measurement. (ii) The `poisson` arm re-draws each minute's count from Poisson(c), which is "
      "why it is the only arm that moves the 60 s statistics; the effect is largest on the "
      "`steady` regime, where the recorded minute-to-minute variance is near zero and the "
      "injected count noise dominates. (iii) All of A1's real effect lives below one minute, "
      "and the `bursty_batch` arm is the only arm that moves it materially.")
    A("")
    A("## 5. Reading guidance")
    A("")
    A("* A `D` of 0.1 on a pooled sample of 10^5 gaps is a small effect with a p-value of zero. "
      "Compare families to each other on `D`, not to a significance threshold.")
    A("* The Wasserstein distance on raw inter-arrivals is in seconds and is dominated by the "
      "rate gap; the normalised column (each trace's gaps divided by their own mean) is the "
      "shape comparison.")
    A("* The index-of-dispersion ratios are medians over traces, so a ratio near 1 means the "
      "*typical* trace matches, not that every trace does; the per-family p10-p90 bands are in "
      "`validation_results.json`.")
    A("")
    return "\n".join(L) + "\n"
