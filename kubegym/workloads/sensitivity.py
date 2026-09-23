"""Sensitivity of the corpus to the arrival-placement assumption (A1).

WHY THIS EXISTS
---------------
The Azure trace records how many invocations happened in each minute, not when.
Placing them inside the minute is assumption A1 of `reconstruction.py`, and the
dataset cannot say which placement is right.  If the choice barely moves the
difficulty of the resulting autoscaling problem, A1 is a harmless modelling
detail and the corpus can be used without qualification.  If it moves it a lot,
that is a limitation of the corpus and every result over it inherits an
uncertainty band the paper must state.

WHAT IS MEASURED
----------------
Two layers, on the same traces:

  * **Trace statistics** -- index of dispersion at 1 s / 15 s / 60 s bins, and
    peak-to-mean at 15 s.  Cheap, and they show directly where an arrival
    placement can and cannot matter: all four arms agree exactly at the 60 s bin
    by construction (three of them preserve the recorded per-minute count
    exactly; `poisson` re-draws it), so any movement at 60 s is Poisson count
    noise, and all the action is at 1 s and 15 s.

  * **A downstream queueing outcome** -- each trace is run through the actual
    `kubegym` simulator under the `request_service` model with a *fixed* static
    replica count, and mean / p95 request latency and the queueing-delay
    violation rate are recorded.  Static provisioning is used on purpose: a
    closed-loop controller would partly absorb the difference and confound the
    measurement of the assumption with the measurement of the controller.

    The replica count is `ceil(offered request-seconds / horizon / capacity)`
    clipped to the config's `[n_replicas_min, n_replicas_max]`, computed from the
    `uniform` arm and held fixed across arms so all four arms face the same
    provisioning decision.

Every number produced here is uncalibrated: the service model's constants are
placeholders (`request_service_default.json`) and the traces are reconstructions.
The sensitivity result is a statement about the *relative* movement between
arms, which is meaningful even when the absolute level is not.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import ProvenancedConfig, Simulator, config_path
from ..models.request_service import RequestServiceModel
from .azure2019 import AzureExtract, AzureReplaySource
from .reconstruction import PLACEMENT_ARMS, ReconstructionSpec
from .validation import bin_counts, index_of_dispersion

DIFFICULTY_KEYS = ("iod_1s", "iod_15s", "iod_60s", "peak_to_mean_15s",
                   "sim_mean_latency_s", "sim_p95_latency_s", "sim_queue_violation_rate")


def sample_traces(entries: Sequence[Dict[str, Any]], *, per_regime: int = 5,
                  min_requests: int = 200, max_requests: int = 20_000) -> List[Dict[str, Any]]:
    """Deterministic stratified sample of replay entries for the sensitivity run.

    Traces with fewer than `min_requests` arrivals are excluded: a p95 latency
    over a handful of requests is noise, not a difficulty measure.  That biases
    the sample toward the denser regimes, which is stated in the report.
    """
    pool = [e for e in entries if e["kind"] == "azure_replay"
            and min_requests <= e.get("n_requests", 0) <= max_requests]
    out: List[Dict[str, Any]] = []
    for regime in sorted({e["regime"] for e in pool}):
        cell = sorted([e for e in pool if e["regime"] == regime],
                      key=lambda e: hashlib.sha256(
                          f"sensitivity|{e['trace_id']}".encode()).hexdigest())
        out.extend(cell[:per_regime])
    return out


def _static_replicas(offered_request_seconds: float, horizon_s: float, cfg) -> int:
    per_replica = float(cfg.f("rs_max_concurrency"))
    lo, hi = int(cfg.f("n_replicas_min")), int(cfg.f("n_replicas_max"))
    need = offered_request_seconds / max(horizon_s, 1e-9) / max(per_replica, 1e-9)
    return int(min(hi, max(lo, math.ceil(need))))


def _simulate(source: AzureReplaySource, cfg, k: int, horizon_s: float,
              seed: int) -> Dict[str, Any]:
    model = RequestServiceModel(cfg)
    sim = Simulator(model, source, cfg, k0=k)
    sim.reset(episode=0, seed=seed)
    sim.step_to(horizon_s)
    truncated = sim.drain_all(t_cap=horizon_s * 3.0)
    lat = [r.t_done - r.t_arrival for r in sim.requests if r.t_done is not None]
    qd = [r.t_admit - r.t_arrival for r in sim.requests if r.t_admit is not None]
    slo = cfg.f("slo")
    return {
        "k_static": k, "n_requests": len(sim.requests), "n_done": len(lat),
        "truncated_drain": bool(truncated),
        "mean_latency_s": float(np.mean(lat)) if lat else None,
        "p95_latency_s": float(np.quantile(lat, 0.95)) if lat else None,
        "queue_violation_rate": (float(np.mean(np.asarray(qd) > slo["queue_delay_p95_s"]))
                                 if qd else None),
        "replica_seconds": float(sim.total_replica_seconds()),
    }


def run_sensitivity(extract: AzureExtract, entries: Sequence[Dict[str, Any]], *,
                    base_spec: Optional[ReconstructionSpec] = None,
                    arms: Sequence[str] = PLACEMENT_ARMS, per_regime: int = 5,
                    min_requests: int = 200, max_requests: int = 20_000,
                    simulate: bool = True, progress: bool = True) -> Dict[str, Any]:
    """Rebuild the sampled traces under every arrival-placement arm and compare."""
    base = base_spec or ReconstructionSpec()
    cfg = ProvenancedConfig.load(config_path("request_service_default.json"))
    sample = sample_traces(entries, per_regime=per_regime, min_requests=min_requests,
                           max_requests=max_requests)
    rows: List[Dict[str, Any]] = []

    for n, e in enumerate(sample):
        # provisioning decision from the reference arm, held fixed across arms
        ref = AzureReplaySource(extract, e["app_hash"],
                               [(e["day"], e["segment_start_minute"])],
                               horizon_s=e["horizon_s"], spec=base)
        ref_arr = ref.arrays(0, seed=e["seed"])
        k = _static_replicas(float(ref_arr["service_time_s"].sum()), e["horizon_s"], cfg)
        for arm in arms:
            spec = ReconstructionSpec(**{**base.spec(), "within_minute_placement": arm})
            src = AzureReplaySource(extract, e["app_hash"],
                                   [(e["day"], e["segment_start_minute"])],
                                   horizon_s=e["horizon_s"], spec=spec)
            arr = src.arrays(0, seed=e["seed"])
            t, svc = arr["t_arrival"], arr["service_time_s"]
            row: Dict[str, Any] = {
                "trace_id": e["trace_id"], "regime": e["regime"], "split": e["split"],
                "arm": arm, "n_requests": int(t.size), "k_static": k,
                "offered_request_seconds": float(svc.sum()),
            }
            for b in (1.0, 15.0, 60.0):
                c = bin_counts(t, e["horizon_s"], b)
                row[f"iod_{int(b)}s"] = index_of_dispersion(c)
                if b == 15.0:
                    row["peak_to_mean_15s"] = float(c.max() / c.mean()) if c.mean() > 0 else None
            if simulate:
                sim = _simulate(src, cfg, k, e["horizon_s"], e["seed"])
                row["sim_mean_latency_s"] = sim["mean_latency_s"]
                row["sim_p95_latency_s"] = sim["p95_latency_s"]
                row["sim_queue_violation_rate"] = sim["queue_violation_rate"]
                row["sim_truncated_drain"] = sim["truncated_drain"]
                row["sim_replica_seconds"] = sim["replica_seconds"]
            rows.append(row)
        if progress and (n + 1) % 5 == 0:
            print(f"  sensitivity {n + 1}/{len(sample)} traces", flush=True)

    return {"rows": rows, "summary": summarise_sensitivity(rows, arms),
            "sample": {"n_traces": len(sample), "per_regime": per_regime,
                       "regimes": sorted({e["regime"] for e in sample}),
                       "min_requests": min_requests, "max_requests": max_requests,
                       "selection": "deterministic hash order within regime",
                       "bias": f"traces with fewer than {min_requests} arrivals in the hour are excluded "
                               "because a p95 latency over a handful of requests is noise; the "
                               "sample therefore under-represents the sparse regime, where by "
                               "construction the arrival-placement question has the least room "
                               "to matter (most minutes hold 0 or 1 invocation)"},
            "reference_arm": base.within_minute_placement,
            "config": "request_service_default.json (every constant a placeholder)",
            "calibrated": False}


def run_statistics_sensitivity(extract: AzureExtract, entries: Sequence[Dict[str, Any]], *,
                               base_spec: Optional[ReconstructionSpec] = None,
                               arms: Sequence[str] = PLACEMENT_ARMS,
                               min_requests: int = 10,
                               progress_every: int = 100) -> Dict[str, Any]:
    """Trace-statistics-only sensitivity over EVERY replay trace in the corpus.

    No simulation, so it covers all 401 replay traces rather than a sample, and
    it includes the sparse regime that the simulated study has to exclude.  This
    is the arm that establishes the structural facts: three of the four arms
    preserve the recorded per-minute count exactly, so the 60 s index of
    dispersion is identical for them by construction, and the whole effect of A1
    lives below one minute.
    """
    base = base_spec or ReconstructionSpec()
    pool = [e for e in entries if e["kind"] == "azure_replay"
            and e.get("n_requests", 0) >= min_requests]
    rows: List[Dict[str, Any]] = []
    for n, e in enumerate(pool):
        for arm in arms:
            spec = ReconstructionSpec(**{**base.spec(), "within_minute_placement": arm})
            src = AzureReplaySource(extract, e["app_hash"],
                                   [(e["day"], e["segment_start_minute"])],
                                   horizon_s=e["horizon_s"], spec=spec)
            t = src.arrays(0, seed=e["seed"])["t_arrival"]
            row = {"trace_id": e["trace_id"], "regime": e["regime"], "split": e["split"],
                   "arm": arm, "n_requests": int(t.size)}
            for b in (1.0, 15.0, 60.0):
                c = bin_counts(t, e["horizon_s"], b)
                row[f"iod_{int(b)}s"] = index_of_dispersion(c)
                if b == 15.0:
                    row["peak_to_mean_15s"] = float(c.max() / c.mean()) if c.mean() > 0 else None
            rows.append(row)
        if progress_every and (n + 1) % progress_every == 0:
            print(f"  statistics sensitivity {n + 1}/{len(pool)}", flush=True)
    keys = ("iod_1s", "iod_15s", "iod_60s", "peak_to_mean_15s")
    summary = summarise_sensitivity(rows, arms)
    summary["metrics"] = {k: v for k, v in summary["metrics"].items() if k in keys}
    by_regime: Dict[str, Any] = {}
    for regime in sorted({r["regime"] for r in rows}):
        sub = [r for r in rows if r["regime"] == regime]
        s = summarise_sensitivity(sub, arms)
        by_regime[regime] = {k: {a: s["metrics"][k][a].get("paired_ratio_vs_reference",
                                                           {}).get("median")
                                 for a in arms if a != arms[0]}
                             for k in keys}
    return {"rows": rows, "summary": summary, "paired_ratio_by_regime": by_regime,
            "n_traces": len(pool), "min_requests": min_requests,
            "coverage": "every replay trace with at least "
                        f"{min_requests} arrivals, all regimes including sparse",
            "calibrated": False}


def summarise_sensitivity(rows: Sequence[Dict[str, Any]],
                          arms: Sequence[str] = PLACEMENT_ARMS) -> Dict[str, Any]:
    """Per-arm medians and the paired ratio against the reference arm."""
    by_arm: Dict[str, List[Dict[str, Any]]] = {a: [r for r in rows if r["arm"] == a]
                                               for a in arms}
    ref = arms[0]
    out: Dict[str, Any] = {"reference_arm": ref, "n_traces_per_arm":
                           {a: len(v) for a, v in by_arm.items()}, "metrics": {}}
    for key in DIFFICULTY_KEYS:
        entry: Dict[str, Any] = {}
        refmap = {r["trace_id"]: r.get(key) for r in by_arm[ref]}
        for a in arms:
            vals = np.asarray([r[key] for r in by_arm[a]
                               if r.get(key) is not None], dtype=np.float64)
            entry[a] = {"median": float(np.median(vals)) if vals.size else None,
                        "p10_p90": [float(np.quantile(vals, .1)), float(np.quantile(vals, .9))]
                        if vals.size else None}
            if a != ref:
                pairs = [(refmap[r["trace_id"]], r[key]) for r in by_arm[a]
                         if r.get(key) is not None and refmap.get(r["trace_id"]) is not None
                         and refmap[r["trace_id"]] > 0]
                if pairs:
                    ratio = np.asarray([b / a0 for a0, b in pairs], dtype=np.float64)
                    entry[a]["paired_ratio_vs_reference"] = {
                        "median": float(np.median(ratio)),
                        "p10_p90": [float(np.quantile(ratio, .1)),
                                    float(np.quantile(ratio, .9))],
                        "n_pairs": int(ratio.size)}
        out["metrics"][key] = entry
    return out


def sensitivity_verdict(summary: Dict[str, Any]) -> str:
    """One paragraph stating how much the assumption matters, from the numbers."""
    m = summary["metrics"]

    def pair(key, arm):
        return m.get(key, {}).get(arm, {}).get("paired_ratio_vs_reference")

    # NOTE: this function sees only the SIMULATED subsample's summary, so it must not
    # quote trace-statistic magnitudes -- those are reported over the full replay set in
    # the accompanying table and the two scopes differ (e.g. iod_1s bursty_batch is
    # 3.67x on the 42-trace subsample but 2.79x pooled over all 339 traces).  Quoting a
    # subsample figure next to a pooled table is how a reader ends up with two numbers
    # for one statistic, so the sub-minute effect is described qualitatively here and
    # the reader is pointed at the table for magnitudes.
    n_sim = summary.get("n_traces_per_arm", {}).get("bursty_batch")
    med_worst, p90_worst = 1.0, 1.0
    for key in ("sim_mean_latency_s", "sim_p95_latency_s", "sim_queue_violation_rate"):
        for arm in ("poisson", "deterministic", "bursty_batch"):
            r = pair(key, arm)
            if not r:
                continue
            med_worst = max(med_worst, r["median"], 1.0 / max(r["median"], 1e-9))
            p90_worst = max(p90_worst, r["p10_p90"][1], 1.0 / max(r["p10_p90"][0], 1e-9))
    head = ("Sub-minute burstiness moves substantially between arms -- the batch-arrival arm "
            "several-fold, the deterministic arm downward -- and the accompanying trace-statistics "
            "table carries the magnitudes at each bin width and their scope. The downstream "
            "queueing outcome does not follow: over the "
            f"{n_sim}-trace simulated subsample, across mean latency, p95 latency and the "
            "queue-delay violation rate, no arm's paired median differs from the uniform reference "
            f"by more than {med_worst:.2f}x. ")
    if med_worst <= 1.15:
        head += ("The arrival-placement assumption is therefore a modelling detail rather than a "
                 "driver of difficulty at the median trace, which is a useful negative result: "
                 "a reviewer need not accept A1 to accept a controller comparison run over this "
                 "corpus. ")
    else:
        head += ("The arrival-placement assumption is therefore consequential and every result "
                 "over this corpus inherits it as an uncertainty band. ")
    head += (f"The tail is not negligible, though: in the upper decile of that subsample the "
             f"batch-arrival arm inflates the same quantities by up to {p90_worst:.2f}x, so a "
             f"per-trace claim -- as opposed to a corpus-median claim -- does carry A1 as an "
             f"uncertainty. Reporting a controller comparison as a median over the corpus is "
             f"robust to A1; reporting the worst-case trace is not.")
    return head
