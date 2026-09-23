"""Corpus assembly: parameter grid, splits, checksummed manifest, regeneration.

THE SPLIT RULE, IN FULL
----------------------
A trace belongs to exactly one of `train` / `dev` / `test`, and the three splits
share **nothing**:

  * **Applications** (replay half).  Each of the 45 selected Azure applications
    is assigned to one split by the selection rule (`azure2019.py`): within each
    of the 15 (regime x volume-tercile) cells the three picks go to train, dev
    and test respectively.  The three application sets are therefore disjoint by
    construction, and every split contains all 5 regimes x 3 terciles.

  * **Time segments** (replay half).  The 14 source days are partitioned:
    train {1,2,3,6,8,9,13}, dev {4,10,14}, test {5,7,11,12}.  A replay trace uses
    only days in its own split, so no minute of source time appears in two
    splits.  Within a day, segments are non-overlapping whole-hour windows.
    The partition is deliberately non-contiguous so that each split contains at
    least one of days 6, 7, 13, 14, which carry markedly fewer distinct
    functions than the rest (see `azure2019.DAY_SPLITS`).

  * **Seeds** (both halves).  Seed bands are disjoint: train [0, 100000), dev
    [100000, 200000), test [200000, 300000).  No realisation is shared even
    where parameters are.

  * **Parameters** (synthetic half) are shared across splits on purpose.  The
    synthetic families exist to measure generalisation over *realisations* of a
    stated rate process, not over hyperparameters; hiding grid points from train
    would make the dev/test numbers measure extrapolation to unseen parameters, a
    different question.  The disjointness that matters for the synthetic half is
    the seed band, and it is enforced.

  * **The synthetic service-time model** is fitted on **train applications and
    train days only**, so no dev/test duration information reaches any trace.

`verify_splits()` checks all of the above programmatically and is exercised by
`kubegym/tests/test_workload_corpus.py`.

REGENERABILITY
--------------
Every trace is a pure function of its manifest entry.  `regenerate(entry)`
rebuilds the source from the serialised spec alone (never from a live object) and
`verify_manifest()` re-derives every digest and compares.  The digest is the
SHA256 of the trace's canonical JSONL bytes, which is also the SHA256 of the
materialised file when a trace is written out, so a shipped file and its manifest
entry are checkable against each other with `sha256sum`.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .azure2019 import (DAY_SPLITS, SEED_BANDS, SPLIT_ORDER, AzureExtract, AzureReplaySource,
                        segment_starts, sha256_file)
from .generator import (BurstRate, Capacity, ConstantRate, DiurnalRate, ServiceTimeModel,
                        SyntheticSource, SyntheticSpec, VariableRate, default_capacity,
                        synthetic_from_spec)
from .reconstruction import PERCENTILE_KNOTS, ReconstructionSpec
from .validation import trace_summary

CORPUS_SCHEMA = "kubegym-workload-corpus-1"
DEFAULT_HORIZON_S = 3600.0
DEFAULT_SEGMENTS_PER_APP_DAY = 2


# ---------------------------------------------------------------------------
# Canonical serialisation and digest
# ---------------------------------------------------------------------------
def canonical_lines(t_arrival: np.ndarray, service_time_s: np.ndarray, *,
                    extra: Optional[Dict[str, Sequence[Any]]] = None) -> bytes:
    """The canonical JSONL bytes of a trace.

    Fixed key order, fixed 6-decimal float formatting, `\\n` line endings, so the
    bytes -- and hence the digest -- do not depend on platform, locale, or json
    library version.  6 decimals is 1 microsecond of simulated time, far below
    any timescale the model expresses.
    """
    t = np.asarray(t_arrival, dtype=np.float64)
    s = np.asarray(service_time_s, dtype=np.float64)
    if t.shape != s.shape:
        raise ValueError("t_arrival and service_time_s must have the same shape")
    cols = {k: list(v) for k, v in (extra or {}).items()}
    parts: List[str] = []
    for i in range(t.size):
        row = f'{{"seq":{i},"t_arrival":{t[i]:.6f},"service_time_s":{s[i]:.6f}'
        for k in sorted(cols):
            v = cols[k][i]
            row += f',"{k}":' + (json.dumps(v) if not isinstance(v, (int, np.integer))
                                 else str(int(v)))
        parts.append(row + "}\n")
    return "".join(parts).encode("utf-8")


def trace_digest(t_arrival: np.ndarray, service_time_s: np.ndarray, **kw) -> str:
    return hashlib.sha256(canonical_lines(t_arrival, service_time_s, **kw)).hexdigest()


# ---------------------------------------------------------------------------
# The synthetic service-time model, fitted on train applications only
# ---------------------------------------------------------------------------
def pooled_service_icdf(extract: AzureExtract, spec: ReconstructionSpec, *,
                        splits: Sequence[str] = ("train",), n_knots: int = 201,
                        n_eval: int = 201) -> Tuple[ServiceTimeModel, Dict[str, Any]]:
    """Count-weighted pooled inverse CDF of reconstructed service times.

    The pooled marginal of the reconstruction, computed analytically from the
    duration percentile knots rather than by sampling, so it carries no seed.
    Restricted to applications in `splits` (default: train only) and to those
    applications' own split days, so the synthetic families' service-time
    marginal contains no dev/test information.
    """
    from .reconstruction import build_icdf_knots

    apps = {a.app_hash for a in extract.apps if a.split in splits}
    days = set()
    for s in splits:
        days.update(DAY_SPLITS[s])
    grid = np.linspace(0.0, 1.0, n_eval)
    vals: List[np.ndarray] = []
    wts: List[np.ndarray] = []
    n_rows = 0
    for a in apps:
        for d in sorted(days):
            rows = extract.function_days(a, d)
            for r in rows:
                r = int(r)
                cnt = float(extract.dur["count"][r])
                if cnt <= 0:
                    continue
                mn = extract.dur["minimum_ms"][r]
                mx = extract.dur["maximum_ms"][r]
                q, v, _ = build_icdf_knots(extract.dur["pct_ms"][r],
                                           None if mn < 0 else float(mn),
                                           None if mx < 0 else float(mx), spec)
                vals.append(np.interp(grid, q, v))
                wts.append(np.full(n_eval, cnt / n_eval))
                n_rows += 1
    if not vals:
        raise ValueError("no duration rows to pool")
    x = np.concatenate(vals)
    w = np.concatenate(wts)
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    qk = np.linspace(0.0, 1.0, n_knots)
    vk = np.interp(qk, cw, x)
    vk = np.maximum.accumulate(np.clip(vk, spec.min_service_time_s, spec.max_service_time_s))
    model = ServiceTimeModel(kind="azure_pooled", q=tuple(float(z) for z in qk),
                             v_s=tuple(float(z) for z in vk),
                             min_s=spec.min_service_time_s, max_s=spec.max_service_time_s)
    report = {
        "fitted_on_splits": list(splits),
        "fitted_on_days": sorted(days),
        "n_applications": len(apps),
        "n_function_day_rows_pooled": n_rows,
        "weighting": "each function-day contributes its durations-table Count",
        "mean_s": float(np.trapezoid(vk, qk)) if hasattr(np, "trapezoid")
                  else float(np.trapz(vk, qk)),
        "median_s": float(np.interp(0.5, qk, vk)),
        "p99_s": float(np.interp(0.99, qk, vk)),
        "leakage_note": "fitted on train applications and train days only, so no dev/test "
                        "duration information reaches any synthetic trace",
        "calibrated": False,
    }
    return model, report


# ---------------------------------------------------------------------------
# The parameter grid
# ---------------------------------------------------------------------------
#: What each family is in the corpus to test.  Copied verbatim into every
#: manifest entry so a downstream table can group by intent.
FAMILY_INTENT: Dict[str, str] = {
    "constant": "steady-state provisioning and the cost of over-provisioning; the four load "
                "levels straddle single-replica capacity so both under- and over-provisioned "
                "regimes appear",
    "variable": "tracking a stochastic rate process whose correlation time tau is above, near "
                "and below the replica cold-start time: the regime where a reactive controller "
                "provably cannot keep up is included on purpose",
    "burst": "scale-up latency against a step change; magnitude and duration are crossed so "
             "that bursts shorter than a cold start and bursts longer than one both appear, "
             "with and without repetition",
    "diurnal": "predictable seasonality at periods shorter than, comparable to, and longer than "
               "the one-hour episode, i.e. multi-cycle, single-cycle and slow-trend cases",
    "azure_replay": "the recorded per-minute arrival-count series of a real serverless "
                    "application, stratified over five load regimes and three volume terciles",
}


def parameter_grid(horizon_s: float = DEFAULT_HORIZON_S) -> List[Dict[str, Any]]:
    """The documented synthetic parameter grid: 61 points over four families.

    Rates are fractions of single-replica capacity.  `grid_dt_s` for the
    deterministic schedules is the control interval (15 s); for `variable` it is
    60 s, matching the granularity at which the real trace is recorded.
    """
    pts: List[Dict[str, Any]] = []

    for load in (0.25, 0.5, 0.75, 1.0, 1.5):
        pts.append({"family": "constant", "schedule": ConstantRate(load=load),
                    "tests": f"steady load at {load:g}x single-replica capacity"})

    for mean_load in (0.5, 1.0):
        for sigma in (0.3, 0.6, 1.0):
            for tau in (60.0, 300.0, 900.0):
                pts.append({"family": "variable",
                            "schedule": VariableRate(mean_load=mean_load, sigma=sigma,
                                                     tau_s=tau, grid_dt_s=60.0),
                            "tests": f"mean-reverting log-rate, sigma={sigma:g}, tau={tau:g}s "
                                     f"({'below' if tau < 120 else 'above'} the placeholder "
                                     f"2 s cold start by a wide margin; tau vs the 15 s control "
                                     f"interval is the binding comparison)"})
    for sigma in (0.3, 0.6):
        pts.append({"family": "variable",
                    "schedule": VariableRate(mean_load=1.0, sigma=sigma,
                                             tau_s=float("inf"), grid_dt_s=60.0),
                    "tests": f"NON-STATIONARY arm: random walk in log-rate, sigma={sigma:g}/h"})

    for base in (0.3, 0.6):
        for mag in (3.0, 6.0, 12.0):
            for dur in (60.0, 300.0):
                for rep in (None, 900.0):
                    pts.append({"family": "burst",
                                "schedule": BurstRate(base_load=base, magnitude=mag,
                                                      start_s=600.0, duration_s=dur,
                                                      repeat_every_s=rep, grid_dt_s=15.0),
                                "tests": f"{mag:g}x step for {dur:g}s"
                                         + (f", repeating every {rep:g}s" if rep else
                                            ", single occurrence")})

    for mean_load in (0.5, 1.0):
        for amp in (0.4, 0.8):
            for period in (900.0, 3600.0, 14400.0):
                pts.append({"family": "diurnal",
                            "schedule": DiurnalRate(mean_load=mean_load, amplitude=amp,
                                                    period_s=period, grid_dt_s=15.0),
                            "tests": f"sinusoid amplitude {amp:g}, period {period:g}s "
                                     f"({'multi-cycle' if period < horizon_s else 'single cycle' if period == horizon_s else 'slow trend'} "
                                     f"within a {horizon_s:g}s episode)"})
    return pts


# ---------------------------------------------------------------------------
# Corpus build
# ---------------------------------------------------------------------------
@dataclass
class CorpusConfig:
    horizon_s: float = DEFAULT_HORIZON_S
    segments_per_app_day: int = DEFAULT_SEGMENTS_PER_APP_DAY
    seeds_per_grid_point: int = 2
    min_invocations_per_segment: int = 5
    reconstruction: ReconstructionSpec = None            # type: ignore[assignment]
    materialise_per_family_split: int = 2

    def __post_init__(self) -> None:
        if self.reconstruction is None:
            self.reconstruction = ReconstructionSpec()

    def spec(self) -> Dict[str, Any]:
        return {"horizon_s": self.horizon_s,
                "segments_per_app_day": self.segments_per_app_day,
                "seeds_per_grid_point": self.seeds_per_grid_point,
                "min_invocations_per_segment": self.min_invocations_per_segment,
                "materialise_per_family_split": self.materialise_per_family_split,
                "reconstruction": self.reconstruction.spec()}


def plan_corpus(extract: AzureExtract, cfg: CorpusConfig,
                service_model: ServiceTimeModel,
                capacity: Optional[Capacity] = None) -> List[Dict[str, Any]]:
    """Every trace's *spec* (no arrays yet), split-assigned and seeded.

    Deterministic: same extract + same config -> same list, in the same order.
    """
    capacity = capacity or default_capacity()
    entries: List[Dict[str, Any]] = []
    grid = parameter_grid(cfg.horizon_s)

    for split in SPLIT_ORDER:
        base_seed = SEED_BANDS[split][0]
        counter = 0
        # -- replay half --------------------------------------------------
        for app in sorted(extract.apps_in_split(split), key=lambda a: a.app_hash):
            for day in DAY_SPLITS[split]:
                starts = segment_starts(extract, app.app_hash, day,
                                        horizon_s=cfg.horizon_s,
                                        n_segments=cfg.segments_per_app_day,
                                        min_invocations=cfg.min_invocations_per_segment)
                for st in starts:
                    seed = base_seed + counter
                    counter += 1
                    entries.append({
                        "trace_id": f"azure-{app.app_hash[:8]}-d{day:02d}-m{st:04d}-s{seed}",
                        "kind": "azure_replay", "family": "azure_replay", "split": split,
                        "seed": seed, "horizon_s": cfg.horizon_s,
                        "app_hash": app.app_hash, "regime": app.regime,
                        "volume_tercile": app.volume_tercile,
                        "dominant_trigger": app.dominant_trigger,
                        "day": day, "segment_start_minute": st,
                        "tests": f"{app.regime} regime, {app.volume_tercile} volume tercile, "
                                 f"{app.dominant_trigger} trigger: "
                                 + FAMILY_INTENT["azure_replay"],
                        "reconstruction": cfg.reconstruction.spec(),
                    })
        # -- synthetic half -----------------------------------------------
        for gi, pt in enumerate(grid):
            for k in range(cfg.seeds_per_grid_point):
                seed = base_seed + 50_000 + gi * 100 + k
                sched = pt["schedule"]
                name = f"{pt['family']}-g{gi:03d}-s{seed}"
                sspec = SyntheticSpec(name=name, family=pt["family"], horizon_s=cfg.horizon_s,
                                      schedule=sched, service=service_model,
                                      capacity=capacity, seed=seed, tests=pt["tests"])
                entries.append({
                    "trace_id": name, "kind": "synthetic", "family": pt["family"],
                    "split": split, "seed": seed, "horizon_s": cfg.horizon_s,
                    "grid_index": gi, "tests": pt["tests"],
                    "synthetic_spec": sspec.spec(),
                })
    return entries


def source_from_entry(entry: Dict[str, Any], extract: Optional[AzureExtract] = None):
    """Rebuild the `WorkloadSource` for one manifest entry from the spec alone."""
    if entry["kind"] == "synthetic":
        return synthetic_from_spec(entry["synthetic_spec"])
    if extract is None:
        raise ValueError("azure_replay entries need the extract")
    spec = ReconstructionSpec(**entry["reconstruction"])
    return AzureReplaySource(extract, entry["app_hash"],
                             [(entry["day"], entry["segment_start_minute"])],
                             horizon_s=entry["horizon_s"], spec=spec)


def realise_entry(entry: Dict[str, Any], extract: Optional[AzureExtract] = None,
                  ) -> Dict[str, Any]:
    """Rebuild one trace's arrays from its spec.  The regeneration path."""
    src = source_from_entry(entry, extract)
    arr = src.arrays(0, seed=entry["seed"])
    # Arrival times are already relative to the episode start for both kinds.
    t = arr["t_arrival"]
    return {"t_arrival": t, "service_time_s": arr["service_time_s"],
            "horizon_s": float(entry["horizon_s"]),
            "diagnostics": arr.get("diagnostics", {}), "source": src}


def build_corpus(extract: AzureExtract, out_dir: str, cfg: Optional[CorpusConfig] = None,
                 *, extract_paths: Optional[Dict[str, str]] = None,
                 progress_every: int = 100) -> Dict[str, Any]:
    """Plan, realise, checksum and write the corpus.  Returns the manifest."""
    cfg = cfg or CorpusConfig()
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "traces"), exist_ok=True)

    service_model, service_report = pooled_service_icdf(extract, cfg.reconstruction)
    capacity = default_capacity()
    entries = plan_corpus(extract, cfg, service_model, capacity)

    # how many of each (family, split) to materialise as JSONL
    quota: Dict[Tuple[str, str], int] = {}
    written: List[str] = []
    for i, e in enumerate(entries):
        out = realise_entry(e, extract)
        t, s = out["t_arrival"], out["service_time_s"]
        e["n_requests"] = int(t.size)
        e["sha256"] = trace_digest(t, s)
        # Per-trace honesty flags, so a consumer that reads one entry in isolation
        # still cannot mistake a reconstructed or synthetic trace for measured data.
        e["calibrated"] = False
        e["is_measured_data"] = False
        e["summary"] = trace_summary(t, s, cfg.horizon_s)
        if e["kind"] == "azure_replay":
            e["reconstruction_diagnostics"] = out["diagnostics"]
        key = (e["family"], e["split"])
        if quota.get(key, 0) < cfg.materialise_per_family_split:
            quota[key] = quota.get(key, 0) + 1
            rel = os.path.join("traces", e["split"], f"{e['trace_id']}.jsonl")
            path = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(canonical_lines(t, s))
            e["file"] = rel
            e["file_sha256"] = sha256_file(path)
            if e["file_sha256"] != e["sha256"]:
                raise AssertionError(f"{rel}: file digest != trace digest")
            written.append(rel)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  realised {i + 1}/{len(entries)}", flush=True)

    grid = parameter_grid(cfg.horizon_s)
    app_hashes = sorted({e["app_hash"] for e in entries if e["kind"] == "azure_replay"})
    manifest = {
        "schema": CORPUS_SCHEMA,
        # The only non-deterministic field in the manifest.  Every trace digest and
        # every other field is a pure function of (extract, config), so two builds
        # differ in this line alone.
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": extract.meta.get("dataset"),
        "license": extract.meta.get("license"),
        "required_attribution": extract.meta.get("required_attribution"),
        "azure_app_hashes": app_hashes,
        "selection_report": extract.meta.get("selection"),
        "population_comparison": extract.meta.get("population_comparison"),
        "duration_table_audit": extract.meta.get("duration_table_audit"),
        "parameter_grid": [{"family": g["family"], "tests": g["tests"],
                            "schedule": g["schedule"].spec()} for g in grid],
        "regeneration_command":
            "python -m kubegym.workloads.cli regenerate --corpus <dir> "
            "--trace-id <trace_id> --out <path.jsonl>   # rebuilds one trace from its spec "
            "and checks its sha256 against this manifest; "
            "`... cli verify --corpus <dir>` does it for every trace",
        "calibrated": False,
        "is_measured_data": False,
        "calibration_note":
            "UNCALIBRATED. Two independent reasons. (1) The Azure half is a RECONSTRUCTION: the "
            "source provides per-minute invocation counts and duration percentiles of "
            "30-second averages, not arrival instants or per-request durations, so arrival "
            "placement, function attribution and every service time are modelling assumptions "
            "of this benchmark (see reconstruction_provenance). (2) The rate unit and the "
            "synthetic service-time model derive from placeholder constants in "
            "request_service_default.json. No number computed over this corpus is an empirical "
            "result, and no statement about it may be phrased as a measurement of the Azure "
            "workload.",
        "config": cfg.spec(),
        "families": sorted({e["family"] for e in entries}),
        "family_intent": FAMILY_INTENT,
        "n_traces": len(entries),
        "n_traces_by_split": {s: sum(1 for e in entries if e["split"] == s)
                              for s in SPLIT_ORDER},
        "n_traces_by_family": {f: sum(1 for e in entries if e["family"] == f)
                               for f in sorted({e["family"] for e in entries})},
        "n_requests_total": int(sum(e["n_requests"] for e in entries)),
        "materialised_files": written,
        "materialisation_policy":
            f"{cfg.materialise_per_family_split} traces per (family, split) are shipped as "
            "canonical JSONL; every other trace is shipped as its spec only and is regenerable "
            "with `python -m kubegym.workloads.cli regenerate`. Each entry's sha256 is the "
            "digest of the canonical JSONL bytes, so a materialised file and its manifest entry "
            "are checkable with sha256sum.",
        "split_rule": split_rule_text(),
        "day_splits": {k: list(v) for k, v in DAY_SPLITS.items()},
        "seed_bands": {k: list(v) for k, v in SEED_BANDS.items()},
        "azure_extract": {
            "schema": AzureExtract.SCHEMA,
            "paths": extract_paths or {},
            "dataset": extract.meta.get("dataset"),
            "license": extract.meta.get("license"),
            "required_attribution": extract.meta.get("required_attribution"),
            "staging_manifest_checksums": (extract.meta.get("staging_manifest", {})
                                           .get("files", {})),
            "selection": extract.meta.get("selection"),
            "population_comparison": extract.meta.get("population_comparison"),
            "duration_table_audit": extract.meta.get("duration_table_audit"),
            "source_facts_verified_against_documentation":
                extract.meta.get("source_facts_verified_against_documentation"),
            "unverified": extract.meta.get("unverified"),
            "app_hashes_by_split": {
                s: sorted(a.app_hash for a in extract.apps_in_split(s)) for s in SPLIT_ORDER},
        },
        "reconstruction_provenance": cfg.reconstruction.provenance(),
        "synthetic_service_time_model": {**service_model.provenance(), "fit": service_report},
        "capacity": capacity.provenance(),
        "parameter_grid_size": len(grid),
        "rng_scheme": {
            "streams": list(__import__("kubegym.workloads.seeding", fromlist=["x"]).STREAM_NAMES),
            "how": "every stochastic component draws from a SHA256-derived substream of "
                   "(seed, component name, trace identity); numpy Generators draw only "
                   "Generator.random(n) and all other variates come from explicit inverse "
                   "transforms in kubegym/workloads/seeding.py, so digests are stable across "
                   "numpy versions",
        },
        "traces": entries,
    }
    return manifest


def split_rule_text() -> str:
    return ("Applications: each of the 45 selected Azure applications belongs to exactly one "
            "split (the three picks of each regime x volume-tercile cell go to train/dev/test), "
            "so application sets are disjoint. Time: source days are partitioned "
            "train{1,2,3,6,8,9,13} / dev{4,10,14} / test{5,7,11,12} and a replay trace uses only "
            "its split's days, so no minute of source time is shared; segments within a day are "
            "non-overlapping whole hours. Seeds: disjoint bands train[0,1e5) dev[1e5,2e5) "
            "test[2e5,3e5). Synthetic parameter grid points are shared across splits on purpose "
            "(the question is generalisation over realisations, not over hyperparameters); the "
            "synthetic service-time model is fitted on train applications and train days only.")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_splits(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Programmatic check of every disjointness claim in the split rule."""
    tr = manifest["traces"]
    apps = {s: set() for s in SPLIT_ORDER}
    days = {s: set() for s in SPLIT_ORDER}
    seeds = {s: set() for s in SPLIT_ORDER}
    segs = {s: set() for s in SPLIT_ORDER}
    for e in tr:
        sp = e["split"]
        seeds[sp].add(e["seed"])
        if e["kind"] == "azure_replay":
            apps[sp].add(e["app_hash"])
            days[sp].add(e["day"])
            segs[sp].add((e["app_hash"], e["day"], e["segment_start_minute"]))
    def pairwise(d):
        bad = {}
        for i, a in enumerate(SPLIT_ORDER):
            for b in SPLIT_ORDER[i + 1:]:
                inter = d[a] & d[b]
                if inter:
                    bad[f"{a}&{b}"] = sorted(list(inter))[:10]
        return bad
    seg_dupes = {}
    seen: Dict[Tuple[Any, ...], str] = {}
    for e in tr:
        if e["kind"] != "azure_replay":
            continue
        k = (e["app_hash"], e["day"], e["segment_start_minute"])
        if k in seen and seen[k] != e["split"]:
            seg_dupes[str(k)] = [seen[k], e["split"]]
        seen[k] = e["split"]
    seed_collisions = [e["trace_id"] for e in tr
                       if not (SEED_BANDS[e["split"]][0] <= e["seed"] < SEED_BANDS[e["split"]][1])]
    all_seeds = [e["seed"] for e in tr]
    return {
        "app_overlap": pairwise(apps),
        "day_overlap": pairwise(days),
        "seed_overlap": pairwise(seeds),
        "segment_shared_across_splits": seg_dupes,
        "seeds_outside_their_band": seed_collisions,
        "duplicate_seeds_within_corpus": len(all_seeds) - len(set(all_seeds)),
        "apps_per_split": {s: len(apps[s]) for s in SPLIT_ORDER},
        "days_per_split": {s: sorted(days[s]) for s in SPLIT_ORDER},
        "regimes_per_split": {
            s: sorted({e["regime"] for e in tr
                       if e["split"] == s and e["kind"] == "azure_replay"})
            for s in SPLIT_ORDER},
        "ok": (not pairwise(apps) and not pairwise(days) and not pairwise(seeds)
               and not seg_dupes and not seed_collisions
               and len(all_seeds) == len(set(all_seeds))),
    }


def verify_manifest(manifest: Dict[str, Any], extract: AzureExtract, *,
                    limit: Optional[int] = None, stride: int = 1,
                    corpus_dir: Optional[str] = None) -> Dict[str, Any]:
    """Regenerate traces from their specs and compare digests.

    `limit`/`stride` subsample for a fast check; the default checks every trace.
    Also re-hashes any materialised file found under `corpus_dir`.
    """
    entries = manifest["traces"][::stride]
    if limit is not None:
        entries = entries[:limit]
    mismatches: List[Dict[str, Any]] = []
    file_mismatches: List[str] = []
    for e in entries:
        out = realise_entry(e, extract)
        got = trace_digest(out["t_arrival"], out["service_time_s"])
        if got != e["sha256"]:
            mismatches.append({"trace_id": e["trace_id"], "expected": e["sha256"], "got": got})
        if corpus_dir and e.get("file"):
            p = os.path.join(corpus_dir, e["file"])
            if os.path.exists(p) and sha256_file(p) != e["sha256"]:
                file_mismatches.append(e["file"])
    return {"n_checked": len(entries), "n_mismatch": len(mismatches),
            "mismatches": mismatches[:20], "n_files_checked":
                sum(1 for e in entries if e.get("file")),
            "file_mismatches": file_mismatches,
            "ok": not mismatches and not file_mismatches}


def write_manifest(manifest: Dict[str, Any], path: str) -> str:
    with open(path, "w", newline="\n") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return sha256_file(path)


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)
