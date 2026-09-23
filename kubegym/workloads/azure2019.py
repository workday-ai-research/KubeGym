"""Azure Functions 2019 replay: application selection, extract, and the source.

DATASET
-------
Azure Functions Trace 2019 (revision 2, 20200618), CC-BY.  Required attribution:

    Shahrad, Fonseca, Goiri, Chaudhry, Batum, Cooke, Laureano, Tresness,
    Russinovich, Bianchini.  "Serverless in the Wild: Characterizing and
    Optimizing the Serverless Workload at a Large Cloud Provider."
    USENIX ATC 2020.

The staged parquet tables and their per-file checksums are described by
`azure2019_staging_manifest.json`, which is embedded in the corpus manifest.
This module never re-downloads anything.

WHAT REPLAY MEANS HERE
----------------------
"Replay" is the *arrival count series* of a real application over a real time
window.  The individual arrival instants and the individual service times are
**reconstructed**, not replayed -- see `reconstruction.py`.  Every trace in the
corpus therefore carries the reconstruction provenance block, and no number
derived from it may be described as a measurement of the Azure workload.

APPLICATION SELECTION
---------------------
`select_applications()` is a deterministic function of the staged tables and a
fixed salt: no RNG, no manual curation.  The rule, in order:

  1. **Candidate filter.**  Keep applications that (a) appear on all 14 days of
     the invocation table, (b) have duration rows on all 14 days, (c) have a
     trigger-mix row, and (d) have a 14-day peak per-minute invocation count at
     or below `peak_per_minute_cap` (default 4800/min = 80 req/s).

     (d) is a **capacity-envelope filter, and it is a selection bias that must
     be reported.**  The shipped `request_service_default.json` allows at most
     `n_replicas_max = 20` replicas at `rs_max_concurrency = 4`, i.e. 80
     concurrent requests; at the Count-weighted mean reconstructed
     service time of about 0.90 s that is roughly 89 req/s of steady capacity.
     An application whose peak minute exceeds that cannot be served by any
     controller in the suite, so a trace built from it would measure the
     platform ceiling rather than the controller.  The filter removes ~1.5% of
     continuously-present applications which nevertheless carry ~79% of total
     invocation volume; `population_report()` reports both numbers.

  2. **Regime label**, from 14-day statistics, evaluated in this order (the
     order matters: a diurnal application also has inflated dispersion, and a
     very sparse series has a large relative Fourier amplitude by construction):

         sparse        zero_minute_fraction >= 0.95
         intermittent  0.60 <= zero_minute_fraction < 0.95
         diurnal       relative 24 h Fourier amplitude >= 0.5
         bursty        index of dispersion >= 4.0
         steady        otherwise

  3. **Volume tercile** within each regime, by 14-day total invocations.

  4. **Trigger spread and split assignment.**  Within each of the 15
     (regime x tercile) cells, candidates are ordered by
     `sha256(salt | HashApp)`, then taken round-robin across dominant trigger
     groups (so the picks in a cell are spread over trigger groups where the
     cell has more than one), and the first `apps_per_cell` are kept.  Pick `i`
     of a cell goes to split `SPLIT_ORDER[i]`, which makes the application sets
     of the three splits disjoint by construction.

Everything above is recorded in the extract's metadata, so the selection is
auditable without re-running it.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.service import Request
from ..core.workload import WorkloadSource
from .reconstruction import (PERCENTILE_COLUMNS, ReconstructionSpec, attribute_functions,
                             audit_duration_table, build_icdf_knots, invert_icdf,
                             minmax_usable, place_arrivals)
from .seeding import np_substream, substream

ATTRIBUTION = ("Shahrad, Fonseca, Goiri, Chaudhry, Batum, Cooke, Laureano, Tresness, "
               "Russinovich, Bianchini. 'Serverless in the Wild: Characterizing and "
               "Optimizing the Serverless Workload at a Large Cloud Provider', "
               "USENIX ATC 2020.")
DATASET_NAME = "Azure Functions Trace 2019 (revision 2, 20200618)"
DATASET_LICENSE = "CC-BY Attribution License (per dataset LICENSE)"
DATASET_DOC_URL = ("https://github.com/Azure/AzurePublicDataset/blob/master/"
                   "AzureFunctionsDataset2019.md")

#: Fixed salt for the deterministic selection order.  Changing it reshuffles the
#: corpus, so it is part of the corpus identity and is recorded in the manifest.
SELECTION_SALT = "kubegym-azure2019-selection-v1"

REGIMES: Tuple[str, ...] = ("sparse", "intermittent", "diurnal", "bursty", "steady")
TERCILES: Tuple[str, ...] = ("low", "mid", "high")
SPLIT_ORDER: Tuple[str, ...] = ("train", "dev", "test")

#: Day assignment.  Days 6, 7, 13 and 14 of the source contain markedly fewer
#: distinct functions than the others (~36k vs ~47k) at comparable invocation
#: volume; the 7-day spacing is consistent with a weekend effect but the source
#: documents no collection-day mapping, so the calendar alignment is UNVERIFIED.
#: The assignment below is non-contiguous on purpose so that each split contains
#: at least one of those four days, rather than concentrating them in one split.
DAY_SPLITS: Dict[str, Tuple[int, ...]] = {
    "train": (1, 2, 3, 6, 8, 9, 13),
    "dev": (4, 10, 14),
    "test": (5, 7, 11, 12),
}

#: Seed bands, disjoint across splits.  A trace's seed is
#: `SEED_BANDS[split][0] + index`, so no realisation is shared.
SEED_BANDS: Dict[str, Tuple[int, int]] = {
    "train": (0, 100_000),
    "dev": (100_000, 200_000),
    "test": (200_000, 300_000),
}

MINUTE_COLUMNS: Tuple[str, ...] = tuple(str(i) for i in range(1, 1441))
N_DAYS = 14
MINUTES_PER_DAY = 1440


# ---------------------------------------------------------------------------
# Population statistics
# ---------------------------------------------------------------------------
def app_day_statistics(invocations_parquet: str) -> "Any":
    """Per (application, day) invocation statistics, streamed by row group.

    Returns a pandas frame with columns `HashApp, day, total, peak, n_nonzero,
    var_minute, fourier24_amp`.  `fourier24_amp` is the amplitude of the
    fundamental (24 h) component of that day's 1440-point minute series.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(invocations_parquet)
    frames = []
    for rg in range(pf.num_row_groups):
        tb = pf.read_row_group(rg)
        app = tb.column("HashApp").to_pylist()
        day = np.asarray(tb.column("day").to_pylist(), dtype=np.int16)
        M = np.column_stack([tb.column(c).to_numpy() for c in MINUTE_COLUMNS]).astype(np.float64)
        ft = np.fft.rfft(M - M.mean(axis=1, keepdims=True), axis=1)
        frames.append(pd.DataFrame({
            "HashApp": app, "day": day,
            "total": M.sum(axis=1), "peak": M.max(axis=1),
            "n_nonzero": (M > 0).sum(axis=1), "var_minute": M.var(axis=1),
            "fourier24_amp": np.abs(ft[:, 1]) * 2.0 / MINUTES_PER_DAY,
        }))
    return pd.concat(frames, ignore_index=True)


def app_aggregates(app_day: "Any", durations: "Any", trigger_mix: "Any") -> "Any":
    """Collapse per-app-day statistics to 14-day per-application features."""
    import pandas as pd

    g = app_day.groupby("HashApp")
    agg = g.agg(n_days=("day", "nunique"), total=("total", "sum"), peak=("peak", "max"),
                n_nonzero=("n_nonzero", "sum"), var_minute=("var_minute", "mean"),
                fourier24_amp=("fourier24_amp", "mean")).reset_index()
    agg["n_minutes"] = agg.n_days * MINUTES_PER_DAY
    agg["zero_minute_fraction"] = 1.0 - agg.n_nonzero / agg.n_minutes
    agg["mean_rate_per_minute"] = agg.total / agg.n_minutes
    denom = agg.mean_rate_per_minute.replace(0.0, np.nan)
    agg["index_of_dispersion"] = agg.var_minute / denom
    agg["rel_fourier24"] = agg.fourier24_amp / denom
    dur_days = durations.groupby("HashApp").day.nunique().rename("duration_days")
    agg = agg.merge(dur_days, left_on="HashApp", right_index=True, how="left")
    tot = trigger_mix.groupby("HashApp").invocations.sum().rename("trigger_total")
    dom = (trigger_mix.sort_values(["invocations", "Trigger"], ascending=[False, True])
           .groupby("HashApp").first().rename(columns={"Trigger": "dominant_trigger"}))
    agg = agg.merge(dom[["dominant_trigger"]], left_on="HashApp", right_index=True, how="left")
    agg = agg.merge(tot, left_on="HashApp", right_index=True, how="left")
    return agg


def label_regime(row) -> str:
    """The regime label of one aggregated application row.  Order matters."""
    if row.zero_minute_fraction >= 0.95:
        return "sparse"
    if row.zero_minute_fraction >= 0.60:
        return "intermittent"
    if row.rel_fourier24 >= 0.5:
        return "diurnal"
    if row.index_of_dispersion >= 4.0:
        return "bursty"
    return "steady"


def _order_key(app_hash: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}|{app_hash}".encode()).hexdigest()


def select_applications(agg: "Any", *, apps_per_cell: int = 3,
                        peak_per_minute_cap: float = 4800.0,
                        ) -> Tuple["Any", Dict[str, Any]]:
    """Deterministic stratified selection.  See the module docstring for the rule."""
    import pandas as pd

    n_all = int(len(agg))
    full = agg[(agg.n_days == N_DAYS)]
    cand = agg[(agg.n_days == N_DAYS) & (agg.duration_days == N_DAYS)
               & (agg.dominant_trigger.notna()) & (agg.peak <= peak_per_minute_cap)].copy()
    cand["regime"] = cand.apply(label_regime, axis=1)
    cand["volume_tercile"] = cand.groupby("regime").total.transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=list(TERCILES)))
    cand["order_key"] = cand.HashApp.map(_order_key)

    picks: List[Dict[str, Any]] = []
    for regime in REGIMES:
        for tercile in TERCILES:
            cell = cand[(cand.regime == regime) & (cand.volume_tercile == tercile)]
            cell = cell.sort_values("order_key")
            # round-robin across dominant trigger groups, each group in order_key order
            groups: Dict[str, List[Any]] = {}
            for _, r in cell.iterrows():
                groups.setdefault(r.dominant_trigger, []).append(r)
            group_names = sorted(groups, key=lambda t: groups[t][0].order_key)
            chosen: List[Any] = []
            k = 0
            while len(chosen) < apps_per_cell and any(groups[g] for g in group_names):
                g = group_names[k % len(group_names)]
                if groups[g]:
                    chosen.append(groups[g].pop(0))
                k += 1
            if len(chosen) < apps_per_cell:
                raise ValueError(f"cell ({regime}, {tercile}) has only {len(chosen)} "
                                 f"candidates, need {apps_per_cell}")
            for i, r in enumerate(chosen):
                picks.append({"HashApp": r.HashApp, "regime": regime, "volume_tercile": tercile,
                              "split": SPLIT_ORDER[i % len(SPLIT_ORDER)],
                              "dominant_trigger": r.dominant_trigger,
                              "total_invocations_14d": float(r.total),
                              "peak_per_minute": float(r.peak),
                              "mean_rate_per_minute": float(r.mean_rate_per_minute),
                              "zero_minute_fraction": float(r.zero_minute_fraction),
                              "index_of_dispersion": float(r.index_of_dispersion),
                              "rel_fourier24": float(r.rel_fourier24)})
    sel = pd.DataFrame(picks)

    excluded_by_cap = full[full.peak > peak_per_minute_cap]
    report = {
        "rule": "see kubegym/workloads/azure2019.py module docstring; deterministic, no RNG",
        "selection_salt": SELECTION_SALT,
        "apps_per_cell": int(apps_per_cell),
        "peak_per_minute_cap": float(peak_per_minute_cap),
        "n_apps_in_source": n_all,
        "n_apps_present_all_14_days": int(len(full)),
        "n_candidates_after_filter": int(len(cand)),
        "n_selected": int(len(sel)),
        "candidate_regime_counts": cand.regime.value_counts().to_dict(),
        "capacity_envelope_exclusion": {
            "n_apps_excluded": int(len(excluded_by_cap)),
            "fraction_of_14day_apps_excluded": float(len(excluded_by_cap) / max(1, len(full))),
            "fraction_of_14day_invocation_volume_excluded":
                float(excluded_by_cap.total.sum() / max(1.0, float(full.total.sum()))),
            "why": "peak per-minute rate above the modelled platform's serving capacity "
                   "(n_replicas_max=20 x rs_max_concurrency=4). A trace above the ceiling "
                   "measures the ceiling, not the controller.",
            "direction_of_bias": "the corpus under-represents the very-high-volume head of the "
                                 "Azure population, which is where most invocation VOLUME lives "
                                 "even though it is a small share of applications",
        },
        "day_splits": {k: list(v) for k, v in DAY_SPLITS.items()},
        "seed_bands": {k: list(v) for k, v in SEED_BANDS.items()},
    }
    return sel, report


def population_report(agg: "Any", sel: "Any") -> Dict[str, Any]:
    """Selected subset vs the full population, on every selection axis."""
    def q(frame, col):
        s = frame[col].replace([np.inf, -np.inf], np.nan).dropna()
        if not len(s):
            return None
        return {"n": int(len(s)), "p10": float(s.quantile(0.10)),
                "p50": float(s.quantile(0.50)), "p90": float(s.quantile(0.90)),
                "mean": float(s.mean())}

    full = agg[agg.n_days == N_DAYS]
    selagg = agg[agg.HashApp.isin(set(sel.HashApp))]
    cols = ["total", "peak", "mean_rate_per_minute", "zero_minute_fraction",
            "index_of_dispersion", "rel_fourier24"]
    out: Dict[str, Any] = {
        "axes": {c: {"population_all_14day_apps": q(full, c), "selected": q(selagg, c)}
                 for c in cols},
        "regime_share": {
            "population": (full.assign(regime=full.apply(label_regime, axis=1))
                           .regime.value_counts(normalize=True).round(4).to_dict()),
            "selected": sel.regime.value_counts(normalize=True).round(4).to_dict(),
        },
        "dominant_trigger_share": {
            "population": full.dominant_trigger.value_counts(normalize=True).round(4).to_dict(),
            "selected": sel.dominant_trigger.value_counts(normalize=True).round(4).to_dict(),
        },
        "reading": "The selection is stratified, not representative by design: it oversamples "
                   "the rare regimes (diurnal, bursty) and the high-volume terciles relative to "
                   "their population share so that every regime an autoscaler faces is present "
                   "with equal weight. The population columns are given so the distortion is "
                   "quantified rather than hidden. Note the population itself is dominated by "
                   "very low-rate applications (median 14-day total ~4.5k invocations, median "
                   "zero-minute fraction ~0.80), and the corpus keeps that regime as two of its "
                   "five strata (sparse, intermittent) rather than dropping it.",
    }
    return out


# ---------------------------------------------------------------------------
# The extract
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppRecord:
    """One selected application, as stored in the extract."""
    app_hash: str
    regime: str
    volume_tercile: str
    split: str
    dominant_trigger: str
    stats: Dict[str, float]


class AzureExtract:
    """The small, checksummed subset of the staged tables the corpus needs.

    Holds, for each selected application: the per-minute invocation counts for
    all 14 days, and the per-function-day duration percentile knots.  About
    4 MB for 45 applications, so it ships inside the corpus tarball and the
    replay traces are regenerable without the 2 GB source.
    """

    SCHEMA = "kubegym-azure-extract-1"

    def __init__(self, counts: np.ndarray, apps: Sequence[AppRecord], meta: Dict[str, Any],
                 dur: Dict[str, np.ndarray]):
        if counts.shape != (len(apps), N_DAYS, MINUTES_PER_DAY):
            raise ValueError(f"counts shape {counts.shape} does not match "
                             f"{(len(apps), N_DAYS, MINUTES_PER_DAY)}")
        self.counts = counts
        self.apps = list(apps)
        self.meta = dict(meta)
        self.dur = dur
        self._app_index = {a.app_hash: i for i, a in enumerate(self.apps)}
        # per (app, day) -> (function slice) index, built once
        self._fd: Dict[Tuple[int, int], np.ndarray] = {}
        key = self.dur["app_idx"].astype(np.int64) * 100 + self.dur["day"].astype(np.int64)
        order = np.argsort(key, kind="stable")
        for k in np.unique(key):
            sel = order[np.searchsorted(key[order], k, "left"):
                        np.searchsorted(key[order], k, "right")]
            self._fd[(int(k // 100), int(k % 100))] = sel

    # -- lookup --------------------------------------------------------
    def app_index(self, app_hash: str) -> int:
        return self._app_index[app_hash]

    def app(self, app_hash: str) -> AppRecord:
        return self.apps[self._app_index[app_hash]]

    def apps_in_split(self, split: str) -> List[AppRecord]:
        return [a for a in self.apps if a.split == split]

    def minute_counts(self, app_hash: str, day: int) -> np.ndarray:
        return self.counts[self._app_index[app_hash], day - 1]

    def function_days(self, app_hash: str, day: int) -> np.ndarray:
        """Row indices into `self.dur` for one (application, day)."""
        return self._fd.get((self._app_index[app_hash], int(day)), np.zeros(0, dtype=np.int64))

    # -- io ------------------------------------------------------------
    def save(self, npz_path: str, json_path: str) -> Dict[str, str]:
        np.savez_compressed(npz_path, counts=self.counts,
                            **{f"dur_{k}": v for k, v in self.dur.items()})
        blob = {"schema": self.SCHEMA,
                "apps": [{"app_hash": a.app_hash, "regime": a.regime,
                          "volume_tercile": a.volume_tercile, "split": a.split,
                          "dominant_trigger": a.dominant_trigger, "stats": a.stats}
                         for a in self.apps],
                "meta": self.meta}
        with open(json_path, "w", newline="\n") as fh:
            json.dump(blob, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return {"npz": npz_path, "json": json_path,
                "npz_sha256": sha256_file(npz_path), "json_sha256": sha256_file(json_path)}

    @classmethod
    def load(cls, npz_path: str, json_path: str) -> "AzureExtract":
        z = np.load(npz_path, allow_pickle=False)
        with open(json_path) as fh:
            blob = json.load(fh)
        if blob.get("schema") != cls.SCHEMA:
            raise ValueError(f"extract schema {blob.get('schema')!r} != {cls.SCHEMA!r}")
        apps = [AppRecord(app_hash=a["app_hash"], regime=a["regime"],
                          volume_tercile=a["volume_tercile"], split=a["split"],
                          dominant_trigger=a["dominant_trigger"], stats=a["stats"])
                for a in blob["apps"]]
        dur = {k[4:]: z[k] for k in z.files if k.startswith("dur_")}
        return cls(z["counts"], apps, blob["meta"], dur)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_extract(invocations_parquet: str, durations_parquet: str,
                  trigger_mix_parquet: str, staging_manifest: str, *,
                  apps_per_cell: int = 3, peak_per_minute_cap: float = 4800.0,
                  ) -> Tuple[AzureExtract, Dict[str, Any]]:
    """Run the selection against the staged tables and build the extract.

    This is the only function in the package that touches the 2 GB-derived
    parquet tables; everything downstream reads the extract.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    with open(staging_manifest) as fh:
        staging = json.load(fh)
    durations = pd.read_parquet(durations_parquet)
    trigger_mix = pd.read_parquet(trigger_mix_parquet)
    app_day = app_day_statistics(invocations_parquet)
    agg = app_aggregates(app_day, durations, trigger_mix)
    sel, sel_report = select_applications(agg, apps_per_cell=apps_per_cell,
                                          peak_per_minute_cap=peak_per_minute_cap)
    pop = population_report(agg, sel)

    chosen = list(sel.HashApp)
    chosen_set = set(chosen)
    counts = np.zeros((len(chosen), N_DAYS, MINUTES_PER_DAY), dtype=np.int32)
    idx_of = {h: i for i, h in enumerate(chosen)}
    pf = pq.ParquetFile(invocations_parquet)
    for rg in range(pf.num_row_groups):
        tb = pf.read_row_group(rg)
        app = tb.column("HashApp").to_pylist()
        keep = [i for i, a in enumerate(app) if a in chosen_set]
        if not keep:
            continue
        day = np.asarray(tb.column("day").to_pylist(), dtype=np.int64)
        M = np.column_stack([tb.column(c).to_numpy() for c in MINUTE_COLUMNS])
        for i in keep:
            counts[idx_of[app[i]], day[i] - 1, :] = M[i]

    d = durations[durations.HashApp.isin(chosen_set)].copy()
    audit_selected = audit_duration_table(d)
    audit_full = audit_duration_table(durations)
    pct = d[list(PERCENTILE_COLUMNS)].to_numpy(dtype=np.float64)
    mn = d["Minimum"].to_numpy(dtype=np.float64)
    mx = d["Maximum"].to_numpy(dtype=np.float64)
    usable = np.array([minmax_usable(mn[i], mx[i], pct[i, 0], pct[i, -1])
                       for i in range(len(d))], dtype=bool)
    func_codes = d.HashFunction.astype("category").cat.codes.to_numpy()
    dur = {
        "app_idx": np.array([idx_of[a] for a in d.HashApp], dtype=np.int32),
        "day": d.day.to_numpy(dtype=np.int16),
        "func_code": func_codes.astype(np.int32),
        "count": d.Count.to_numpy(dtype=np.int64),
        "pct_ms": pct,
        "minimum_ms": np.nan_to_num(mn, nan=-1.0),
        "maximum_ms": np.nan_to_num(mx, nan=-1.0),
        "minmax_usable": usable,
    }

    meta = {
        "dataset": DATASET_NAME,
        "license": DATASET_LICENSE,
        "required_attribution": ATTRIBUTION,
        "documentation_url": DATASET_DOC_URL,
        "staging_manifest": staging,
        "selection": sel_report,
        "population_comparison": pop,
        "duration_table_audit": {"selected_apps": audit_selected, "full_table": audit_full},
        "source_facts_verified_against_documentation": [
            "execution time is in milliseconds and does NOT include cold start time (note 4)",
            "the percentile_Average_* columns are weighted percentiles of 30-second-interval "
            "AVERAGES, not of individual invocation durations (note 7)",
            "Minimum/Maximum are the true extremes but were not recorded for a few functions "
            "owing to a field naming issue in some Azure Functions runtime versions (note 6)",
            "the durations-table Count comes from a different log than the invocation counts and "
            "may rarely diverge (note 5)",
            "invocation counts are recorded after the functions execute (note 3)",
        ],
        "unverified": [
            "the calendar alignment of days 1-14 (the source documents no collection-day "
            "mapping), so the weekday/weekend interpretation of the low-function-count days "
            "6, 7, 13, 14 is a conjecture and is not asserted anywhere in the corpus",
        ],
    }
    apps = [AppRecord(app_hash=r.HashApp, regime=r.regime, volume_tercile=r.volume_tercile,
                      split=r.split, dominant_trigger=r.dominant_trigger,
                      stats={k: float(getattr(r, k)) for k in
                             ("total_invocations_14d", "peak_per_minute",
                              "mean_rate_per_minute", "zero_minute_fraction",
                              "index_of_dispersion", "rel_fourier24")})
            for r in sel.itertuples()]
    return AzureExtract(counts, apps, meta, dur), {"selection": sel_report,
                                                   "population": pop,
                                                   "duration_audit": audit_selected}


# ---------------------------------------------------------------------------
# Segment selection
# ---------------------------------------------------------------------------
def segment_starts(extract: AzureExtract, app_hash: str, day: int, *,
                   horizon_s: float, n_segments: int,
                   min_invocations: int = 5) -> List[int]:
    """Deterministic non-overlapping segment start minutes for one (app, day).

    The day is tiled into `1440 / horizon_minutes` non-overlapping windows.
    Windows with at least `min_invocations` recorded invocations are eligible;
    eligible windows are ordered by `sha256(salt | app | day | start)` and the
    first `n_segments` are taken.  If fewer are eligible, the highest-volume
    windows are used instead (and there may be fewer than `n_segments`).
    """
    hm = int(round(horizon_s / 60.0))
    if hm <= 0 or MINUTES_PER_DAY % hm:
        raise ValueError(f"horizon_s={horizon_s} must be a whole-minute divisor of a day")
    c = extract.minute_counts(app_hash, day)
    starts = np.arange(0, MINUTES_PER_DAY, hm)
    vols = np.array([int(c[s:s + hm].sum()) for s in starts])
    eligible = [int(s) for s, v in zip(starts, vols) if v >= min_invocations]
    if len(eligible) >= n_segments:
        eligible.sort(key=lambda s: hashlib.sha256(
            f"{SELECTION_SALT}|seg|{app_hash}|{day}|{s}".encode()).hexdigest())
        return sorted(eligible[:n_segments])
    order = np.argsort(-vols, kind="stable")
    return sorted(int(starts[i]) for i in order[:n_segments] if vols[i] > 0)


# ---------------------------------------------------------------------------
# The reconstruction proper
# ---------------------------------------------------------------------------
def reconstruct_segment(extract: AzureExtract, app_hash: str, day: int, start_minute: int,
                        horizon_s: float, spec: ReconstructionSpec, seed: int,
                        ) -> Dict[str, Any]:
    """Reconstruct one replay segment into numpy arrays.

    Pure function of `(extract, app_hash, day, start_minute, horizon_s, spec,
    seed)`.  Returns `{"t_arrival", "service_time_s", "func_code", "diagnostics"}`.
    """
    hm = int(round(horizon_s / 60.0))
    counts = extract.minute_counts(app_hash, day)[start_minute:start_minute + hm]
    rng_arr = np_substream(seed, "arrival", app_hash, day, start_minute,
                           spec.within_minute_placement)
    t = place_arrivals(counts, spec, rng_arr)
    n = int(t.size)

    rows = extract.function_days(app_hash, day)
    if rows.size == 0:
        raise ValueError(f"no duration rows for app {app_hash[:12]} day {day}")
    fcount = extract.dur["count"][rows].astype(np.float64)
    fcode = extract.dur["func_code"][rows]
    usable = extract.dur["minmax_usable"][rows]
    if spec.missing_minmax_policy == "drop_functions" and usable.any():
        rows, fcount, fcode = rows[usable], fcount[usable], fcode[usable]
        usable = usable[usable]

    rng_fn = np_substream(seed, "func_attrib", app_hash, day, start_minute)
    fidx = attribute_functions(n, fcount, spec, rng_fn)

    rng_sv = np_substream(seed, "service", app_hash, day, start_minute)
    u = rng_sv.random(n)
    svc = np.empty(n, dtype=np.float64)
    minmax_used = 0
    for j in range(rows.size):
        m = fidx == j
        if not m.any():
            continue
        r = int(rows[j])
        q, v, used = build_icdf_knots(extract.dur["pct_ms"][r],
                                      _or_none(extract.dur["minimum_ms"][r]),
                                      _or_none(extract.dur["maximum_ms"][r]), spec)
        minmax_used += int(used)
        svc[m] = invert_icdf(u[m], q, v, spec.interpolation)
    svc = np.clip(svc, spec.min_service_time_s, spec.max_service_time_s)

    diagnostics = {
        "n_requests": n,
        "recorded_invocations_in_segment": int(counts.sum()),
        "count_preserved": bool(spec.within_minute_placement != "poisson"),
        "n_functions_in_app_day": int(rows.size),
        "n_functions_with_unusable_minmax": int((~usable).sum()),
        "n_functions_where_true_extremes_used": int(minmax_used),
        "offered_load_request_seconds": float(svc.sum()),
        "mean_service_time_s": float(svc.mean()) if n else 0.0,
    }
    return {"t_arrival": t, "service_time_s": svc,
            "func_code": fcode[fidx] if n else np.zeros(0, dtype=np.int32),
            "diagnostics": diagnostics}


def _or_none(x: float) -> Optional[float]:
    """The extract stores a missing Minimum/Maximum as -1.0 (npz cannot hold None)."""
    return None if float(x) < 0.0 else float(x)


# ---------------------------------------------------------------------------
# The WorkloadSource
# ---------------------------------------------------------------------------
class AzureReplaySource(WorkloadSource):
    """`WorkloadSource` replaying one selected Azure application segment.

    Contract compliance (INTERFACE.md section 4):

      * `build()` is not overridden.
      * `begin_build(episode, seed, rng)` stores the seed and invalidates the
        reconstruction cache; `records(episode)` then reconstructs, so the record
        payloads are a pure function of `(episode, seed)` and the spec.
      * Service times are drawn at **build** time and handed to the engine as
        `service_time_s` on the record, so a given `(episode, seed)` presents
        every controller with the identical realisation -- comparisons are paired.
      * `Request.demand` is in seconds, the unit `request_service` expects.

    One source instance covers `len(segments)` episodes; `episode` indexes the
    segment list.
    """

    def __init__(self, extract: AzureExtract, app_hash: str,
                 segments: Sequence[Tuple[int, int]], *, horizon_s: float = 3600.0,
                 spec: Optional[ReconstructionSpec] = None,
                 name: Optional[str] = None):
        super().__init__()
        self.extract = extract
        self.app_hash = str(app_hash)
        self.app = extract.app(self.app_hash)
        self.segments = [(int(d), int(s)) for d, s in segments]
        self.horizon = float(horizon_s)
        self.spec = spec or ReconstructionSpec()
        self.name = name or f"azure2019/{self.app_hash[:12]}"
        self._seed = 0
        self._cache_key: Optional[Tuple[int, int]] = None
        self._cache: Optional[List[Dict[str, Any]]] = None
        self._last_diagnostics: Dict[str, Any] = {}

    # -- WorkloadSource surface ---------------------------------------
    def n_episodes(self) -> Optional[int]:
        return len(self.segments)

    def horizon_s(self, episode: int) -> Optional[float]:
        return self.horizon

    def begin_build(self, episode: int, seed: int, rng) -> None:
        self._seed = int(seed)
        if self._cache_key != (int(episode), int(seed)):
            self._cache = None

    def arrays(self, episode: int, seed: Optional[int] = None) -> Dict[str, Any]:
        """The reconstructed arrays for one episode, without building `Request`s.

        Used by the corpus checksum and the validation/diagnostics code, which
        need the numeric trace and not the engine's objects.
        """
        s = self._seed if seed is None else int(seed)
        day, start = self.segments[int(episode) % len(self.segments)]
        return reconstruct_segment(self.extract, self.app_hash, day, start,
                                   self.horizon, self.spec, s)

    def records(self, episode: int) -> Sequence[Any]:
        key = (int(episode), int(self._seed))
        if self._cache is not None and self._cache_key == key:
            return self._cache
        out = self.arrays(int(episode))
        self._last_diagnostics = out["diagnostics"]
        day, start = self.segments[int(episode) % len(self.segments)]
        t, svc, fc = out["t_arrival"], out["service_time_s"], out["func_code"]
        cls = self.app.dominant_trigger if self.spec.class_label_from == "dominant_trigger" else ""
        # `reconstruct_segment` already slices the minute vector at the segment
        # start, so `t` is relative to the episode start; no offset to remove.
        recs = [{"seq": i, "t_arrival": float(t[i]), "service_time_s": float(svc[i]),
                 "cls": cls, "attrs": {"app": self.app_hash, "day": day,
                                       "segment_start_minute": start,
                                       "function_code": int(fc[i]),
                                       "reconstructed": True}}
                for i in range(t.size)]
        self._cache, self._cache_key = recs, key
        return recs

    def make_request(self, rec: Any, i: int, rng) -> Request:
        # Service time comes from the record (drawn at build time by the
        # reconstruction), never re-drawn here.
        return Request(seq=int(rec["seq"]), t_arrival=float(rec["t_arrival"]),
                       demand=float(rec["service_time_s"]), size=0.0,
                       cls=str(rec["cls"]), attrs=dict(rec["attrs"]))

    def manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name, "class": type(self).__name__, "family": "azure_replay",
            "calibrated": False,
            "is_measured_data": False,
            "what_is_real": "the per-minute invocation COUNT series of application "
                            f"{self.app_hash[:12]}... on the listed source days",
            "what_is_reconstructed": "individual arrival instants, which function each "
                                     "invocation belongs to, and every service time",
            "dataset": DATASET_NAME, "license": DATASET_LICENSE,
            "required_attribution": ATTRIBUTION,
            "app_hash": self.app_hash, "regime": self.app.regime,
            "volume_tercile": self.app.volume_tercile, "split": self.app.split,
            "dominant_trigger": self.app.dominant_trigger, "app_stats_14d": self.app.stats,
            "segments": [{"day": d, "start_minute": s} for d, s in self.segments],
            "horizon_s": self.horizon,
            "reconstruction": self.spec.provenance(),
            "last_build_diagnostics": dict(self._last_diagnostics),
        }
