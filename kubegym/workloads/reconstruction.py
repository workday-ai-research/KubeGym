"""Request-level reconstruction from the Azure Functions 2019 summary tables.

WHAT THIS MODULE IS
-------------------
The Azure Functions 2019 trace does **not** contain a request-level arrival log
and does **not** contain per-invocation service times.  It contains, per
24-hour period:

  * per-function invocation **counts** for each of the 1440 minutes, and
  * per-function execution-time **summary statistics**: `Average`, `Count`,
    `Minimum`, `Maximum`, and weighted percentiles at q = 0, 1, 25, 50, 75, 99,
    100 **of the 30-second-interval averages**, excluding cold-start time.

An autoscaling simulator needs individual arrival instants and individual
service times.  Producing them therefore requires sampling assumptions that are
**not** in the data.  Everything in this module is such an assumption.

    NOTHING PRODUCED HERE IS MEASURED DATA.  Every field of
    `ReconstructionSpec` is a modelling choice of this benchmark.  The
    provenance block emitted by `ReconstructionSpec.provenance()` carries
    `calibrated: false` and `kind: "modelling_assumption"`, and every corpus
    manifest embeds it.  A result computed over a reconstructed trace is a
    result about this benchmark's model of the Azure workload, not about the
    Azure workload.

THE ASSUMPTIONS, NAMED
----------------------
A1  `within_minute_placement` -- how the `c` invocations recorded in a minute are
    placed inside that minute.  Four arms, all implemented, `uniform` default:

      uniform        c i.i.d. Uniform[0, 60) offsets, sorted.  This is exactly
                     the conditional law of a homogeneous Poisson process on the
                     minute given N = c (order statistics of uniforms), so it is
                     the maximum-entropy placement consistent with the recorded
                     count.  Preserves the per-minute count exactly.
      poisson        a piecewise-constant-rate Poisson process with rate c/60 in
                     that minute.  Does NOT preserve the count: the realised
                     count is Poisson(c).  It preserves the minute's *rate* and
                     adds the count dispersion a genuine Poisson source would
                     have, so it is the arm that tests whether treating the
                     recorded count as exact matters.
      deterministic  evenly spaced at (j + 1/2) * 60/c.  Zero within-minute
                     randomness: the minimum-burstiness reference.
      bursty_batch   batch arrivals.  c is partitioned into geometric batches of
                     mean `batch_mean_size`; each batch epoch is Uniform[0, 60)
                     and every request in a batch arrives at that instant.
                     Preserves the count exactly and inflates sub-minute
                     burstiness by roughly the mean batch size.

    Which arm is right is unknowable from this dataset.  `sensitivity.py` runs
    all four and reports how far the downstream difficulty measures move.

A2  `service_time_source` -- how a per-request service time is drawn.  Default
    `percentile_icdf`: the 7 percentile knots of a function-day are treated as
    an inverse CDF and inverted at a uniform draw, interpolating between knots
    (`interpolation`: `linear` default, or `loglinear`).

    The load-bearing caveat, from the source documentation: those percentiles
    are of the **30-second-interval averages**, not of individual invocations.
    Averaging over an interval removes within-interval variance, so this
    reconstruction **understates** per-request service-time dispersion by an
    amount that grows with the number of invocations per 30 s interval.  Under
    a bounded-concurrency queueing model, less service-time dispersion means
    shorter queues at the same utilisation, so the bias direction is
    **OPTIMISTIC**: capacity planning over the reconstruction is easier than
    over the true workload.  The source documentation states the same limit --
    the percentiles of averages tend to the percentiles of the true
    distribution only when the per-interval sample count is small.

A3  `function_attribution` -- the staged invocation table is aggregated to
    application level (applications are Azure's unit of resource allocation), so
    a per-minute count does not say which of the app's functions fired.
    Default `count_weighted`: each invocation is attributed to a function of the
    app by a categorical draw with probability proportional to that
    function-day's `Count` in the durations table.  The alternative,
    `uniform_functions`, ignores `Count`.

    Two documented facts make A3 an assumption rather than an identity: (i) the
    source states `Count` in the durations table is taken from a different log
    than the invocation counts and "in a few rare cases they may diverge (even
    by a lot)"; (ii) attribution is independent across invocations, so any real
    correlation between which function fires and when is destroyed.

A4  `missing_minmax_policy` -- the source documents that `Minimum` and `Maximum`
    "were not recorded" for a few functions "because of a field naming issue in
    a few versions of the Azure Functions runtime".  In the staged tables the
    unusable cases are: null values, negative values (execution time cannot be
    negative), and rows whose `Minimum` exceeds `percentile_Average_0` or whose
    `Maximum` falls below `percentile_Average_100`.  `audit_duration_table()`
    counts each category.

    Default policy `percentile_bounds`: `Minimum`/`Maximum` are **not used as
    knots at all** -- the inverse CDF spans `percentile_Average_0` to
    `percentile_Average_100`, which are always present, so a missing extreme
    cannot change a single draw and there is nothing to impute.  A per-function
    flag `minmax_usable` is recorded either way so the affected fraction is
    reportable.  Policy `true_extremes` instead stretches the outer segments
    (q in [0,1] and [99,100]) out to `Minimum`/`Maximum` where those are
    usable, widening the tail; it falls back to `percentile_bounds` per function
    where they are not.  `drop_functions` excludes affected functions from
    attribution entirely, and is offered only to bound the effect of the other
    two.

A5  `min_service_time_s` -- durations are recorded in integer milliseconds and
    6085 of 662927 function-days have `Maximum == 0`, i.e. every 30 s average
    rounded to 0 ms.  A served request cannot take zero time, so every draw is
    floored at `min_service_time_s` (default 1 ms, the recording granularity).

A6  cold start is **excluded** from the reconstructed service time, because the
    source states execution time "does not include the cold start time".  Cold
    start therefore enters the simulation exactly once, through the replica
    pool's `replica_cold_start_s`, and is not double counted.  That constant is
    a labelled placeholder in the shipped config, so cold-start cost in any run
    over this corpus is uncalibrated.

A7  `class_label` -- requests carry a coarse class label derived from the
    application's dominant trigger group, for use as an observable covariate.
    It is a property of the application, not of the request, so it carries no
    information about an individual request's service time beyond the app
    identity.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .seeding import (categorical_from_uniform, geometric_from_uniform,
                      poisson_from_uniform)

#: quantile knots of the durations table, in percent
PERCENTILE_KNOTS: Tuple[int, ...] = (0, 1, 25, 50, 75, 99, 100)
PERCENTILE_COLUMNS: Tuple[str, ...] = tuple(f"percentile_Average_{q}" for q in PERCENTILE_KNOTS)

PLACEMENT_ARMS: Tuple[str, ...] = ("uniform", "poisson", "deterministic", "bursty_batch")
MISSING_MINMAX_POLICIES: Tuple[str, ...] = ("percentile_bounds", "true_extremes", "drop_functions")
INTERPOLATIONS: Tuple[str, ...] = ("linear", "loglinear")
ATTRIBUTIONS: Tuple[str, ...] = ("count_weighted", "uniform_functions")


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReconstructionSpec:
    """Every knob of the per-minute-counts -> request-level reconstruction.

    Defaults are the corpus defaults.  All of them are modelling choices; see the
    module docstring for the reasoning behind each and the direction of the bias
    it introduces.
    """

    within_minute_placement: str = "uniform"        # A1
    batch_mean_size: float = 4.0                    # A1, bursty_batch only
    service_time_source: str = "percentile_icdf"    # A2
    interpolation: str = "linear"                   # A2
    function_attribution: str = "count_weighted"    # A3
    missing_minmax_policy: str = "percentile_bounds"  # A4
    min_service_time_s: float = 0.001               # A5
    max_service_time_s: float = 3600.0              # A5, sanity clamp
    class_label_from: str = "dominant_trigger"      # A7

    def __post_init__(self) -> None:
        if self.within_minute_placement not in PLACEMENT_ARMS:
            raise ValueError(f"within_minute_placement must be one of {PLACEMENT_ARMS}")
        if self.missing_minmax_policy not in MISSING_MINMAX_POLICIES:
            raise ValueError(f"missing_minmax_policy must be one of {MISSING_MINMAX_POLICIES}")
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of {INTERPOLATIONS}")
        if self.function_attribution not in ATTRIBUTIONS:
            raise ValueError(f"function_attribution must be one of {ATTRIBUTIONS}")
        if self.service_time_source != "percentile_icdf":
            raise ValueError("service_time_source: only 'percentile_icdf' is implemented")
        if self.batch_mean_size < 1.0:
            raise ValueError("batch_mean_size must be >= 1")
        if not (0.0 < self.min_service_time_s <= self.max_service_time_s):
            raise ValueError("need 0 < min_service_time_s <= max_service_time_s")

    # -- serialisation -------------------------------------------------
    def spec(self) -> Dict[str, Any]:
        return dict(sorted(asdict(self).items()))

    def provenance(self) -> Dict[str, Any]:
        """The machine-readable record embedded in every corpus manifest."""
        return {
            "schema": "kubegym-reconstruction-1",
            "kind": "modelling_assumption",
            "calibrated": False,
            "is_measured_data": False,
            "statement": "The Azure Functions 2019 trace provides per-minute invocation COUNTS "
                         "and duration information only as weighted percentiles of "
                         "30-second-interval averages, excluding cold-start time. Individual "
                         "arrival instants and individual service times are NOT in the data. "
                         "This block records the sampling assumptions used to synthesise them. "
                         "Results over the reconstructed corpus are results about this model of "
                         "the Azure workload, not measurements of the Azure workload.",
            "spec": self.spec(),
            "assumptions": [
                {"id": "A1", "field": "within_minute_placement",
                 "value": self.within_minute_placement,
                 "choice": "Placement of the c invocations recorded in a minute inside that "
                           "minute. 'uniform' is the conditional law of a homogeneous Poisson "
                           "process given the count, i.e. the maximum-entropy placement "
                           "consistent with the recorded count.",
                 "arms": list(PLACEMENT_ARMS),
                 "bias_direction": "unknown sign a priori; measured by the sensitivity study, "
                                   "see sensitivity_report in the corpus manifest",
                 "not_in_the_data": True},
                {"id": "A2", "field": "service_time_source",
                 "value": f"{self.service_time_source}/{self.interpolation}",
                 "choice": "Per-request service time drawn by inverting the 7 percentile knots "
                           "of the request's function-day as a piecewise inverse CDF.",
                 "bias_direction": "OPTIMISTIC. The percentiles are of 30-second-interval "
                                   "AVERAGES, so within-interval variance is absent from the "
                                   "reconstruction. Lower service-time dispersion at equal "
                                   "utilisation means shorter queues, so capacity planning over "
                                   "the reconstruction is easier than over the true workload. "
                                   "Magnitude is unbounded from this dataset alone and grows "
                                   "with invocations per 30 s interval.",
                 "not_in_the_data": True},
                {"id": "A3", "field": "function_attribution",
                 "value": self.function_attribution,
                 "choice": "The staged invocation table is aggregated to application level, so "
                           "each invocation is attributed to one of the app's functions by an "
                           "independent categorical draw proportional to that function-day's "
                           "Count.",
                 "bias_direction": "unknown sign. Destroys any real correlation between which "
                                   "function fires and when; the source also documents that "
                                   "durations-table Count and the invocation counts come from "
                                   "different logs and rarely diverge.",
                 "not_in_the_data": True},
                {"id": "A4", "field": "missing_minmax_policy",
                 "value": self.missing_minmax_policy,
                 "choice": "The source documents that Minimum/Maximum were not recorded for a "
                           "few functions owing to a field-naming issue in some runtime "
                           "versions. The default policy uses only the percentile knots, which "
                           "are always present, so a missing extreme changes no draw and "
                           "nothing is imputed. Affected functions are counted and flagged.",
                 "bias_direction": "default 'percentile_bounds' truncates the service-time "
                                   "support at the extreme 30 s averages rather than the true "
                                   "extremes: OPTIMISTIC on the tail. 'true_extremes' widens it.",
                 "not_in_the_data": True},
                {"id": "A5", "field": "min_service_time_s",
                 "value": self.min_service_time_s,
                 "choice": "Durations are integer milliseconds and 6085 of 662927 function-days "
                           "report Maximum == 0. A served request cannot take zero time, so "
                           "draws are floored at the recording granularity.",
                 "bias_direction": "pessimistic on those functions (a genuinely sub-millisecond "
                                   "function is charged 1 ms), negligible in magnitude",
                 "not_in_the_data": True},
                {"id": "A6", "field": "cold_start",
                 "value": "excluded from service time; supplied by replica_cold_start_s",
                 "choice": "The source states execution time excludes cold-start time, so cold "
                           "start enters the simulation once, via the replica pool, and is not "
                           "double counted.",
                 "bias_direction": "n/a for the reconstruction; replica_cold_start_s is itself a "
                                   "labelled placeholder in the shipped config, so cold-start "
                                   "cost over this corpus is uncalibrated",
                 "not_in_the_data": False},
                {"id": "A7", "field": "class_label_from",
                 "value": self.class_label_from,
                 "choice": "Requests carry the application's dominant trigger group as a coarse "
                           "class label, an observable app-level covariate.",
                 "bias_direction": "none; carries no per-request work information",
                 "not_in_the_data": False},
            ],
        }


# ---------------------------------------------------------------------------
# A4: auditing the durations table
# ---------------------------------------------------------------------------
def audit_duration_table(df) -> Dict[str, Any]:
    """Count the unusable-`Minimum`/`Maximum` cases in a durations frame.

    `df` is the staged `azure2019_function_durations` frame (or any subset).
    Returns per-category counts; no rows are modified.
    """
    p0 = df[PERCENTILE_COLUMNS[0]].to_numpy(dtype=np.float64)
    p100 = df[PERCENTILE_COLUMNS[-1]].to_numpy(dtype=np.float64)
    mn = df["Minimum"].to_numpy(dtype=np.float64)
    mx = df["Maximum"].to_numpy(dtype=np.float64)
    null = np.isnan(mn) | np.isnan(mx)
    with np.errstate(invalid="ignore"):
        neg = (mn < 0) | (mx < 0)
        above = mn > p0
        below = mx < p100
    unusable = null | neg | above | below
    pct = df[list(PERCENTILE_COLUMNS)].to_numpy(dtype=np.float64)
    return {
        "n_function_days": int(len(df)),
        "n_functions": int(df["HashFunction"].nunique()),
        "n_apps": int(df["HashApp"].nunique()),
        "minmax_null": int(null.sum()),
        "minmax_negative": int(neg.sum()),
        "minimum_above_percentile_0": int(above.sum()),
        "maximum_below_percentile_100": int(below.sum()),
        "minmax_unusable_total": int(unusable.sum()),
        "minmax_unusable_fraction": float(unusable.mean()) if len(df) else 0.0,
        "functions_affected": int(df.loc[unusable, "HashFunction"].nunique()),
        "apps_affected": int(df.loc[unusable, "HashApp"].nunique()),
        "percentiles_nonmonotone_rows": int((np.diff(pct, axis=1) < 0).any(axis=1).sum()),
        "percentiles_negative_rows": int((pct < 0).any(axis=1).sum()),
        "maximum_equals_zero_rows": int((mx == 0).sum()),
        "note": "Categories are not disjoint. 'unusable' drives the minmax_usable flag; under "
                "the default missing_minmax_policy='percentile_bounds' it changes no draw, "
                "because Minimum/Maximum are not used as knots.",
    }


def minmax_usable(minimum: float, maximum: float, p0: float, p100: float) -> bool:
    """Whether a function-day's `Minimum`/`Maximum` pair is usable as a support bound."""
    if minimum is None or maximum is None:
        return False
    if not (math.isfinite(minimum) and math.isfinite(maximum)):
        return False
    if minimum < 0.0 or maximum < 0.0:
        return False
    if minimum > p0 or maximum < p100:
        return False
    return True


# ---------------------------------------------------------------------------
# A2 / A4: the service-time inverse CDF
# ---------------------------------------------------------------------------
def build_icdf_knots(percentiles_ms: Sequence[float], minimum_ms: Optional[float],
                     maximum_ms: Optional[float], spec: ReconstructionSpec,
                     ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Knots of one function-day's service-time inverse CDF, in seconds.

    Returns `(q, v_s, minmax_used)` with `q` in [0, 1] non-decreasing and `v_s`
    non-decreasing seconds.  `q`/`v_s` are consumed by `invert_icdf`.
    """
    q = np.asarray(PERCENTILE_KNOTS, dtype=np.float64) / 100.0
    v = np.asarray(percentiles_ms, dtype=np.float64) / 1000.0
    if v.shape != q.shape:
        raise ValueError(f"need {q.size} percentile knots, got {v.shape}")
    # Percentile columns are non-negative in the staged tables except for a
    # handful of rows on one application (7 of 662927); clamp defensively so a
    # negative knot can never produce a negative service time.
    v = np.maximum(v, 0.0)
    v = np.maximum.accumulate(v)                      # enforce monotone inverse CDF
    used = False
    if spec.missing_minmax_policy == "true_extremes":
        p0_ms = float(percentiles_ms[0])
        p100_ms = float(percentiles_ms[-1])
        if minmax_usable(minimum_ms, maximum_ms, p0_ms, p100_ms):
            v = v.copy()
            v[0] = max(0.0, float(minimum_ms) / 1000.0)
            v[-1] = max(v[-2], float(maximum_ms) / 1000.0)
            v = np.maximum.accumulate(v)
            used = True
    v = np.clip(v, spec.min_service_time_s, spec.max_service_time_s)
    return q, v, used


def invert_icdf(u: np.ndarray, q: np.ndarray, v: np.ndarray,
                interpolation: str = "linear") -> np.ndarray:
    """Invert a piecewise inverse CDF at uniform draws `u`."""
    u = np.asarray(u, dtype=np.float64)
    if interpolation == "linear":
        return np.interp(u, q, v)
    if interpolation == "loglinear":
        lv = np.log(np.maximum(v, 1e-12))
        return np.exp(np.interp(u, q, lv))
    raise ValueError(f"unknown interpolation {interpolation!r}")


# ---------------------------------------------------------------------------
# A1: within-minute arrival placement
# ---------------------------------------------------------------------------
def place_arrivals(counts: np.ndarray, spec: ReconstructionSpec,
                   rng: np.random.Generator, *, t0_s: float = 0.0,
                   minute_s: float = 60.0) -> np.ndarray:
    """Arrival times, in seconds from `t0_s`, for a per-minute count vector.

    `counts[m]` is the number of invocations recorded in minute `m`.  The return
    value is sorted ascending.  Only `rng.random(n)` is drawn, so the output is
    reproducible across numpy versions (see `seeding`).
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1:
        raise ValueError("counts must be 1-D")
    arm = spec.within_minute_placement

    if arm == "poisson":
        # Re-draw the count from Poisson(c) at the minute's rate: the one arm
        # that does not preserve the recorded count.
        u = rng.random(counts.size)
        counts = poisson_from_uniform(u, counts.astype(np.float64))

    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.float64)
    minute_index = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    base = t0_s + minute_index.astype(np.float64) * minute_s

    if arm == "deterministic":
        # (j + 1/2) * minute/c, with j the within-minute rank.
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        rank = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
        c_rep = np.repeat(counts, counts).astype(np.float64)
        off = (rank.astype(np.float64) + 0.5) * minute_s / c_rep
        return np.sort(base + off)

    if arm in ("uniform", "poisson"):
        off = rng.random(total) * minute_s
        return np.sort(base + off)

    if arm == "bursty_batch":
        # Partition each minute's count into geometric batches; one epoch per
        # batch, every request of a batch at that epoch.
        out = np.empty(total, dtype=np.float64)
        pos = 0
        # Draw batch sizes in chunks so the number of rng calls stays O(1) per
        # minute-block rather than O(batches).
        for m in np.nonzero(counts)[0]:
            c = int(counts[m])
            sizes: List[int] = []
            remaining = c
            while remaining > 0:
                need = max(4, int(remaining / spec.batch_mean_size) + 2)
                draw = geometric_from_uniform(rng.random(need), spec.batch_mean_size)
                for s in draw:
                    s = int(min(int(s), remaining))
                    sizes.append(s)
                    remaining -= s
                    if remaining <= 0:
                        break
            epochs = np.sort(rng.random(len(sizes))) * minute_s
            expanded = np.repeat(epochs, np.asarray(sizes, dtype=np.int64))
            out[pos:pos + c] = t0_s + m * minute_s + expanded
            pos += c
        return np.sort(out[:pos])

    raise ValueError(f"unknown placement arm {arm!r}")


# ---------------------------------------------------------------------------
# A3: function attribution
# ---------------------------------------------------------------------------
def attribute_functions(n: int, counts_by_function: Sequence[float],
                        spec: ReconstructionSpec, rng: np.random.Generator) -> np.ndarray:
    """Function index per invocation, as an array of length `n`."""
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    w = np.asarray(counts_by_function, dtype=np.float64)
    if w.size == 0:
        raise ValueError("no functions to attribute to")
    if spec.function_attribution == "uniform_functions":
        w = np.ones_like(w)
    else:
        w = np.maximum(w, 0.0)
        if w.sum() <= 0:
            w = np.ones_like(w)
    return categorical_from_uniform(rng.random(n), w)


__all__ = [
    "ReconstructionSpec", "PERCENTILE_KNOTS", "PERCENTILE_COLUMNS", "PLACEMENT_ARMS",
    "MISSING_MINMAX_POLICIES", "INTERPOLATIONS", "ATTRIBUTIONS",
    "audit_duration_table", "minmax_usable", "build_icdf_knots", "invert_icdf",
    "place_arrivals", "attribute_functions",
]
