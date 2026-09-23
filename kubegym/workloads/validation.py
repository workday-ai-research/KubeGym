"""Distributional validation: generated families vs the reconstructed Azure trace.

WHAT IS BEING COMPARED, AND WHAT IT CAN AND CANNOT SHOW
-------------------------------------------------------
The comparison is between the four synthetic families and the **reconstructed**
Azure replay traces, on the four quantities that decide how hard an autoscaling
problem is:

  1. `interarrival`   the inter-arrival-time distribution (shape of the arrival
                      process), compared both in raw seconds and after dividing
                      each trace's gaps by their own mean, so that a family which
                      matches the *shape* but not the *rate* is distinguishable
                      from one that matches neither.
  2. `rate_per_bin`   the distribution of per-bin arrival counts at 15 s and 60 s
                      (the control interval and the source's recording
                      granularity).
  3. `dispersion`     the index of dispersion (variance / mean of per-bin counts)
                      at 1 s, 15 s and 60 s bins -- one scalar per trace, so the
                      comparison is between the *distributions over traces*.
  4. `autocorrelation` the autocorrelation of the per-minute count series at lags
                      1..10 plus the integrated ACF, again one vector per trace.

Two-sample Kolmogorov-Smirnov statistics and earth-mover (1-Wasserstein)
distances are reported for (1) and (2); (3) and (4) are scalar/vector summaries
compared by their distribution over traces and by Wasserstein distance between
those distributions.  This is the validation battery an independently published
position paper in this area (BatchBench, arXiv:2605.12272) proposes.

    THE 'REAL' SIDE OF EVERY COMPARISON IS ITSELF RECONSTRUCTED.  The Azure
    arrival instants are synthesised from per-minute counts under assumption A1
    and the service times under A2 (see `reconstruction.py`).  Below the
    one-minute timescale the 'real' inter-arrival distribution IS assumption A1,
    so agreement or disagreement there is a statement about two models, not a
    validation against ground truth.  At and above one minute the count series is
    the recorded data and the comparison is meaningful.  Every table produced
    here carries `timescale_caveat` saying which side of that line it falls on.

    A KS p-VALUE IS NOT THE POINT.  Pooled samples run to 10^5-10^6 gaps, where
    any difference whatever is significant; the p-values are reported for
    completeness but the KS statistic D (an effect size, the maximum CDF gap) and
    the Wasserstein distance are the numbers to read.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

BIN_WIDTHS_S: Tuple[float, ...] = (1.0, 15.0, 60.0)
ACF_LAGS: Tuple[int, ...] = tuple(range(1, 11))


# ---------------------------------------------------------------------------
# Per-trace summaries
# ---------------------------------------------------------------------------
def interarrivals(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return np.diff(np.sort(t)) if t.size >= 2 else np.zeros(0)


def bin_counts(t: np.ndarray, horizon_s: float, bin_s: float) -> np.ndarray:
    n = max(1, int(math.ceil(horizon_s / bin_s)))
    edges = np.arange(n + 1, dtype=np.float64) * bin_s
    return np.histogram(np.asarray(t, dtype=np.float64), bins=edges)[0].astype(np.float64)


def index_of_dispersion(counts: np.ndarray) -> Optional[float]:
    """Variance / mean of a count series.  1.0 for a Poisson process."""
    c = np.asarray(counts, dtype=np.float64)
    m = c.mean() if c.size else 0.0
    return float(c.var() / m) if m > 0 else None


def acf(counts: np.ndarray, lags: Sequence[int] = ACF_LAGS) -> Dict[str, Optional[float]]:
    """Sample autocorrelation of a count series at the given lags."""
    c = np.asarray(counts, dtype=np.float64)
    out: Dict[str, Optional[float]] = {}
    if c.size < max(lags) + 2 or c.std() == 0:
        return {f"lag{k}": None for k in lags} | {"integrated": None}
    x = c - c.mean()
    denom = float((x * x).sum())
    vals = []
    for k in lags:
        v = float((x[:-k] * x[k:]).sum() / denom)
        out[f"lag{k}"] = v
        vals.append(v)
    out["integrated"] = float(np.sum(vals))
    return out


def trace_summary(t: np.ndarray, service_s: np.ndarray, horizon_s: float) -> Dict[str, Any]:
    """Every per-trace quantity the validation and diagnostics need."""
    t = np.asarray(t, dtype=np.float64)
    ia = interarrivals(t)
    out: Dict[str, Any] = {
        "n_requests": int(t.size),
        "horizon_s": float(horizon_s),
        "mean_rate_rps": float(t.size / horizon_s) if horizon_s > 0 else 0.0,
        "mean_interarrival_s": float(ia.mean()) if ia.size else None,
        "cv_interarrival": float(ia.std() / ia.mean()) if ia.size and ia.mean() > 0 else None,
        "mean_service_s": float(np.mean(service_s)) if len(service_s) else None,
        "p99_service_s": float(np.quantile(service_s, 0.99)) if len(service_s) else None,
        "offered_load_request_seconds": float(np.sum(service_s)) if len(service_s) else 0.0,
    }
    for b in BIN_WIDTHS_S:
        c = bin_counts(t, horizon_s, b)
        out[f"iod_{int(b)}s"] = index_of_dispersion(c)
        out[f"peak_to_mean_{int(b)}s"] = (float(c.max() / c.mean()) if c.mean() > 0 else None)
        out[f"zero_bin_fraction_{int(b)}s"] = float((c == 0).mean())
    out["acf_60s"] = acf(bin_counts(t, horizon_s, 60.0))
    return out


# ---------------------------------------------------------------------------
# Two-sample comparisons
# ---------------------------------------------------------------------------
def two_sample(a: np.ndarray, b: np.ndarray, *, max_n: int = 200_000,
               subsample_seed: int = 12345) -> Dict[str, Any]:
    """KS statistic, KS p-value and 1-Wasserstein distance between two samples.

    Samples longer than `max_n` are thinned by a deterministic stride (not a
    random draw, so the number is reproducible without carrying an RNG state);
    the stride is reported.
    """
    from scipy.stats import ks_2samp, wasserstein_distance
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {"n_a": int(a.size), "n_b": int(b.size), "ks_d": None, "ks_p": None,
                "wasserstein": None, "note": "insufficient sample"}
    sa = max(1, int(math.ceil(a.size / max_n)))
    sb = max(1, int(math.ceil(b.size / max_n)))
    aa, bb = a[::sa], b[::sb]
    ks = ks_2samp(aa, bb)
    return {
        "n_a": int(a.size), "n_b": int(b.size),
        "n_a_used": int(aa.size), "n_b_used": int(bb.size),
        "stride_a": sa, "stride_b": sb,
        "ks_d": float(ks.statistic), "ks_p": float(ks.pvalue),
        "wasserstein": float(wasserstein_distance(aa, bb)),
        "mean_a": float(aa.mean()), "mean_b": float(bb.mean()),
        "median_a": float(np.median(aa)), "median_b": float(np.median(bb)),
    }


def _pool(traces: Iterable[Dict[str, Any]], key: str) -> np.ndarray:
    parts = [np.asarray(tr[key], dtype=np.float64) for tr in traces]
    parts = [p for p in parts if p.size]
    return np.concatenate(parts) if parts else np.zeros(0)


def compare_family(real: Sequence[Dict[str, Any]], gen: Sequence[Dict[str, Any]],
                   ) -> Dict[str, Any]:
    """Full validation battery for one generated family against the Azure side.

    Each element of `real` / `gen` is `{"t_arrival", "service_time_s",
    "horizon_s"}` -- the arrays a source's `arrays()` returns.
    """
    def gaps(traces, normalise):
        parts = []
        for tr in traces:
            ia = interarrivals(tr["t_arrival"])
            if ia.size:
                parts.append(ia / ia.mean() if normalise and ia.mean() > 0 else ia)
        return np.concatenate(parts) if parts else np.zeros(0)

    def binned(traces, b):
        parts = [bin_counts(tr["t_arrival"], tr["horizon_s"], b) for tr in traces]
        return np.concatenate(parts) if parts else np.zeros(0)

    def scalars(traces, fn):
        v = [fn(tr) for tr in traces]
        return np.asarray([x for x in v if x is not None], dtype=np.float64)

    out: Dict[str, Any] = {
        "n_real_traces": len(real), "n_gen_traces": len(gen),
        "interarrival_raw_s": two_sample(gaps(real, False), gaps(gen, False)),
        "interarrival_normalised": two_sample(gaps(real, True), gaps(gen, True)),
    }
    out["interarrival_raw_s"]["timescale_caveat"] = (
        "SUB-MINUTE: the real side here is arrival-placement assumption A1, not recorded data. "
        "Read as a comparison between two models.")
    out["interarrival_normalised"]["timescale_caveat"] = out["interarrival_raw_s"][
        "timescale_caveat"]

    for b in (15.0, 60.0):
        r = two_sample(binned(real, b), binned(gen, b))
        r["timescale_caveat"] = (
            "AT the source's recording granularity: the real side is the recorded per-minute "
            "count series (aggregated to this bin), so this comparison is against data."
            if b >= 60.0 else
            "SUB-MINUTE aggregation of the recorded counts under assumption A1; partially "
            "model-dependent.")
        out[f"rate_per_{int(b)}s_bin"] = r

    for b in BIN_WIDTHS_S:
        key = f"iod_{int(b)}s"
        rv = scalars(real, lambda tr, b=b: index_of_dispersion(
            bin_counts(tr["t_arrival"], tr["horizon_s"], b)))
        gv = scalars(gen, lambda tr, b=b: index_of_dispersion(
            bin_counts(tr["t_arrival"], tr["horizon_s"], b)))
        out[key] = {
            "real_median": float(np.median(rv)) if rv.size else None,
            "gen_median": float(np.median(gv)) if gv.size else None,
            "real_p10_p90": [float(np.quantile(rv, .1)), float(np.quantile(rv, .9))]
            if rv.size else None,
            "gen_p10_p90": [float(np.quantile(gv, .1)), float(np.quantile(gv, .9))]
            if gv.size else None,
            "log10_median_ratio": (float(np.log10(np.median(gv) / np.median(rv)))
                                   if rv.size and gv.size and np.median(rv) > 0
                                   and np.median(gv) > 0 else None),
            "over_traces": two_sample(rv, gv, max_n=10_000),
        }

    for lag in (1, 5):
        rv = scalars(real, lambda tr, l=lag: acf(
            bin_counts(tr["t_arrival"], tr["horizon_s"], 60.0)).get(f"lag{l}"))
        gv = scalars(gen, lambda tr, l=lag: acf(
            bin_counts(tr["t_arrival"], tr["horizon_s"], 60.0)).get(f"lag{l}"))
        out[f"acf60_lag{lag}"] = {
            "real_median": float(np.median(rv)) if rv.size else None,
            "gen_median": float(np.median(gv)) if gv.size else None,
            "difference_gen_minus_real": (float(np.median(gv) - np.median(rv))
                                          if rv.size and gv.size else None),
            "over_traces": two_sample(rv, gv, max_n=10_000),
        }
    return out


def worst_axes(comparison: Dict[str, Any], k: int = 3) -> List[Tuple[str, float]]:
    """The axes on which a family agrees least, by KS D (for reporting mismatches)."""
    scored: List[Tuple[str, float]] = []
    for name, v in comparison.items():
        if isinstance(v, dict):
            d = v.get("ks_d")
            if d is None and isinstance(v.get("over_traces"), dict):
                d = v["over_traces"].get("ks_d")
            if d is not None:
                scored.append((name, float(d)))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]
