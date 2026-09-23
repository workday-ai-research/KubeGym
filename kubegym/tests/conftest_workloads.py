"""Shared fixtures for the workload-corpus tests.

The tests must run without the 2 GB Azure source and without the staged parquet
tables, so `tiny_extract()` builds a synthetic `AzureExtract` in memory with the
same schema: 6 applications (one per regime plus a spare), all 14 days, 3
functions each, and duration knots chosen so the tests can assert exact
properties (a degenerate function with all knots equal, a function with missing
extremes, a heavy-tailed one).

Tests that want the *real* extract read it from `KUBEGYM_AZURE_EXTRACT`, a
directory holding `azure2019_selection.npz` and `azure2019_selection.json`, and
skip when it is unset -- the same pattern the core package uses for its
testbed-gated tests.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pytest

from kubegym.workloads.azure2019 import (MINUTES_PER_DAY, N_DAYS, AppRecord, AzureExtract)
from kubegym.workloads.reconstruction import PERCENTILE_KNOTS

#: (regime, volume_tercile, split, dominant_trigger, mean rate per minute)
TINY_APPS: Tuple[Tuple[str, str, str, str, float], ...] = (
    ("steady", "high", "train", "http", 20.0),
    ("bursty", "mid", "train", "queue", 6.0),
    ("diurnal", "high", "dev", "timer", 12.0),
    ("intermittent", "low", "dev", "http", 0.4),
    ("sparse", "low", "test", "event", 0.02),
    ("steady", "mid", "test", "http", 3.0),
)


def _counts_for(regime: str, mean_rate: float, app_i: int) -> np.ndarray:
    """Deterministic per-minute counts for one synthetic application, 14 x 1440."""
    rng = np.random.default_rng(1000 + app_i)
    t = np.arange(MINUTES_PER_DAY)
    out = np.zeros((N_DAYS, MINUTES_PER_DAY), dtype=np.int32)
    for d in range(N_DAYS):
        if regime == "diurnal":
            lam = mean_rate * (1.0 + 0.8 * np.sin(2 * np.pi * (t / MINUTES_PER_DAY)))
        elif regime == "bursty":
            lam = np.full(MINUTES_PER_DAY, mean_rate * 0.4)
            for start in (200 + 97 * d) % 1200, (700 + 53 * d) % 1200:
                lam[start:start + 20] = mean_rate * 12.0
        else:
            lam = np.full(MINUTES_PER_DAY, float(mean_rate))
        c = rng.poisson(lam)
        if regime in ("sparse", "intermittent"):
            keep = rng.random(MINUTES_PER_DAY) < (0.03 if regime == "sparse" else 0.35)
            c = np.where(keep, c, 0)
        out[d] = c
    return out


def _dur_rows(n_apps: int) -> dict:
    """Duration knots for 3 functions x 14 days x n_apps, with deliberate edge cases.

    function 0: log-normal-ish spread, extremes present and consistent
    function 1: degenerate -- every percentile identical (a real pattern in the
                source, where a function's 30 s averages never vary)
    function 2: missing minimum and maximum (the dataset's field-naming defect)
    """
    nq = len(PERCENTILE_KNOTS)
    app_idx, day, code, count = [], [], [], []
    pct, mn, mx, usable = [], [], [], []
    for a in range(n_apps):
        for d in range(1, N_DAYS + 1):
            for f in range(3):
                app_idx.append(a); day.append(d); code.append(f)
                if f == 0:
                    v = np.array([8.0, 12.0, 40.0, 90.0, 260.0, 3000.0, 9000.0])
                    count.append(1000); mn.append(6.0); mx.append(11000.0); usable.append(True)
                elif f == 1:
                    v = np.full(nq, 55.0)
                    count.append(300); mn.append(55.0); mx.append(55.0); usable.append(True)
                else:
                    v = np.array([20.0, 22.0, 30.0, 45.0, 70.0, 400.0, 800.0])
                    count.append(50); mn.append(-1.0); mx.append(-1.0); usable.append(False)
                pct.append(v)
    return {
        "app_idx": np.asarray(app_idx, dtype=np.int32),
        "day": np.asarray(day, dtype=np.int16),
        "func_code": np.asarray(code, dtype=np.int32),
        "count": np.asarray(count, dtype=np.int64),
        "pct_ms": np.asarray(pct, dtype=np.float64),
        "minimum_ms": np.asarray(mn, dtype=np.float64),
        "maximum_ms": np.asarray(mx, dtype=np.float64),
        "minmax_usable": np.asarray(usable, dtype=bool),
    }


def tiny_extract() -> AzureExtract:
    apps, counts = [], []
    for i, (regime, tercile, split, trigger, rate) in enumerate(TINY_APPS):
        c = _counts_for(regime, rate, i)
        counts.append(c)
        nz = int((c > 0).sum())
        apps.append(AppRecord(
            app_hash=f"tiny{i:02d}" + "0" * 58, regime=regime, volume_tercile=tercile,
            split=split, dominant_trigger=trigger,
            stats={"total_invocations": float(c.sum()),
                   "mean_rate_per_minute": float(c.mean()),
                   "peak_per_minute": float(c.max()),
                   "zero_minute_fraction": float(1.0 - nz / c.size),
                   "index_of_dispersion": float(c.var() / max(c.mean(), 1e-9)),
                   "relative_24h_amplitude": 0.0}))
    meta = {"synthetic_fixture": True, "dataset": "synthetic (test fixture, not Azure data)",
            "population_comparison": {"regime_share": {"population": {}, "selected": {}}},
            "duration_audit": {"note": "synthetic"}}
    return AzureExtract(np.stack(counts), apps, meta, _dur_rows(len(TINY_APPS)))


@pytest.fixture(scope="module")
def extract() -> AzureExtract:
    return tiny_extract()


@pytest.fixture(scope="module")
def real_extract() -> AzureExtract:
    d = os.environ.get("KUBEGYM_AZURE_EXTRACT")
    if not d:
        pytest.skip("set KUBEGYM_AZURE_EXTRACT to a directory holding "
                    "azure2019_selection.{npz,json} to run this test")
    return AzureExtract.load(os.path.join(d, "azure2019_selection.npz"),
                             os.path.join(d, "azure2019_selection.json"))
