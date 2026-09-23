"""Tests for the Azure reconstruction layer: assumptions A1-A7.

Every test here pins a *stated assumption*, so a change in behaviour that is not
also a documentation change fails the suite.
"""
from __future__ import annotations

import numpy as np
import pytest

from kubegym.tests.conftest_workloads import extract, real_extract, tiny_extract  # noqa: F401
from kubegym.workloads.azure2019 import AzureReplaySource, reconstruct_segment, segment_starts
from kubegym.workloads.reconstruction import (MISSING_MINMAX_POLICIES, PERCENTILE_KNOTS,
                                              PLACEMENT_ARMS, ReconstructionSpec,
                                              build_icdf_knots, invert_icdf, minmax_usable,
                                              place_arrivals)
from kubegym.workloads.seeding import (categorical_from_uniform, geometric_from_uniform,
                                       normal_from_uniform, np_substream,
                                       poisson_from_uniform)

APP = "tiny00" + "0" * 58
SPARSE = "tiny04" + "0" * 58


# ---------------------------------------------------------------------------
# A1: arrival placement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm", PLACEMENT_ARMS)
def test_placement_stays_inside_its_minute_and_is_sorted(arm):
    counts = np.array([0, 3, 0, 17, 1, 250], dtype=np.int64)
    spec = ReconstructionSpec(within_minute_placement=arm)
    t = place_arrivals(counts, spec, np_substream(7, "arrival"), t0_s=0.0)
    assert np.all(np.diff(t) >= 0), "arrival times must be non-decreasing"
    assert t.min() >= 0.0 and t.max() < counts.size * 60.0
    # every arrival lands in a minute that had a recorded count
    minute = np.floor(t / 60.0).astype(int)
    assert set(np.unique(minute)).issubset(set(np.nonzero(counts)[0].tolist()))


@pytest.mark.parametrize("arm", ["uniform", "deterministic", "bursty_batch"])
def test_count_preserving_arms_preserve_the_recorded_count_exactly(arm):
    """The documented distinction between the arms: only `poisson` re-draws."""
    counts = np.array([0, 3, 0, 17, 1, 250, 44], dtype=np.int64)
    spec = ReconstructionSpec(within_minute_placement=arm)
    t = place_arrivals(counts, spec, np_substream(11, "arrival"), t0_s=0.0)
    assert t.size == int(counts.sum())
    per_minute = np.bincount(np.floor(t / 60.0).astype(int), minlength=counts.size)
    assert np.array_equal(per_minute, counts)


def test_poisson_arm_redraws_the_count_but_preserves_the_rate():
    counts = np.full(600, 20, dtype=np.int64)
    spec = ReconstructionSpec(within_minute_placement="poisson")
    t = place_arrivals(counts, spec, np_substream(3, "arrival"), t0_s=0.0)
    assert t.size != int(counts.sum()), "the poisson arm must not preserve the count"
    # mean of Poisson(20) over 600 minutes: 12000 +- ~5 sigma = 12000 +- 550
    assert abs(t.size - 12000) < 600
    per_minute = np.bincount(np.floor(t / 60.0).astype(int), minlength=counts.size)
    assert per_minute.var() > 5.0, "expected Poisson count dispersion"


def test_deterministic_arm_is_evenly_spaced():
    counts = np.array([4], dtype=np.int64)
    t = place_arrivals(counts, ReconstructionSpec(within_minute_placement="deterministic"),
                       np_substream(1, "arrival"), t0_s=0.0)
    assert np.allclose(t, [7.5, 22.5, 37.5, 52.5])


def test_bursty_batch_inflates_sub_minute_burstiness():
    counts = np.full(200, 30, dtype=np.int64)
    def iod(arm):
        t = place_arrivals(counts, ReconstructionSpec(within_minute_placement=arm),
                           np_substream(5, "arrival"), t0_s=0.0)
        c = np.bincount(np.floor(t / 1.0).astype(int), minlength=200 * 60)
        return c.var() / c.mean()
    assert iod("bursty_batch") > 2.0 * iod("uniform")


# ---------------------------------------------------------------------------
# A2 / A4: service times from the percentile summary, and missing extremes
# ---------------------------------------------------------------------------
def test_service_draws_stay_within_the_knot_support():
    """Knots come back in SECONDS; the draw support is exactly [v[0], v[-1]]."""
    q, v, used = build_icdf_knots(
        np.array([8.0, 12.0, 40.0, 90.0, 260.0, 3000.0, 9000.0]), 6.0, 11000.0,
        ReconstructionSpec(missing_minmax_policy="true_extremes"))
    assert used is True
    assert v[0] == pytest.approx(0.006) and v[-1] == pytest.approx(11.0)
    x = invert_icdf(np_substream(2, "service").random(20_000), q, v)
    assert x.min() >= v[0] - 1e-9 and x.max() <= v[-1] + 1e-9
    # the median draw must sit near the p50 knot (90 ms)
    assert 0.045 <= np.median(x) <= 0.180


def test_percentile_bounds_policy_ignores_the_extremes_entirely():
    """The default policy: a missing min/max cannot change any draw."""
    pct = np.array([8.0, 12.0, 40.0, 90.0, 260.0, 3000.0, 9000.0])
    spec = ReconstructionSpec(missing_minmax_policy="percentile_bounds")
    q1, v1, u1 = build_icdf_knots(pct, 6.0, 11000.0, spec)
    q2, v2, u2 = build_icdf_knots(pct, None, None, spec)
    assert np.array_equal(q1, q2) and np.array_equal(v1, v2)
    assert u1 is False and u2 is False
    assert v1[0] == pytest.approx(pct[0] / 1000.0)
    assert v1[-1] == pytest.approx(pct[-1] / 1000.0)


def test_true_extremes_policy_falls_back_when_they_are_missing():
    pct = np.array([20.0, 22.0, 30.0, 45.0, 70.0, 400.0, 800.0])
    spec = ReconstructionSpec(missing_minmax_policy="true_extremes")
    q, v, used = build_icdf_knots(pct, None, None, spec)
    assert used is False
    assert v[0] == pytest.approx(pct[0] / 1000.0)
    assert v[-1] == pytest.approx(pct[-1] / 1000.0)


def test_minmax_usable_rejects_negative_and_inconsistent_extremes():
    p0, p100 = 10.0, 200.0
    assert minmax_usable(5.0, 300.0, p0, p100) is True
    assert minmax_usable(-1.0, 300.0, p0, p100) is False    # negative (missing sentinel)
    assert minmax_usable(float("nan"), 300.0, p0, p100) is False
    assert minmax_usable(5.0, 150.0, p0, p100) is False     # Maximum below p100
    assert minmax_usable(20.0, 300.0, p0, p100) is False    # Minimum above p0


def test_drop_functions_policy_is_offered_and_documented():
    assert set(MISSING_MINMAX_POLICIES) == {"percentile_bounds", "true_extremes",
                                            "drop_functions"}


def test_degenerate_function_yields_a_constant_service_time(extract):
    """A function whose percentiles are all equal must draw that value exactly."""
    rows = extract.function_days(APP, 1)
    r = rows[extract.dur["func_code"][rows] == 1][0]
    q, v, _ = build_icdf_knots(extract.dur["pct_ms"][r], None, None, ReconstructionSpec())
    x = invert_icdf(np_substream(9, "service").random(500), q, v)
    assert np.allclose(x, 0.055)


def test_min_service_time_floor_is_applied(extract):
    spec = ReconstructionSpec(min_service_time_s=0.5)
    out = reconstruct_segment(extract, APP, 1, 0, 600.0, spec, seed=17)
    assert out["service_time_s"].min() >= 0.5 - 1e-12


# ---------------------------------------------------------------------------
# A3: function attribution
# ---------------------------------------------------------------------------
def test_function_attribution_follows_the_count_weights(extract):
    """`count_weighted` must reproduce the per-function Count shares."""
    out = reconstruct_segment(extract, APP, 1, 0, 3600.0, ReconstructionSpec(), seed=23)
    codes = out["func_code"]
    assert codes.size > 500
    share = np.bincount(codes, minlength=3) / codes.size
    expected = np.array([1000.0, 300.0, 50.0]); expected /= expected.sum()
    assert np.allclose(share, expected, atol=0.05), (share, expected)


def test_uniform_attribution_differs_from_count_weighted(extract):
    a = reconstruct_segment(extract, APP, 1, 0, 3600.0,
                            ReconstructionSpec(function_attribution="count_weighted"), seed=23)
    b = reconstruct_segment(extract, APP, 1, 0, 3600.0,
                            ReconstructionSpec(function_attribution="uniform_functions"),
                            seed=23)
    assert not np.array_equal(a["func_code"], b["func_code"])
    assert abs(np.bincount(b["func_code"], minlength=3).min() / b["func_code"].size
               - 1 / 3) < 0.05


# ---------------------------------------------------------------------------
# Provenance honesty
# ---------------------------------------------------------------------------
def test_reconstruction_provenance_is_flagged_as_an_assumption():
    p = ReconstructionSpec().provenance()
    assert p["calibrated"] is False
    assert p["is_measured_data"] is False
    assert p["kind"] == "modelling_assumption"
    ids = {b["id"] for b in p["assumptions"]}
    assert set("A1 A2 A3 A4 A5 A6 A7".split()).issubset(ids)
    for block in p["assumptions"]:
        assert block.get("bias_direction"), f"{block['id']} has no stated bias direction"
        assert isinstance(block.get("not_in_the_data"), bool), block["id"]
    # A1-A5 synthesise information the dataset does not contain; A6 (cold start
    # excluded) and A7 (trigger group as class label) are grounded in the source's
    # own documentation, so they are flagged differently on purpose.
    flags = {b["id"]: b["not_in_the_data"] for b in p["assumptions"]}
    assert all(flags[i] for i in ("A1", "A2", "A3", "A4", "A5"))
    assert not flags["A6"] and not flags["A7"]


def test_replay_source_manifest_never_claims_measurement(extract):
    src = AzureReplaySource(extract, APP, [(1, 0)], horizon_s=600.0)
    m = src.manifest()
    assert m["calibrated"] is False and m["is_measured_data"] is False
    assert "COUNT series" in m["what_is_real"]
    assert "service time" in m["what_is_reconstructed"]
    assert m["reconstruction"]["kind"] == "modelling_assumption"


# ---------------------------------------------------------------------------
# Version-stable transforms
# ---------------------------------------------------------------------------
def test_inverse_transform_helpers_have_the_right_moments():
    u = np_substream(1, "x").random(200_000)
    z = normal_from_uniform(u)
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1.0) < 0.02
    g = geometric_from_uniform(u, 4.0)
    assert g.min() >= 1 and abs(g.mean() - 4.0) < 0.1
    assert np.array_equal(geometric_from_uniform(u, 1.0), np.ones(u.size, dtype=np.int64))
    c = categorical_from_uniform(u, [1.0, 3.0])
    assert abs(np.mean(c) - 0.75) < 0.01
    p = poisson_from_uniform(u, np.full(u.size, 7.0))
    assert abs(p.mean() - 7.0) < 0.05 and abs(p.var() - 7.0) < 0.1


def test_sparse_application_yields_a_usable_but_tiny_trace(extract):
    """The corpus must not silently drop the regime where cold starts dominate."""
    starts = segment_starts(extract, SPARSE, 1, horizon_s=3600.0, n_segments=2,
                            min_invocations=1)
    out = reconstruct_segment(extract, SPARSE, 1, starts[0], 3600.0,
                              ReconstructionSpec(), seed=5)
    assert out["t_arrival"].size >= 1
    assert np.all(np.diff(out["t_arrival"]) >= 0)
