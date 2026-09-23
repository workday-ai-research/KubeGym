"""Tests for the parametrized generator, especially the new `variable` family.

The `variable` family is defined in this package rather than inherited, so its
stated properties -- stationarity, the mean it claims to hold, the effect of tau
and sigma -- are pinned here.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from kubegym.workloads.corpus import parameter_grid
from kubegym.workloads.generator import (FAMILIES, BurstRate, Capacity, ConstantRate,
                                         DiurnalRate, ServiceTimeModel, SyntheticSpec,
                                         SyntheticSource, VariableRate, arrival_times,
                                         default_capacity, schedule_from_spec,
                                         synthetic_from_spec)
from kubegym.workloads.validation import acf, bin_counts, index_of_dispersion

CAP = Capacity(rps=4.0, source="test fixture, not calibrated")
SVC = ServiceTimeModel(kind="lognormal", mu=-0.7, sigma=1.0)


def _spec(schedule, horizon_s=3600.0, name="t", svc=SVC):
    return SyntheticSpec(name=name, family=schedule.family, horizon_s=horizon_s,
                         schedule=schedule, service=svc, capacity=CAP, seed=0,
                         tests="unit test")


def _rate(schedule, horizon_s=3600.0, seeds=(1, 2, 3, 4, 5, 6, 7, 8)):
    out = []
    for s in seeds:
        src = SyntheticSource(_spec(schedule, horizon_s))
        out.append(src.arrays(0, seed=s)["t_arrival"].size / horizon_s)
    return np.asarray(out)


# ---------------------------------------------------------------------------
# All four families exist and are named
# ---------------------------------------------------------------------------
def test_the_four_families_are_declared():
    assert set(FAMILIES) == {"constant", "variable", "burst", "diurnal"}


def test_parameter_grid_covers_every_family_and_states_what_each_point_tests():
    pts = parameter_grid()
    assert {p["family"] for p in pts} == set(FAMILIES)
    assert len(pts) >= 40
    for p in pts:
        assert p["tests"], p
        assert p["schedule"].family == p["family"]
    # every point must round-trip through its serialised spec
    for p in pts:
        assert schedule_from_spec(p["schedule"].spec()) == p["schedule"]


# ---------------------------------------------------------------------------
# Rate fidelity: the realised rate must match the schedule's stated mean
# ---------------------------------------------------------------------------
def test_constant_family_hits_its_stated_mean_rate():
    r = _rate(ConstantRate(load=1.0))
    assert abs(r.mean() - 4.0) / 4.0 < 0.03


def test_diurnal_family_hits_its_stated_mean_and_shows_the_period():
    sched = DiurnalRate(mean_load=1.0, amplitude=0.6, period_s=3600.0, phase=0.0)
    r = _rate(sched)
    assert abs(r.mean() - 4.0) / 4.0 < 0.05
    src = SyntheticSource(_spec(sched))
    t = src.arrays(0, seed=11)["t_arrival"]
    c = bin_counts(t, 3600.0, 60.0)
    assert acf(c)["lag1"] > 0.5, "a one-hour sinusoid must autocorrelate strongly at 1 min"


def test_burst_family_raises_the_rate_only_inside_the_burst():
    sched = BurstRate(base_load=0.5, magnitude=12.0, start_s=1200.0, duration_s=300.0)
    src = SyntheticSource(_spec(sched))
    t = src.arrays(0, seed=13)["t_arrival"]
    inside = ((t >= 1200.0) & (t < 1500.0)).sum() / 300.0
    outside = ((t < 1200.0) | (t >= 1500.0)).sum() / 3300.0
    assert abs(inside - 6.0 * 4.0) / (6.0 * 4.0) < 0.12   # 0.5 x 12 = 6.0 capacity units
    assert abs(outside - 0.5 * 4.0) / (0.5 * 4.0) < 0.12


def test_variable_family_is_stationary_at_its_stated_mean():
    """The `- sigma^2/2` correction: E[lam] must equal `mean_load x capacity`."""
    sched = VariableRate(mean_load=1.0, sigma=0.6, tau_s=300.0)
    r = _rate(sched, seeds=tuple(range(1, 25)))
    # clipping at 3 sigma biases the mean slightly low; the docstring claims ~0.3%
    assert 0.94 <= r.mean() / 4.0 <= 1.02, r.mean() / 4.0


def test_variable_family_sigma_controls_the_spread():
    lo = VariableRate(mean_load=1.0, sigma=0.15, tau_s=300.0)
    hi = VariableRate(mean_load=1.0, sigma=1.0, tau_s=300.0)
    def iod(sched):
        src = SyntheticSource(_spec(sched))
        t = src.arrays(0, seed=21)["t_arrival"]
        return index_of_dispersion(bin_counts(t, 3600.0, 60.0))
    assert iod(hi) > 3.0 * iod(lo)


def test_variable_family_tau_controls_the_memory():
    fast = VariableRate(mean_load=1.0, sigma=0.6, tau_s=60.0)
    slow = VariableRate(mean_load=1.0, sigma=0.6, tau_s=1200.0)
    def lag1(sched):
        vals = []
        for s in (1, 2, 3, 4, 5, 6):
            t = SyntheticSource(_spec(sched)).arrays(0, seed=s)["t_arrival"]
            v = acf(bin_counts(t, 3600.0, 60.0))["lag1"]
            if v is not None:
                vals.append(v)
        return float(np.median(vals))
    assert lag1(slow) > lag1(fast) + 0.2


def test_variable_random_walk_arm_is_not_stationary():
    """The documented non-stationary arm (tau = inf): spread grows with the horizon."""
    sched = VariableRate(mean_load=1.0, sigma=0.8, tau_s=float("inf"))
    short = _rate(sched, horizon_s=600.0, seeds=tuple(range(1, 25)))
    long = _rate(sched, horizon_s=3600.0, seeds=tuple(range(1, 25)))
    assert long.std() / long.mean() > short.std() / short.mean()


# ---------------------------------------------------------------------------
# Determinism, spec round-trip, and independence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sched", [
    ConstantRate(load=1.0),
    VariableRate(mean_load=1.0, sigma=0.5, tau_s=240.0),
    BurstRate(base_load=0.5, magnitude=4.0, start_s=600.0, duration_s=120.0),
    BurstRate(base_load=0.3, magnitude=6.0, start_s=600.0, duration_s=60.0,
              repeat_every_s=900.0),
    DiurnalRate(mean_load=1.0, amplitude=0.5, period_s=3600.0, phase=0.25),
])
def test_schedule_round_trips_through_its_spec(sched):
    again = schedule_from_spec(sched.spec())
    a = SyntheticSource(_spec(sched)).arrays(0, seed=31)
    b = SyntheticSource(_spec(again)).arrays(0, seed=31)
    assert np.array_equal(a["t_arrival"], b["t_arrival"])
    assert np.array_equal(a["service_time_s"], b["service_time_s"])


def test_synthetic_source_round_trips_through_json():
    src = SyntheticSource(_spec(VariableRate(mean_load=0.75, sigma=0.4, tau_s=180.0),
                                name="rt"))
    spec = json.loads(json.dumps(src.spec_obj.spec()))
    a = src.arrays(0, seed=41)
    b = synthetic_from_spec(spec).arrays(0, seed=41)
    assert np.array_equal(a["t_arrival"], b["t_arrival"])
    assert np.array_equal(a["service_time_s"], b["service_time_s"])


def test_arrival_and_service_streams_are_independent():
    sched = ConstantRate(load=1.0)
    a = SyntheticSource(_spec(sched)).arrays(0, seed=51)
    b = SyntheticSource(_spec(sched, svc=ServiceTimeModel(kind="lognormal", mu=0.7, sigma=1.0))
                        ).arrays(0, seed=51)
    assert np.array_equal(a["t_arrival"], b["t_arrival"])
    assert not np.array_equal(a["service_time_s"], b["service_time_s"])


def test_changing_the_rate_process_does_not_move_the_service_draws():
    """Same seed, same request count -> the first N service draws must agree."""
    a = SyntheticSource(_spec(ConstantRate(load=1.0), name="x")).arrays(0, seed=61)
    b = SyntheticSource(_spec(ConstantRate(load=1.5), name="x")).arrays(0, seed=61)
    n = min(a["service_time_s"].size, b["service_time_s"].size)
    assert np.array_equal(a["service_time_s"][:n], b["service_time_s"][:n])


def test_manifest_declares_the_rate_unit_and_is_uncalibrated():
    src = SyntheticSource(_spec(ConstantRate(load=1.0)))
    m = src.manifest()
    assert m["calibrated"] is False and m["is_measured_data"] is False
    assert "FRACTIONS of single-replica capacity" in m["rate_units"]
    assert m["capacity"]["calibrated"] is False


def test_default_capacity_is_labelled_a_placeholder():
    p = default_capacity().provenance()
    assert p["calibrated"] is False
    assert "placeholder" in json.dumps(p).lower()


def test_thinning_terminates_for_an_extreme_variable_spec():
    sched = VariableRate(mean_load=1.0, sigma=2.5, tau_s=30.0, clip_sigmas=3.0)
    t = SyntheticSource(_spec(sched, horizon_s=1800.0)).arrays(0, seed=71)["t_arrival"]
    assert t.size > 0 and np.all(np.diff(t) >= 0) and t.max() < 1800.0
