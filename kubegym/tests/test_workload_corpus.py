"""Corpus-level tests: determinism, regeneration from spec, splits, manifest, pairing.

These are the properties the benchmark's paired controller comparisons rest on.
If any of them fails, results computed over the corpus are not comparable across
controllers and the manifest is not a reproduction recipe.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from kubegym import ProvenancedConfig, Simulator, config_path
from kubegym.models.request_service import RequestServiceModel
from kubegym.tests.conftest_workloads import extract, real_extract, tiny_extract  # noqa: F401
from kubegym.workloads.azure2019 import DAY_SPLITS, SEED_BANDS, SPLIT_ORDER, AzureReplaySource
from kubegym.workloads.corpus import (CORPUS_SCHEMA, CorpusConfig, build_corpus, canonical_lines,
                                      load_manifest, plan_corpus, pooled_service_icdf,
                                      realise_entry, source_from_entry, split_rule_text,
                                      trace_digest, verify_manifest, verify_splits,
                                      write_manifest)
from kubegym.workloads.generator import default_capacity, synthetic_from_spec
from kubegym.workloads.reconstruction import ReconstructionSpec

APP = "tiny00" + "0" * 58


@pytest.fixture(scope="module")
def small_corpus(tmp_path_factory, extract):
    """A complete, small corpus built from the synthetic extract."""
    d = tmp_path_factory.mktemp("corpus")
    cfg = CorpusConfig(horizon_s=900.0, segments_per_app_day=1, seeds_per_grid_point=1,
                       materialise_per_family_split=1)
    man = build_corpus(extract, str(d), cfg, extract_paths={}, progress_every=0)
    write_manifest(man, os.path.join(str(d), "corpus_manifest.json"))
    return str(d), man


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_same_spec_and_seed_give_byte_identical_traces(small_corpus, extract):
    _, man = small_corpus
    for e in man["traces"][::17]:
        a, b = realise_entry(e, extract), realise_entry(e, extract)
        assert trace_digest(a["t_arrival"], a["service_time_s"]) == \
               trace_digest(b["t_arrival"], b["service_time_s"])
        assert np.array_equal(a["t_arrival"], b["t_arrival"])
        assert np.array_equal(a["service_time_s"], b["service_time_s"])


def test_different_seed_gives_a_different_trace(small_corpus, extract):
    _, man = small_corpus
    e = next(x for x in man["traces"] if x["n_requests"] > 50)
    other = dict(e, seed=e["seed"] + 1)
    a, b = realise_entry(e, extract), realise_entry(other, extract)
    assert trace_digest(a["t_arrival"], a["service_time_s"]) != \
           trace_digest(b["t_arrival"], b["service_time_s"])


def test_substreams_are_independent(extract):
    """Changing the service model must not move the arrival times, and vice versa."""
    base = ReconstructionSpec()
    a = AzureReplaySource(extract, APP, [(1, 0)], horizon_s=900.0, spec=base).arrays(0, seed=4)
    b = AzureReplaySource(extract, APP, [(1, 0)], horizon_s=900.0,
                          spec=ReconstructionSpec(min_service_time_s=0.25)
                          ).arrays(0, seed=4)
    assert np.array_equal(a["t_arrival"], b["t_arrival"])
    assert not np.array_equal(a["service_time_s"], b["service_time_s"])


# ---------------------------------------------------------------------------
# Regeneration from the spec alone
# ---------------------------------------------------------------------------
def test_every_trace_regenerates_from_its_manifest_entry(small_corpus, extract):
    d, man = small_corpus
    v = verify_manifest(man, extract, corpus_dir=d)
    assert v["ok"], v["mismatches"]
    assert v["n_checked"] == man["n_traces"]
    assert v["n_files_checked"] == len(man["materialised_files"])


def test_synthetic_trace_rebuilds_from_the_serialised_spec_only(small_corpus, extract):
    """No live object: the spec dict round-trips through JSON and still matches."""
    _, man = small_corpus
    e = next(x for x in man["traces"] if x["kind"] == "synthetic")
    spec = json.loads(json.dumps(e["synthetic_spec"]))
    src = synthetic_from_spec(spec)
    out = src.arrays(0, seed=e["seed"])
    assert trace_digest(out["t_arrival"], out["service_time_s"]) == e["sha256"]


def test_materialised_file_hash_equals_the_manifest_digest(small_corpus, extract):
    import hashlib
    d, man = small_corpus
    e = next(x for x in man["traces"] if x.get("file"))
    raw = open(os.path.join(d, e["file"]), "rb").read()
    assert hashlib.sha256(raw).hexdigest() == e["sha256"]
    out = realise_entry(e, extract)
    assert canonical_lines(out["t_arrival"], out["service_time_s"]) == raw


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def test_split_disjointness_is_verified_programmatically(small_corpus):
    _, man = small_corpus
    v = verify_splits(man)
    assert v["ok"], {k: x for k, x in v.items() if x and k != "ok"}
    assert not v["app_overlap"] and not v["day_overlap"] and not v["seed_overlap"]
    assert not v["segment_shared_across_splits"]
    assert v["duplicate_seeds_within_corpus"] == 0


def test_day_partition_covers_all_14_days_exactly_once():
    seen = sorted(d for s in SPLIT_ORDER for d in DAY_SPLITS[s])
    assert seen == list(range(1, 15))


def test_seed_bands_do_not_overlap():
    bands = sorted(SEED_BANDS[s] for s in SPLIT_ORDER)
    for (a0, a1), (b0, b1) in zip(bands, bands[1:]):
        assert a1 <= b0


def test_every_seed_lies_in_its_split_band(small_corpus):
    _, man = small_corpus
    for e in man["traces"]:
        lo, hi = SEED_BANDS[e["split"]]
        assert lo <= e["seed"] < hi, e["trace_id"]


def test_service_time_model_is_fitted_on_train_only(extract):
    _, report = pooled_service_icdf(extract, ReconstructionSpec())
    assert report["fitted_on_splits"] == ["train"]
    assert set(report["fitted_on_days"]) == set(DAY_SPLITS["train"])
    assert report["n_applications"] == len(extract.apps_in_split("train"))
    assert report["calibrated"] is False
    assert "no dev/test duration information" in report["leakage_note"]


def test_split_rule_text_is_present_and_mentions_every_axis(small_corpus):
    _, man = small_corpus
    txt = man["split_rule"]
    assert txt == split_rule_text()
    for word in ("application", "segment", "seed", "parameter"):
        assert word in txt.lower()


# ---------------------------------------------------------------------------
# Manifest completeness
# ---------------------------------------------------------------------------
REQUIRED_TOP = ("schema", "created_utc", "split_rule", "day_splits", "seed_bands",
                "reconstruction_provenance", "dataset", "license", "required_attribution",
                "families", "parameter_grid", "n_traces", "n_traces_by_family",
                "n_traces_by_split", "n_requests_total", "traces", "azure_app_hashes",
                "selection_report", "regeneration_command", "calibrated")
REQUIRED_TRACE = ("trace_id", "kind", "family", "split", "seed", "horizon_s", "sha256",
                  "n_requests", "tests", "calibrated")


def test_manifest_has_every_required_field(small_corpus):
    _, man = small_corpus
    assert man["schema"] == CORPUS_SCHEMA
    for k in REQUIRED_TOP:
        assert k in man, k
    assert man["calibrated"] is False
    assert man["reconstruction_provenance"]["kind"] == "modelling_assumption"
    for e in man["traces"]:
        for k in REQUIRED_TRACE:
            assert k in e, (e.get("trace_id"), k)
        assert e["calibrated"] is False
        assert e["tests"], e["trace_id"]
        if e["kind"] == "azure_replay":
            for k in ("app_hash", "day", "segment_start_minute", "regime", "volume_tercile"):
                assert k in e, k
        else:
            assert "synthetic_spec" in e


def test_manifest_records_the_azure_app_hashes_used(small_corpus, extract):
    _, man = small_corpus
    used = {e["app_hash"] for e in man["traces"] if e["kind"] == "azure_replay"}
    assert used == set(man["azure_app_hashes"])
    assert used.issubset({a.app_hash for a in extract.apps})


def test_manifest_is_json_round_trippable(small_corpus):
    d, man = small_corpus
    again = load_manifest(os.path.join(d, "corpus_manifest.json"))
    assert again["n_traces"] == man["n_traces"]
    assert [e["sha256"] for e in again["traces"]] == [e["sha256"] for e in man["traces"]]


def test_every_family_is_present_and_named(small_corpus):
    _, man = small_corpus
    assert set(man["families"]) == {"azure_replay", "constant", "variable", "burst", "diurnal"}
    for fam in man["families"]:
        assert man["n_traces_by_family"].get(fam, 0) > 0


# ---------------------------------------------------------------------------
# Build-time sampling and paired comparison
# ---------------------------------------------------------------------------
def _demands(source, k0, seed, horizon_s):
    cfg = ProvenancedConfig.load(config_path("request_service_default.json"))
    sim = Simulator(RequestServiceModel(cfg), source, cfg, k0=k0)
    sim.reset(episode=0, seed=seed)
    sim.step_to(horizon_s)
    sim.drain_all(t_cap=horizon_s * 4.0)
    return (np.array([r.demand for r in sim.requests]),
            np.array([r.t_arrival for r in sim.requests]))


@pytest.mark.parametrize("kind", ["azure_replay", "synthetic"])
def test_per_request_work_is_drawn_at_build_time_not_in_the_episode(small_corpus, extract, kind):
    """The paired-comparison guarantee of INTERFACE.md section 4.

    Two controllers (here, two static replica counts) must see the identical
    request stream: same arrival times AND same demands.
    """
    _, man = small_corpus
    e = next(x for x in man["traces"] if x["kind"] == kind and x["n_requests"] > 30)
    h = e["horizon_s"]
    d1, t1 = _demands(source_from_entry(e, extract), 1, e["seed"], h)
    d2, t2 = _demands(source_from_entry(e, extract), 8, e["seed"], h)
    assert np.array_equal(t1, t2)
    assert np.array_equal(d1, d2)
    assert d1.size == e["n_requests"]


def test_request_demand_equals_the_records_service_time(small_corpus, extract):
    _, man = small_corpus
    e = next(x for x in man["traces"] if x["kind"] == "azure_replay" and x["n_requests"] > 30)
    src = source_from_entry(e, extract)
    src.begin_build(0, e["seed"], None)
    recs = src.records(0)
    reqs = [src.make_request(r, i, None) for i, r in enumerate(recs)]
    assert np.allclose([r.demand for r in reqs], [r["service_time_s"] for r in recs])
    assert all(r.attrs.get("reconstructed") is True for r in reqs)


def test_arrays_and_records_agree(small_corpus, extract):
    _, man = small_corpus
    for e in man["traces"][::11]:
        src = source_from_entry(e, extract)
        src.begin_build(0, e["seed"], None)
        arr = src.arrays(0, seed=e["seed"])
        recs = src.records(0)
        assert len(recs) == arr["t_arrival"].size
        assert np.allclose([r["t_arrival"] for r in recs], arr["t_arrival"])
        assert np.allclose([r["service_time_s"] for r in recs], arr["service_time_s"])


def test_arrivals_are_inside_the_horizon(small_corpus, extract):
    _, man = small_corpus
    for e in man["traces"][::7]:
        out = realise_entry(e, extract)
        t = out["t_arrival"]
        if t.size:
            assert t.min() >= 0.0
            assert t.max() < e["horizon_s"] + 1e-6
