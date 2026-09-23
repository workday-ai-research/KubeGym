"""The claims gate, and the no-silent-upgrade rule."""
from __future__ import annotations

import json

import pytest

from kubegym import ProvenancedConfig, config_path
from kubegym.provenance import ProvenanceError


def _raw(fields):
    return {"schema": "kubegym-config-1", "calibration_note": "test", "fields": fields}


def test_claims_gate_is_and_over_required_fields():
    cfg = ProvenancedConfig.from_dict(_raw({
        "a": {"value": 1, "calibrated": True, "required_for_claims": True, "source": "measured"},
        "b": {"value": 2, "calibrated": False, "required_for_claims": True, "source": "guess"},
        "c": {"value": 3, "calibrated": False, "required_for_claims": False, "source": "guess"},
    }))
    assert cfg.calibrated is False
    assert cfg.uncalibrated_required == ["b"]

    ok = ProvenancedConfig.from_dict(_raw({
        "a": {"value": 1, "calibrated": True, "required_for_claims": True, "source": "measured"},
        "c": {"value": 3, "calibrated": False, "required_for_claims": False, "source": "guess"},
    }))
    assert ok.calibrated is True
    assert ok.uncalibrated_required == []
    with pytest.raises(RuntimeError):
        cfg.require_calibrated("a paper number")
    ok.require_calibrated("a paper number")           # must not raise


def test_field_without_source_is_refused():
    with pytest.raises(ProvenanceError):
        ProvenancedConfig.from_dict(_raw({"a": {"value": 1, "calibrated": True}}))
    with pytest.raises(ProvenanceError):
        ProvenancedConfig.from_dict(_raw({"a": 1}))    # bare constant, no provenance


def test_override_cannot_silently_upgrade_a_placeholder():
    cfg = ProvenancedConfig.from_dict(_raw({
        "p": {"value": 6000.0, "calibrated": False, "required_for_claims": True,
              "source": "PLACEHOLDER, never measured"},
    }))
    assert cfg.calibrated is False
    bumped = cfg.override("p", 9000.0, source="sweep value")
    assert bumped.is_calibrated("p") is False
    assert bumped.calibrated is False
    assert bumped.provenance()["overrides"][0]["field"] == "p"

    attested = cfg.override("p", 9000.0, source="measured on the testbed",
                            attest="n=12 prefill sweep, median 9000 tok/s")
    assert attested.is_calibrated("p") is True
    assert attested.calibrated is True
    assert "ATTESTED MEASUREMENT" in attested.source("p")


def test_override_of_a_measured_field_drops_calibration():
    cfg = ProvenancedConfig.from_dict(_raw({
        "m": {"value": 31.9, "calibrated": True, "required_for_claims": True,
              "source": "MEASURED n=9"},
    }))
    assert cfg.calibrated is True
    moved = cfg.overridden(m=60.0)
    assert moved.is_calibrated("m") is False, "changing a measured value must drop the flag"
    assert moved.calibrated is False


def test_stamp_attaches_the_gate_to_a_result_row():
    cfg = ProvenancedConfig.load(config_path("llm_serving_l40s.json"))
    row = cfg.stamp({"controller": "hpa", "replica_seconds": 1234.5})
    assert row["calibrated"] is False
    assert "output_length_distribution" in row["uncalibrated_required_fields"]
    assert row["config"] == "llm_serving_l40s.json"
    assert "decode_batch_extrapolation_above_8" in row["flagged_optimistic_fields"]


def test_shipped_llm_config_keeps_the_measured_constants():
    cfg = ProvenancedConfig.load(config_path("llm_serving_l40s.json"))
    assert cfg.calibrated is False
    assert cfg.f("n_replicas_max") == 4 and cfg.is_calibrated("n_replicas_max")
    assert cfg.f("kv_tokens_per_replica") == 55200 and cfg.is_calibrated("kv_tokens_per_replica")
    assert cfg.f("replica_cold_start_s") == 31.9 and cfg.is_calibrated("replica_cold_start_s")
    st = cfg.f("decode_step_time_s")
    assert st["a0_s"] == 0.01657539 and st["a1_s_per_seq"] == 0.00011063
    assert cfg.is_calibrated("decode_step_time_s")
    assert cfg.field("replica_teardown_s").weak_evidence is True
    # placeholders that block reporting
    assert set(cfg.uncalibrated_required) == {
        "output_length_distribution", "prefill_tokens_per_s",
        "decode_batch_extrapolation_above_8", "slo"}
    assert cfg.field("decode_batch_extrapolation_above_8").flagged_optimistic is True


def test_shipped_request_service_config_is_entirely_placeholder():
    cfg = ProvenancedConfig.load(config_path("request_service_default.json"))
    assert cfg.calibrated is False
    for name in ("replica_cold_start_s", "rs_max_concurrency", "rs_service_time_distribution",
                 "rs_contention_slowdown"):
        assert cfg.is_calibrated(name) is False, f"{name} must not claim to be measured"
        assert "PLACEHOLDER" in cfg.source(name)


def test_subset_reports_missing_fields():
    cfg = ProvenancedConfig.load(config_path("llm_serving_l40s.json"))
    sub = cfg.subset(["kv_tokens_per_replica", "prefill_tokens_per_s"])
    assert sub.names() == ["kv_tokens_per_replica", "prefill_tokens_per_s"]
    assert sub.calibrated is False              # prefill is a required placeholder
    with pytest.raises(KeyError):
        cfg.subset(["kv_tokens_per_replica", "no_such_field"])


def test_roundtrip_preserves_provenance(tmp_path):
    cfg = ProvenancedConfig.load(config_path("llm_serving_l40s.json"))
    p = tmp_path / "out.json"
    cfg.save(p)
    back = ProvenancedConfig.load(p)
    for name in cfg.names():
        assert back.f(name) == cfg.f(name)
        assert back.is_calibrated(name) == cfg.is_calibrated(name)
        assert back.source(name) == cfg.source(name)
    assert back.calibrated == cfg.calibrated
    assert json.loads(p.read_text())["uncalibrated_required_fields"] == cfg.uncalibrated_required
