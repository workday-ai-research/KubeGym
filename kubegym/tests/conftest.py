"""Shared fixtures.

The parity test needs the SOURCE testbed (`sim_cluster.py`, `length_model.py`,
`controllers.py`) to compare against.  Point `KUBEGYM_SOURCE_TESTBED` at the
directory holding those files; without it the parity test SKIPS with an explicit
message rather than silently passing.  `parity_report.md` records the result of
the run that was actually executed.
"""
from __future__ import annotations

import os
import sys

import pytest

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture(scope="session")
def data_dir() -> str:
    return DATA


@pytest.fixture(scope="session")
def trace_s1() -> str:
    return os.path.join(DATA, "S1_dev.jsonl")


@pytest.fixture(scope="session")
def trace_s3() -> str:
    return os.path.join(DATA, "S3_dev.jsonl")


@pytest.fixture(scope="session")
def llm_cfg():
    from kubegym import ProvenancedConfig, config_path
    return ProvenancedConfig.load(config_path("llm_serving_l40s.json"))


@pytest.fixture(scope="session")
def rs_cfg():
    from kubegym import ProvenancedConfig, config_path
    return ProvenancedConfig.load(config_path("request_service_default.json"))


@pytest.fixture(scope="session")
def length_model():
    from kubegym.models.length_model import load_length_model
    # No measured_lengths.json in the repo: this returns the explicitly
    # labelled PLACEHOLDER, and `calibrated` is False.
    return load_length_model("measured_lengths.json")


@pytest.fixture(scope="session")
def source_testbed():
    """The source testbed module, or a skip."""
    path = os.environ.get("KUBEGYM_SOURCE_TESTBED", "")
    if not path or not os.path.isdir(path):
        pytest.skip("set KUBEGYM_SOURCE_TESTBED to the source testbed directory "
                    "(sim_cluster.py, length_model.py, controllers.py) to run the parity test")
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        import sim_cluster  # noqa: F401
    except Exception as e:                                    # pragma: no cover
        pytest.skip(f"cannot import sim_cluster from {path}: {e}")
    return path
