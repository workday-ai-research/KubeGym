"""Build `regression_fixture.json`: reference numbers with tolerances.

TOLERANCE POLICY -- the point of the file is that it can actually fail
---------------------------------------------------------------------
Different quantities are reproducible to very different precision, so a single
tolerance would be either useless or a false alarm.  Three classes:

* **`exact`** -- integers that are properties of the protocol, not of a
  computation: budget spends, configuration counts, episode counts, the number of
  port-parity disagreements.  Any deviation is a protocol change and must fail.
* **`deterministic`** -- analytic controller metrics.  Given the parameters, the
  episode index and the work seed, a corpus rollout is a pure function: the
  offered load is fixed by the manifest seed and the controllers hold no
  randomness.  Reproduction should be bit-identical, so the tolerance is a
  floating-point epsilon (`rtol = 1e-9`), tight enough to catch a changed
  constant or a reordered accumulation.
* **`stochastic`** -- learned-policy metrics.  PyTorch reductions are not
  bit-reproducible across BLAS thread counts or minor versions, so the tolerance
  is derived from *measured* dispersion: `3 x` the across-seed standard deviation
  of the quantity, floored at 5 % relative.  A future run outside that band is
  evidence of a real change, not of numerical drift.  The seed dispersion used is
  recorded next to each entry so the tolerance is auditable rather than
  hand-picked.

Every entry also carries the calibration stamp, because a reproduced number from
an uncalibrated configuration is still not an empirical performance claim.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

HEADLINE = ("static", "hpa", "queue", "leading", "forecast", "ppo", "dqn", "oracle")
KEY_METRICS = ("cost_total", "slo_attainment", "replica_seconds", "scale_events",
               "latency_p95_s")
STOCH_SD_MULT = 3.0
STOCH_FLOOR_RTOL = 0.05
DET_RTOL = 1e-9


def _entry(value: float, cls: str, rtol: float, note: str = "",
           sd_seeds: Optional[float] = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {"value": None if value is None or not np.isfinite(value)
                         else float(value),
                         "tolerance_class": cls, "rtol": float(rtol)}
    if sd_seeds is not None and np.isfinite(sd_seeds):
        d["sd_across_seeds"] = float(sd_seeds)
    if note:
        d["note"] = note
    return d


def build(out_dir: str = "out", out_json: str = "out/regression_fixture.json"
          ) -> Dict[str, Any]:
    summary = pd.read_csv(os.path.join(out_dir, "results_summary.csv"))
    fam = summary[summary["scope"] == "family"]
    budget = pd.read_csv(os.path.join(out_dir, "tuning_budget.csv"))
    with open(os.path.join(out_dir, "port_parity.json")) as fh:
        parity = json.load(fh)

    fixture: Dict[str, Any] = {
        "schema": "kubegym-baselines-regression/1",
        "generated_from": ["results_summary.csv", "tuning_budget.csv",
                           "port_parity.json"],
        "calibration": {
            "calibrated": bool(fam["calibrated"].iloc[0]) if len(fam) else None,
            "uncalibrated_required_fields": (
                str(fam["uncalibrated_required_fields"].iloc[0]) if len(fam) else None),
            "statement": ("these are in-simulator numbers from an uncalibrated "
                          "configuration; reproducing them verifies the pipeline, not "
                          "any empirical performance claim"),
        },
        "tolerance_policy": {
            "exact": "integer protocol quantities; any deviation fails",
            "deterministic": f"analytic rollouts are pure functions; rtol = {DET_RTOL}",
            "stochastic": (f"learned policies; rtol = max({STOCH_SD_MULT} x "
                           f"sd_across_seeds / |value|, {STOCH_FLOOR_RTOL})"),
        },
        "port_parity": {
            "source_controllers_sha256": _entry(
                None, "exact", 0.0, parity["source_controllers_sha256"]),
            "total_shadowed_steps": _entry(
                sum(s["n_steps"] for s in parity["summary"]), "exact", 0.0),
            "total_disagreements": _entry(
                sum(s["n_disagree"] for s in parity["summary"]), "exact", 0.0,
                "must stay 0: the ported controllers are decision-identical to the "
                "source implementation on every shadowed control step"),
            "per_controller": {
                f"{s['service_model']}/{s['controller']}": {
                    "n_runs": s["n_runs"], "n_steps": s["n_steps"],
                    "n_disagree": s["n_disagree"], "kind": s["kind"],
                    "tolerance_class": "exact"}
                for s in parity["summary"]},
        },
        "oracle_property_checks": [
            {"family": c["family"],
             "P1a_demand_is_read": c["P1a_demand_is_read"],
             "P1b_future_arrivals_are_read": c["P1b_future_arrivals_are_read"],
             "P2_privilege_buys_anticipation": c["P2_privilege_buys_anticipation"],
             "tolerance_class": "exact",
             "note": "booleans must all stay True"}
            for c in parity.get("oracle_property_checks", [])],
        "budget": {}, "results": {},
    }

    for _, r in budget.iterrows():
        fixture["budget"][f"{r['family']}/{r['method']}"] = {
            "cap_env_steps": _entry(r["cap_env_steps"], "exact", 0.0),
            "spent_env_steps": _entry(r["spent_env_steps"], "exact", 0.0),
            "n_configs_evaluated": _entry(r["n_configs_evaluated"], "exact", 0.0),
        }

    for _, r in fam.iterrows():
        if r["method"] not in HEADLINE:
            continue
        key = f"{r['family']}/{r['method']}"
        learned = r["method_class"] == "learned"
        ent: Dict[str, Any] = {
            "method_class": r["method_class"],
            "uses_privileged_info": bool(r["uses_privileged_info"]),
            "is_adaptation": bool(r["is_adaptation"]),
            "n_episodes": _entry(r["n_episodes"], "exact", 0.0),
            "n_policy_seeds": _entry(r["n_policy_seeds"], "exact", 0.0),
            "params": str(r["params"]),
            "is_constant_target": bool(r.get("is_constant_target", False)),
        }
        for m in KEY_METRICS:
            v = r.get(f"{m}_mean", np.nan)
            sd = r.get(f"{m}_sd_seeds", np.nan)
            if learned:
                rtol = STOCH_FLOOR_RTOL
                if np.isfinite(sd) and np.isfinite(v) and abs(v) > 0:
                    rtol = max(STOCH_SD_MULT * sd / abs(v), STOCH_FLOOR_RTOL)
                ent[f"{m}_mean"] = _entry(v, "stochastic", rtol, sd_seeds=sd)
            else:
                ent[f"{m}_mean"] = _entry(v, "deterministic", DET_RTOL)
            ent[f"{m}_sd_episodes"] = _entry(
                r.get(f"{m}_sd_episodes", np.nan),
                "stochastic" if learned else "deterministic",
                STOCH_FLOOR_RTOL if learned else DET_RTOL)
        fixture["results"][key] = ent

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(fixture, fh, indent=1, default=str)
    return fixture


# ---------------------------------------------------------------------------
# the checker a future run calls
# ---------------------------------------------------------------------------
def check(fixture_path: str = "out/regression_fixture.json", out_dir: str = "out"
          ) -> Dict[str, Any]:
    """Re-check a fresh run against the fixture.  Returns pass/fail per entry."""
    with open(fixture_path) as fh:
        fx = json.load(fh)
    summary = pd.read_csv(os.path.join(out_dir, "results_summary.csv"))
    fam = summary[summary["scope"] == "family"].set_index(["family", "method"])
    failures: List[Dict[str, Any]] = []
    n_checked = 0
    for key, ent in fx["results"].items():
        f, m = key.split("/", 1)
        if (f, m) not in fam.index:
            failures.append({"key": key, "reason": "missing from results_summary.csv"})
            continue
        row = fam.loc[(f, m)]
        for field, spec in ent.items():
            if not isinstance(spec, dict) or "value" not in spec:
                continue
            n_checked += 1
            want, got = spec["value"], row.get(field, None)
            if want is None:
                continue
            if got is None or not np.isfinite(got):
                failures.append({"key": key, "field": field, "reason": "missing/NaN"})
                continue
            tol = spec["rtol"] * max(abs(want), 1e-12)
            if abs(got - want) > tol:
                failures.append({"key": key, "field": field, "expected": want,
                                 "got": float(got), "abs_tol": tol,
                                 "tolerance_class": spec["tolerance_class"]})
    return {"n_checked": n_checked, "n_failures": len(failures), "failures": failures,
            "passed": not failures}


if __name__ == "__main__":
    fx = build()
    print(json.dumps({"n_results": len(fx["results"]),
                      "n_budget": len(fx["budget"]),
                      "parity_disagreements":
                          fx["port_parity"]["total_disagreements"]["value"]}, indent=1))
