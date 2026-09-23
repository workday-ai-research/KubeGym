#!/usr/bin/env python3
"""`kubegym verify` — the reproduction harness.

One entry point that re-checks every claim the artifact makes about itself, and
exits non-zero if any of them has drifted. Run it on a clean machine after
`pip install -e .[all]`.

Checks, in order:

  1. environment      — Python and the versions of every load-bearing dependency
  2. tests            — the full pytest suite
  3. calibration      — CALIBRATION.md regenerated from the configs and compared
  4. claims gate      — every shipped config reports calibrated=false, and the
                        reason is a labelled placeholder (not a missing field)
  5. corpus           — all 767 traces regenerate from spec with matching
                        sha256, and the split-disjointness rules hold
  6. gym registry     — every builtin and corpus task constructs, and one
                        episode runs end to end
  7. determinism      — a fixed action sequence replays bit-identically
  8. parity           — bit-exact parity against the source simulator
                        (SKIPPED unless KUBEGYM_SOURCE_TESTBED is set)
  9. reference results— published numbers reproduce within the tolerances in
                        regression_fixture.json (SKIPPED if absent)

Each check prints PASS / FAIL / SKIP with a one-line reason. A SKIP is never
counted as a PASS, and the summary states how many checks were skipped, because
"it passed" and "it did not run" are different claims.

Usage:
    python tools/verify.py                    # everything available
    python tools/verify.py --corpus DIR       # corpus location
    python tools/verify.py --quick            # skip the slow checks (tests, corpus)
    python tools/verify.py --list             # list checks without running
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
Result = Tuple[str, str]          # (status, message)

DEPS = ["numpy", "gymnasium", "pandas", "pyarrow", "scipy", "torch",
        "stable_baselines3", "matplotlib", "pytest"]


# --------------------------------------------------------------------------- #
def check_environment(ctx) -> Result:
    rows = [f"python {sys.version.split()[0]}"]
    missing = []
    for d in DEPS:
        try:
            m = importlib.import_module(d)
            rows.append(f"{d} {getattr(m, '__version__', '?')}")
        except Exception:
            missing.append(d)
    ctx["env"] = rows
    msg = "; ".join(rows)
    if missing:
        # torch/sb3 are only needed for the RL baselines
        hard = [d for d in missing if d in ("numpy", "gymnasium")]
        if hard:
            return FAIL, f"missing required: {', '.join(hard)}"
        return PASS, msg + f"  (absent, RL checks will skip: {', '.join(missing)})"
    return PASS, msg


def check_tests(ctx) -> Result:
    if ctx["quick"]:
        return SKIP, "--quick"
    env = dict(os.environ)
    if ctx["corpus"]:
        env["KUBEGYM_CORPUS_DIR"] = ctx["corpus"]
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "kubegym/tests", "-q",
                        "--no-header", "-x"],
                       capture_output=True, text=True, env=env, cwd=ctx["root"])
    tail = [l for l in p.stdout.strip().splitlines() if l.strip()][-1:] or [""]
    if p.returncode != 0:
        return FAIL, f"pytest exit {p.returncode}: {tail[0][:200]}"
    return PASS, f"{tail[0][:120]} in {time.time() - t0:.0f}s"


def check_calibration_report(ctx) -> Result:
    script = os.path.join(ctx["root"], "tools", "gen_calibration_report.py")
    out = os.path.join(ctx["root"], "CALIBRATION.md")
    if not os.path.isfile(script):
        return SKIP, "tools/gen_calibration_report.py absent"
    p = subprocess.run([sys.executable, script, "--check", "--out", out],
                       capture_output=True, text=True, cwd=ctx["root"])
    if p.returncode != 0:
        return FAIL, (p.stderr or p.stdout).strip()[:200]
    return PASS, "CALIBRATION.md matches the shipped configs"


def check_claims_gate(ctx) -> Result:
    from kubegym import config_path
    from kubegym.provenance import ProvenancedConfig
    notes = []
    for name in ("llm_serving_l40s.json", "request_service_default.json"):
        cfg = ProvenancedConfig.load(config_path(name))
        if cfg.calibrated:
            return FAIL, (f"{name} reports calibrated=True; the shipped artifact must not "
                          f"claim calibration it does not have")
        blocking = list(cfg.uncalibrated_required)
        if not blocking:
            return FAIL, f"{name} is uncalibrated but names no blocking field"
        for n in blocking:
            fl = cfg.field(n)
            if not (fl.source or "").strip():
                return FAIL, f"{name}:{n} is a placeholder with no source string"
        notes.append(f"{name}: {len(blocking)} blocking ({', '.join(blocking)})")
    return PASS, "; ".join(notes)


def check_corpus(ctx) -> Result:
    if not ctx["corpus"]:
        return SKIP, "no --corpus / KUBEGYM_CORPUS_DIR"
    if ctx["quick"]:
        return SKIP, "--quick"
    p = subprocess.run([sys.executable, "-m", "kubegym.workloads.cli", "verify",
                        "--corpus", ctx["corpus"]],
                       capture_output=True, text=True, cwd=ctx["root"])
    if p.returncode != 0:
        return FAIL, (p.stderr or p.stdout).strip()[-200:]
    try:
        d = json.loads(p.stdout)
    except Exception:
        return FAIL, "verify did not return JSON"
    reg, spl = d.get("regeneration", {}), d.get("splits", {})
    if not (reg.get("ok") and spl.get("ok")):
        return FAIL, (f"regeneration ok={reg.get('ok')} "
                      f"mismatches={reg.get('n_mismatch')}; splits ok={spl.get('ok')}")
    ctx["corpus_n"] = reg.get("n_checked")
    return PASS, (f"{reg.get('n_checked')} traces regenerate with matching sha256; "
                  f"splits disjoint")


def _register(ctx):
    """Register builtin + corpus tasks once per run (registration is global)."""
    from kubegym.gym import list_tasks, register_builtin_tasks
    if "n_corpus" in ctx:
        return list_tasks(), ctx["n_corpus"]
    register_builtin_tasks()
    n_corpus = 0
    if ctx["corpus"]:
        try:
            from kubegym.workloads import register_corpus_tasks
            n_corpus = len(register_corpus_tasks(ctx["corpus"]))
        except Exception as e:                                   # pragma: no cover
            ctx["corpus_reg_error"] = f"{type(e).__name__}: {e}"
    ctx["n_corpus"] = n_corpus
    return list_tasks(), n_corpus


def check_gym_registry(ctx) -> Result:
    try:
        import gymnasium as gym
    except Exception as e:
        return SKIP, f"gymnasium unavailable ({e})"
    tasks, n_corpus = _register(ctx)
    if "corpus_reg_error" in ctx:
        return FAIL, "corpus task registration failed: " + ctx["corpus_reg_error"]
    bad = []
    for t in tasks:
        try:
            e = gym.make(t["id"])
            e.close()
        except Exception as ex:
            bad.append(f"{t['id']} ({type(ex).__name__})")
    if bad:
        return FAIL, "could not construct: " + ", ".join(bad[:4])
    # one full episode on one task, to prove the loop closes
    t = tasks[0]
    e = gym.make(t["id"])
    opts = {"episode": 0, "work_seed": 0}
    e.reset(seed=0, options=opts)
    n, done = 0, False
    while not done:
        _, _, term, trunc, info = e.step(0)
        done, n = term or trunc, n + 1
    e.close()
    if info.get("calibrated") is not False:
        return FAIL, "info['calibrated'] is not False on an uncalibrated config"
    return PASS, (f"{len(tasks)} tasks construct ({n_corpus} corpus ids); "
                  f"{n} steps ran on {t['id']}, calibration flag propagated")


def check_determinism(ctx) -> Result:
    try:
        import gymnasium as gym
        import numpy as np
    except Exception as e:
        return SKIP, f"gymnasium unavailable ({e})"
    tasks, _ = _register(ctx)
    t = tasks[0]
    e = gym.make(t["id"])

    def roll():
        e.reset(seed=7, options={"episode": 0, "work_seed": 3})
        out, done, i = [], False, 0
        while not done:
            _, r, term, trunc, _ = e.step(i % int(e.action_space.n))
            out.append(r)
            done = term or trunc
            i += 1
        return np.asarray(out)

    a, b = roll(), roll()
    e.close()
    if not np.array_equal(a, b):
        return FAIL, f"same (episode, seed) gave different rollouts (max |d|={np.max(np.abs(a-b)):g})"
    return PASS, f"{len(a)} steps replay bit-identically on {t['id']}"


def check_parity(ctx) -> Result:
    src = os.environ.get("KUBEGYM_SOURCE_TESTBED")
    if not src:
        return SKIP, "KUBEGYM_SOURCE_TESTBED not set (source simulator not present)"
    p = subprocess.run([sys.executable, "-m", "pytest", "kubegym/tests/test_parity.py",
                        "-q", "--no-header"],
                       capture_output=True, text=True, cwd=ctx["root"])
    tail = [l for l in p.stdout.strip().splitlines() if l.strip()][-1:] or [""]
    return (PASS if p.returncode == 0 else FAIL), tail[0][:160]


def check_reference_results(ctx) -> Result:
    fx = ctx["fixture"]
    if not fx or not os.path.isfile(fx):
        return SKIP, "regression_fixture.json absent"
    d = json.load(open(fx))
    if d.get("schema", "").split("/")[0] != "kubegym-baselines-regression":
        return FAIL, f"unexpected fixture schema {d.get('schema')!r}"

    policy = d.get("tolerance_policy") or {}
    if not policy:
        return FAIL, "fixture declares no tolerance_policy"

    # Walk every leaf that carries a reference number and check it is fully
    # specified: a value, a tolerance CLASS drawn from the declared policy, and
    # an rtol. A number without a stated tolerance is not a reproducible claim.
    n_num, bad = 0, []

    def walk(node, path):
        nonlocal n_num
        if isinstance(node, dict):
            if "value" in node and "tolerance_class" in node:
                n_num += 1
                tc = node["tolerance_class"]
                if tc not in policy:
                    bad.append(f"{path}: tolerance_class {tc!r} not in policy")
                if "rtol" not in node:
                    bad.append(f"{path}: no rtol")
                elif tc == "stochastic" and not node["rtol"] > 0:
                    bad.append(f"{path}: stochastic entry with rtol={node['rtol']!r}")
                return
            for k, v in node.items():
                walk(v, f"{path}/{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    for key in ("results", "budget", "port_parity", "oracle_property_checks"):
        walk(d.get(key), key)

    if bad:
        return FAIL, f"{len(bad)} malformed entries, e.g. {bad[0]}"
    if not n_num:
        return FAIL, "fixture contains no reference numbers"

    cal = d.get("calibration") or {}
    if cal.get("calibrated") is not False:
        return FAIL, ("fixture does not record calibrated=false; reference numbers from an "
                      "uncalibrated configuration must say so")

    # Be explicit about what this check does and does not establish.
    return PASS, (f"{n_num} reference numbers, all with a declared tolerance class "
                  f"({', '.join(sorted(policy))}); calibration=false recorded. "
                  f"STRUCTURAL check only -- recomputing the numbers needs the baselines "
                  f"scripts (see BASELINES.md)")


CHECKS: List[Tuple[str, Callable]] = [
    ("environment", check_environment),
    ("tests", check_tests),
    ("calibration-report", check_calibration_report),
    ("claims-gate", check_claims_gate),
    ("corpus", check_corpus),
    ("gym-registry", check_gym_registry),
    ("determinism", check_determinism),
    ("parity", check_parity),
    ("reference-results", check_reference_results),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="KubeGym reproduction harness")
    ap.add_argument("--corpus", default=os.environ.get("KUBEGYM_CORPUS_DIR"),
                    help="corpus directory (contains corpus_manifest.json)")
    ap.add_argument("--fixture", default="regression_fixture.json")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--quick", action="store_true", help="skip pytest and corpus regeneration")
    ap.add_argument("--only", default="", help="comma-separated check names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", default="", help="also write the report as JSON here")
    a = ap.parse_args()

    if a.list:
        for n, _ in CHECKS:
            print(n)
        return 0

    ctx: Dict = {"corpus": os.path.abspath(a.corpus) if a.corpus else None,
                 "fixture": a.fixture, "root": a.root, "quick": a.quick}
    only = {s.strip() for s in a.only.split(",") if s.strip()}

    print(f"KubeGym verify — root {a.root}")
    print(f"corpus: {ctx['corpus'] or '(none)'}\n")
    results, width = [], max(len(n) for n, _ in CHECKS)
    for name, fn in CHECKS:
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            status, msg = fn(ctx)
        except Exception as e:
            status, msg = FAIL, f"{type(e).__name__}: {e}"
        results.append({"check": name, "status": status, "message": msg,
                        "seconds": round(time.time() - t0, 2)})
        print(f"  [{status}] {name:<{width}}  {msg}")

    n_fail = sum(r["status"] == FAIL for r in results)
    n_skip = sum(r["status"] == SKIP for r in results)
    n_pass = sum(r["status"] == PASS for r in results)
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped"
          f"{' — a skipped check is not a passed check' if n_skip else ''}")
    if a.json:
        json.dump({"results": results, "n_pass": n_pass, "n_fail": n_fail,
                   "n_skip": n_skip}, open(a.json, "w"), indent=1)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
