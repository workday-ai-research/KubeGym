"""Evaluate every method on the held-out **test** split.

THE PROTOCOL
------------
* Every method is run on **every test episode of every family**, with the
  parameters (analytic) or the policy weights (learned) that were selected on
  dev.  Nothing is selected here; the test split is opened once per method and
  the numbers it produces are the reported numbers.
* `Static-k` is run at **every feasible k** (1..`n_replicas_max`), giving the
  trivial reference front.  The dev-selected k is recorded so the aggregation can
  label one of those rows as the tuned `static` baseline without re-running it.
* Comparisons are **paired**: for a given (family, episode) every method faces a
  byte-identical trace, because the corpus samples per-request work at build time
  (`INTERFACE.md` section 4).  A method-vs-method contrast therefore carries no
  workload sampling variance.
* Dispersion is across **episodes** and, for the learners, across **training
  seeds**.  Never across work seeds: for corpus-backed tasks `seed` does not
  redraw the offered load, so a "work-seed replicate" would be a byte-identical
  rerun.  `llm_side.py` measures the other kind of variance, on the builtin
  `llm_serving` tasks where `seed` does redraw work, and the two are never
  pooled.
* Every row carries the calibration stamp (`cfg.stamp`) and the
  `uses_privileged_info` flag.
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .gym_controllers import REGISTRY
from .obsmap import ObsView
from .runner import (FAMILIES, load_manifest, make_env, regime_of, rollout,
                     trace_entries)

ANALYTIC = ("static", "hpa", "queue", "leading", "forecast", "oracle")


def load_tuned(family: str, out_dir: str = "out") -> Dict[str, Any]:
    with open(os.path.join(out_dir, f"tuned_{family}.json")) as fh:
        return json.load(fh)


def load_rl(family: str, algo: str, out_dir: str = "out/rl") -> Optional[Dict[str, Any]]:
    p = os.path.join(out_dir, f"train_{algo}_{family}.json")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def _static_k_max(cfg) -> int:
    return int(cfg.f("n_replicas_max"))


def evaluate_family(family: str, out_csv: Optional[str] = None,
                    out_dir: str = "out", rl_dir: str = "out/rl",
                    include_static_front: bool = True) -> List[Dict[str, Any]]:
    import pandas as pd
    manifest = load_manifest()
    entries = trace_entries(manifest, family, "test")
    tuned = load_tuned(family, out_dir)
    rows: List[Dict[str, Any]] = []

    envs: Dict[str, Any] = {}

    def env_for(mode: str):
        if mode not in envs:
            envs[mode] = make_env(family, "test", mode)
        return envs[mode]

    env = env_for("absolute")
    u = env.unwrapped
    view = ObsView(u.task_spec)
    cfg = u.cfg
    ws = int(u.task_spec.split_work_seeds["test"][0])
    n_eps = u.task_spec.n_episodes
    assert n_eps == len(entries), (n_eps, len(entries))

    def emit(method: str, seed: Optional[int], r: Dict[str, Any], ep: int,
             extra: Dict[str, Any]) -> None:
        e = dict(entries[ep])
        row = dict(r)
        row.update({
            "method": method,
            "policy_seed": (-1 if seed is None else int(seed)),
            "family": family,
            "regime": regime_of(e),
            "trace_id": e["trace_id"],
            "trace_n_requests": int(e["n_requests"]),
            "trace_mean_rate_rps": float(e.get("summary", {}).get("mean_rate_rps", float("nan"))),
        })
        row.update(extra)
        rows.append(row)

    # ---- analytic controllers at their dev-selected parameters -------
    for method in ANALYTIC:
        md = tuned["methods"][method]
        params = dict(md["selected_params"])
        cls = REGISTRY[method]
        ctrl = cls(view, cfg, action_mode="absolute", **params)
        for ep in range(n_eps):
            r = rollout(env, ctrl, ep, ws)
            emit(method, None, r, ep,
                 {"params": json.dumps(params, sort_keys=True, default=str),
                  "uses_privileged_info": cls.uses_privileged_info,
                  "is_adaptation": cls.is_adaptation,
                  "method_class": "analytic",
                  "tuning_budget_spent_env_steps": md["ledger"]["spent_env_steps"],
                  "tuning_budget_cap_env_steps": md["ledger"]["cap_env_steps"],
                  "n_configs_evaluated": md["n_configs_evaluated"]})
        print(f"[eval/{family}] {method} done", flush=True)

    # ---- the trivial Static-k reference front ------------------------
    if include_static_front:
        for k in range(1, _static_k_max(cfg) + 1):
            ctrl = REGISTRY["static"](view, cfg, action_mode="absolute", k=k)
            for ep in range(n_eps):
                r = rollout(env, ctrl, ep, ws)
                emit(f"static-k{k}", None, r, ep,
                     {"params": json.dumps({"k": k}),
                      "uses_privileged_info": False, "is_adaptation": False,
                      "method_class": "static_front",
                      "tuning_budget_spent_env_steps": 0,
                      "tuning_budget_cap_env_steps":
                          tuned["methods"]["static"]["ledger"]["cap_env_steps"],
                      "n_configs_evaluated": 0})
        print(f"[eval/{family}] static front done", flush=True)

    # ---- the learners, one row per (seed, episode) -------------------
    from stable_baselines3 import DQN, PPO
    for algo, cls_sb3 in (("ppo", PPO), ("dqn", DQN)):
        d = load_rl(family, algo, rl_dir)
        if d is None:
            print(f"[eval/{family}] {algo}: no training record, skipped", flush=True)
            continue
        mode = d["action_mode_selected"]
        e_rl = env_for(mode)
        for s in d["seeds"]:
            model = cls_sb3.load(s["model_path"], device="cpu")
            for ep in range(n_eps):
                r = rollout(e_rl, model, ep, ws)
                emit(algo, s["seed"], r, ep,
                     {"params": json.dumps(d["hp_selected"], sort_keys=True),
                      "uses_privileged_info": False, "is_adaptation": False,
                      "method_class": "learned",
                      "tuning_budget_spent_env_steps": d["ledger"]["spent_env_steps"],
                      "tuning_budget_cap_env_steps": d["ledger"]["cap_env_steps"],
                      "n_configs_evaluated": len(d["search"]),
                      "model_sha256": s.get("model_sha256")})
            del model
        print(f"[eval/{family}] {algo} done ({len(d['seeds'])} seeds)", flush=True)

    for e in envs.values():
        e.close()

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return rows


if __name__ == "__main__":
    import sys
    fams = sys.argv[1:] or list(FAMILIES)
    for fam in fams:
        t0 = time.perf_counter()
        rows = evaluate_family(fam, out_csv=f"out/eval_{fam}.csv")
        print(f"[eval] {fam}: {len(rows)} rows in {time.perf_counter() - t0:.0f} s",
              flush=True)
