"""Side study: work-seed variance on the builtin `llm_serving` tasks.

WHY THIS EXISTS
---------------
The two task families in KubeGym have different seeding semantics and the
difference is easy to get wrong:

* **Corpus tasks** (`Corpus-<Family>-<Split>`): the env `seed` does **not**
  redraw the offered load.  A trace's realisation is fixed by its manifest seed
  so its checksum stays verifiable.  Two work seeds on the same episode give
  byte-identical rollouts, so a "work-seed replicate" is not a replicate.
* **Builtin `llm_serving` tasks**: `seed` **does** redraw work -- the per-request
  output lengths are drawn at build time from the (placeholder) length model.

Pooling the two would manufacture degrees of freedom.  This script measures the
work-seed component on `LLMServing-S1` so the main tables can state, with a
number rather than an assertion, that the corpus results carry none of it.

SCOPE, STATED PLAINLY
---------------------
Analytic controllers only.  PPO and DQN are **not** trained here, and the reason
is wall-clock, not a result: an `llm_serving` control step costs ~13 ms against
the ~0.7-2.2 ms quoted for `request_service`, so the same enforced budget
(`tune.CAP_ENV_STEPS` = 1.2 M train+dev steps per method) is about 4.3 h of pure
environment time per (algorithm, family) on this machine against 0.23-0.73 h for a
corpus family (measured RL wall-clock per (algorithm, family) was 0.14-0.64 h).
Running it at a smaller budget would break the parity the whole protocol exists
to enforce, and running it at the same budget was outside this track's time.
No conclusion about learned control on `llm_serving` is drawn or implied.

Controller parameters here are the source project's shipped tuned values where a
controller has them, and otherwise the class defaults; they are **not** re-tuned
on `llm_serving` dev, so these numbers are a variance measurement and not a
controller comparison.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence

import numpy as np

from .gym_controllers import HPAGym, REGISTRY
from .obsmap import ObsView
from .runner import rollout

TASK_ID = "KubeGym/LLMServing-S1-Abs-v0"
CONFIGS = {
    "static": {"k": 2},
    "hpa": {"metric": "kv", "target": 0.6, "tolerance": 0.1,
            "down_stabilization_s": 300.0},
    "queue": {"target_concurrency": 8.0, "hyst": 0.0, "cooldown_s": 0.0},
    "leading": {"per_replica_rps": 1.0, "kappa": 15.0, "window_s": 30.0,
                "backlog_target": 8.0},
    "forecast": {"forecaster": "holt", "alpha": 0.4, "beta": 0.3, "H_s": 30.0,
                 "per_replica_rps": 1.0, "safety": 1.2, "backlog_target": 12.0},
}


def run(out_json: str = "out/llm_side_variance.json", split: str = "test",
        n_episodes: int = 4, n_work_seeds: int = 4) -> Dict[str, Any]:
    import gymnasium
    from .runner import _ensure_registered
    _ensure_registered()
    env = gymnasium.make(TASK_ID, split=split)
    u = env.unwrapped
    view = ObsView(u.task_spec)
    cfg = u.cfg
    seeds = list(u.task_spec.split_work_seeds[split])[:n_work_seeds]
    eps = list(range(min(n_episodes, u.task_spec.n_episodes)))
    rows: List[Dict[str, Any]] = []
    for name, params in CONFIGS.items():
        ctrl = REGISTRY[name](view, cfg, action_mode="absolute", **params)
        for ep in eps:
            for ws in seeds:
                r = rollout(env, ctrl, ep, ws)
                r.update({"method": name, "params": json.dumps(params, sort_keys=True)})
                rows.append(r)
    env.close()

    # variance decomposition: within-episode (across work seeds) vs
    # between-episode, on the objective the benchmark scores
    out_methods: Dict[str, Any] = {}
    for name in CONFIGS:
        sub = [r for r in rows if r["method"] == name]
        by_ep: Dict[int, List[float]] = {}
        for r in sub:
            by_ep.setdefault(r["episode"], []).append(r["cost_total"])
        ep_means = np.asarray([np.mean(v) for v in by_ep.values()])
        within_sd = float(np.mean([np.std(v, ddof=1) for v in by_ep.values()
                                   if len(v) > 1]))
        out_methods[name] = {
            "params": CONFIGS[name],
            "n_episodes": len(by_ep), "n_work_seeds": len(seeds),
            "mean_cost_total": float(np.mean([r["cost_total"] for r in sub])),
            "sd_within_episode_across_work_seeds": within_sd,
            "sd_between_episodes": float(np.std(ep_means, ddof=1)) if len(ep_means) > 1
            else float("nan"),
            "identical_across_work_seeds": bool(within_sd == 0.0),
        }
    out = {
        "task": TASK_ID, "split": split, "work_seeds": seeds, "episodes": eps,
        "calibrated": bool(rows[0]["calibrated"]) if rows else None,
        "uncalibrated_required_fields": (rows[0].get("uncalibrated_required_fields")
                                        if rows else None),
        "semantics": ("on builtin llm_serving tasks the env seed DOES redraw the work "
                      "realisation, unlike the corpus tasks where it does not; the two "
                      "variance sources are never pooled"),
        "rl_not_run_reason": ("wall-clock only: ~13 ms per control step against the "
                              "~0.7-2.2 ms quoted for request_service, so the same enforced "
                              "1.2 M-step budget is ~4.3 h of environment time per (algorithm, "
                              "family) on this machine against 0.23-0.73 h for a corpus family. "
                              "Running a smaller budget would break budget parity."),
        "methods": out_methods, "rows": rows,
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    return out


if __name__ == "__main__":
    o = run()
    for m, d in o["methods"].items():
        print(f"{m:<9} mean_cost={d['mean_cost_total']:>8.2f} "
              f"sd_within_episode(work seeds)={d['sd_within_episode_across_work_seeds']:>7.3f} "
              f"sd_between_episodes={d['sd_between_episodes']:>7.3f}")
