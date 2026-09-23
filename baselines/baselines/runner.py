"""Episode rollout, metric extraction, corpus metadata, and the budget ledger.

Everything here is shared by tuning, training and evaluation so that a metric
means the same thing in all three.  Three rules are enforced mechanically rather
than asserted in prose:

* **`assert_not_test(split)`** raises if a `test` split is touched outside
  `evaluate.py`.  It is called by the tuner and by the trainer before any env is
  built.
* **`BudgetLedger`** counts every environment step charged to a method and raises
  `BudgetExceeded` past the cap.  The cap is per (method, family); the ledger is
  serialised into the results so the spend is auditable, not claimed.
* **Every episode row carries the calibration stamp** via `cfg.stamp`, because a
  `calibrated=False` number is not an empirical result.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

FAMILIES = ("azure_replay", "constant", "variable", "burst", "diurnal")
FAMILY_TITLE = {"azure_replay": "AzureReplay", "constant": "Constant",
                "variable": "Variable", "burst": "Burst", "diurnal": "Diurnal"}


# ---------------------------------------------------------------------------
# split discipline
# ---------------------------------------------------------------------------
class TestSplitTouched(RuntimeError):
    """Raised when tuning or training reaches for the held-out split."""


def assert_not_test(split: str, what: str = "") -> None:
    if str(split).lower() == "test":
        raise TestSplitTouched(
            f"{what or 'this phase'} attempted to open the 'test' split. Selection happens on "
            "dev; the test split is opened exactly once per method, in evaluate.py. This is the "
            "failure mode arXiv:2608.07303 is about, so it raises rather than warns.")


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------
class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetLedger:
    """Environment-step budget for one (method, family), enforced in code.

    The currency is **environment steps taken on the train and dev splits**:
    every step of a tuning sweep, of RL training, and of dev evaluation is
    charged here.  Test-split steps are not charged, because they are not
    search -- they are the single measurement at the end.

    One number per method per family, so the parity claim is checkable by
    reading `spend` against `cap`.
    """

    method: str
    family: str
    cap: int
    spent: int = 0
    by_phase: Dict[str, int] = field(default_factory=dict)
    trials: List[Dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0

    def charge(self, n: int, phase: str) -> None:
        self.spent += int(n)
        self.by_phase[phase] = self.by_phase.get(phase, 0) + int(n)
        if self.spent > self.cap:
            raise BudgetExceeded(
                f"{self.method}/{self.family}: budget of {self.cap} train+dev env steps "
                f"exhausted (spent {self.spent} after charging {n} to phase {phase!r}). "
                "Raise the cap explicitly and re-run every method at the new cap, or reduce "
                "the sweep -- silently overspending is what makes short-budget comparisons "
                "unreproducible.")

    def log_trial(self, params: Dict[str, Any], objective: float, extra: Dict[str, Any]) -> None:
        self.trials.append({"trial": len(self.trials), "params": params,
                            "objective": float(objective), **extra})

    def to_dict(self) -> Dict[str, Any]:
        return {"method": self.method, "family": self.family, "cap_env_steps": self.cap,
                "spent_env_steps": self.spent, "by_phase": dict(self.by_phase),
                "n_trials": len(self.trials), "wall_s": round(self.wall_s, 2),
                "headroom_env_steps": self.cap - self.spent}


# ---------------------------------------------------------------------------
# corpus metadata
# ---------------------------------------------------------------------------
def load_manifest(corpus_dir: Optional[str] = None) -> Dict[str, Any]:
    root = corpus_dir or os.environ["KUBEGYM_CORPUS_DIR"]
    with open(os.path.join(root, "corpus_manifest.json")) as fh:
        return json.load(fh)


def trace_entries(manifest: Dict[str, Any], family: str, split: str) -> List[Dict[str, Any]]:
    """Manifest entries in the order `register_corpus_tasks` uses.

    `kubegym.workloads.gym_tasks._group` sorts by `trace_id`, and the Gym task's
    `episode` index is the position in that sorted list, so this is the exact
    episode -> trace mapping.
    """
    g = [e for e in manifest["traces"]
         if e.get("family") == family and e.get("split") == split]
    return sorted(g, key=lambda e: e["trace_id"])


def regime_of(entry: Dict[str, Any]) -> str:
    """A reportable regime label for one trace.

    For `azure_replay` this is the selection regime of the source application
    (`sparse` / `intermittent` / `steady` / `diurnal` / `bursty`), which is the
    axis `WORKLOADS.md` section 3 says is the reportable unit -- the corpus
    deliberately oversamples the rare ones, so a corpus average is not a
    population average.  For the synthetic families it is the grid point, since a
    family's difficulty is set by its schedule parameters.
    """
    if entry.get("family") == "azure_replay":
        return str(entry.get("regime", "unknown"))
    sched = (entry.get("synthetic_spec") or {}).get("schedule") or {}
    parts = [f"{k}={v}" for k, v in sorted(sched.items()) if k != "kind"]
    return f"{sched.get('kind', 'unknown')}[" + ",".join(parts) + "]"


# ---------------------------------------------------------------------------
# env construction
# ---------------------------------------------------------------------------
def task_id(family: str, split: str, action_mode: str = "absolute") -> str:
    mode = {"absolute": "Abs", "delta": "Delta"}[action_mode]
    return f"KubeGym/Corpus-{FAMILY_TITLE[family]}-{split.capitalize()}-{mode}-v0"


def make_env(family: str, split: str, action_mode: str = "absolute"):
    import gymnasium
    _ensure_registered()
    return gymnasium.make(task_id(family, split, action_mode), split=split)


_REGISTERED = False


def _ensure_registered() -> None:
    """Register the corpus tasks and install the engine stall guard.

    The guard is installed here, at the single choke point every entry point goes
    through, so it is impossible to run one method with it and another without.
    See `engine_guard` for the defect it works around.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    from .engine_guard import install as _install_engine_guard
    _install_engine_guard()
    import kubegym.gym  # noqa: F401
    from kubegym.workloads import register_corpus_tasks
    try:
        register_corpus_tasks()
    except Exception as exc:                                  # already registered
        if "already registered" not in str(exc).lower():
            raise
    _REGISTERED = True


def episode_list(env, split: str) -> List[Tuple[int, int]]:
    """`(episode, work_seed)` pairs covering every distinct **episode** once.

    For corpus-backed tasks `work_seed` does not redraw the offered load -- a
    trace's realisation is fixed by its manifest seed so its checksum stays
    verifiable -- so varying it would produce identical rollouts and a fake
    replicate.  One work seed is therefore held fixed and the episode index is
    what varies.  Dispersion is reported across episodes and across policy /
    training seeds, never across work seeds.
    """
    spec = env.unwrapped.task_spec
    ws = int(spec.split_work_seeds[split][0])
    return [(e, ws) for e in range(spec.n_episodes)]


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------
def _percentile(vals: Sequence[float], q: float) -> float:
    if not len(vals):
        return float("nan")
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def rollout(env, policy, episode: int, work_seed: int,
            on_step: Optional[Callable[[int, Dict[str, Any]], None]] = None
            ) -> Dict[str, Any]:
    """Run one episode under `policy(obs) -> action` and return a metric row.

    `policy` may be a `GymController` (via `.act`), a callable, or an SB3 model
    (via `.predict`).  Controllers get `reset(env)` called first so a privileged
    controller can bind to the trace.
    """
    obs, info = env.reset(options={"episode": int(episode), "work_seed": int(work_seed)})
    if hasattr(policy, "reset") and hasattr(policy, "act"):
        policy.reset(env)
    act = _policy_fn(policy)

    t_wall = time.perf_counter()
    churn_total = 0.0
    scale_events = 0
    targets: List[int] = []
    n_clamped = 0
    n_sat_steps = 0
    step_i = 0
    terminated = truncated = False
    while not (terminated or truncated):
        a = act(obs)
        prev = int(info["target"]) if "target" in info else int(env.unwrapped.sim.target)
        obs, reward, terminated, truncated, info = env.step(a)
        step_i += 1
        ch = abs(int(info["target"]) - prev)
        churn_total += ch
        scale_events += int(ch > 0)
        targets.append(int(info["target"]))
        n_clamped += int(bool(info.get("action_clamped")))
        if on_step is not None:
            on_step(step_i, info)
    n_sat_steps = int(getattr(policy, "n_saturated_steps", 0))

    ec = dict(info["episode_cost"])
    drain = dict(info.get("post_horizon_drain", {}))
    summary = dict(info.get("episode_summary", {}))
    sim = env.unwrapped.sim
    reqs = sim.requests
    lat = [r.latency for r in reqs if r.latency is not None]
    qd = [r.queue_delay for r in reqs if r.queue_delay is not None]
    n_done = sum(1 for r in reqs if r.done)
    slo = env.unwrapped.cost_model
    targets_arr = np.asarray(targets, dtype=np.float64)

    # The irreducible SLO floor: a request whose SERVICE TIME alone exceeds the
    # latency target violates under any control policy, because end-to-end
    # latency >= service time.  Reported per episode so an SLO number is read
    # against its floor rather than against 1.0.
    #
    # Only defined for the `request_service` profile, where `Request.demand` is
    # service SECONDS and the target is a latency in seconds. Under
    # `llm_serving`, demand is output TOKENS and the SLO profile is
    # (ttft_p95_s, tbt_p95_s, ttft_hard_s), so no comparable one-request floor
    # exists; the field is NaN there rather than a wrong number.
    slo_cfg = env.unwrapped.cfg.f("slo")
    lat_target = slo_cfg.get("latency_p95_s") if isinstance(slo_cfg, dict) else None
    if lat_target is None:
        n_infeasible = -1
    else:
        n_infeasible = sum(1 for r in reqs if float(r.demand) > float(lat_target))

    row: Dict[str, Any] = {
        "episode": int(episode),
        "work_seed": int(work_seed),
        "n_control_steps": int(step_i),
        "return": float(ec["return"]),
        "cost_total": float(ec["cost_total"]),
        "cost_slo": float(ec["cost_slo"]),
        "cost_resource": float(ec["cost_resource"]),
        "cost_churn": float(ec["cost_churn"]),
        "cost_slo_drain": float(ec["cost_slo_drain"]),
        "cost_resource_drain": float(ec["cost_resource_drain"]),
        "cost_slo_unfinished": float(ec["cost_slo_unfinished"]),
        "replica_seconds": float(summary.get("replica_seconds", float("nan"))),
        "replica_seconds_drain": float(drain.get("replica_seconds", 0.0)),
        "n_requests": int(len(reqs)),
        "n_done": int(n_done),
        "slo_violations": int(ec["slo_violations_total"]),
        "n_infeasible_by_service_time": int(n_infeasible),
        "unfinished_requests": int(drain.get("unfinished_requests", 0)),
        "truncated_drain": bool(drain.get("truncated_drain", False)),
        "churn_replicas": float(churn_total),
        "scale_events": int(scale_events),
        "target_mean": float(targets_arr.mean()) if len(targets_arr) else float("nan"),
        "target_max": float(targets_arr.max()) if len(targets_arr) else float("nan"),
        "target_min": float(targets_arr.min()) if len(targets_arr) else float("nan"),
        "latency_mean_s": float(np.mean(lat)) if lat else float("nan"),
        "latency_p95_s": _percentile(lat, 95),
        "latency_p99_s": _percentile(lat, 99),
        "queue_delay_p95_s": _percentile(qd, 95),
        "n_actions_clamped": int(n_clamped),
        "n_obs_saturated_steps": n_sat_steps,
        "wall_s": round(time.perf_counter() - t_wall, 4),
    }
    row["slo_attainment"] = (1.0 - row["slo_violations"] / row["n_requests"]
                             if row["n_requests"] else float("nan"))
    row["slo_attainment_feasible_floor"] = (
        1.0 - row["n_infeasible_by_service_time"] / row["n_requests"]
        if row["n_requests"] and n_infeasible >= 0 else float("nan"))
    # calibration stamp: every row, always
    row = env.unwrapped.cfg.stamp(row)
    row["task_id"] = ec["task_id"]
    row["split"] = ec["split"]
    row["action_mode"] = ec["action_mode"]
    return row


def _policy_fn(policy) -> Callable[[Any], int]:
    if hasattr(policy, "act"):
        return lambda obs: int(policy.act(obs))
    if hasattr(policy, "predict"):
        def f(obs):
            a, _ = policy.predict(obs, deterministic=True)
            return int(np.asarray(a).reshape(-1)[0])
        return f
    if callable(policy):
        return lambda obs: int(policy(obs))
    raise TypeError(f"policy {policy!r} has no act/predict and is not callable")


# ---------------------------------------------------------------------------
# the tuning objective
# ---------------------------------------------------------------------------
def objective(rows: Sequence[Dict[str, Any]]) -> float:
    """The single selection objective, identical for every method.

    It is the **environment's own reward**, mean episode cost (lower is better).
    Nothing is added and nothing is reweighted: `GYM_API.md` section 7.1 is
    explicit that a baseline tuned against a different objective than the one the
    benchmark scores -- in particular one that ignores the `churn` shaping term --
    will look worse for a reason that is not about control quality.  So every
    method, analytic and learned, is selected on the benchmark objective it will
    be scored on.
    """
    if not rows:
        return float("inf")
    return float(np.mean([r["cost_total"] for r in rows]))
