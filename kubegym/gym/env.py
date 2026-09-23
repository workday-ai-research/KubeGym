"""`KubeGymEnv`: the Gymnasium face of the KubeGym core.

WHAT THIS LAYER IS AND IS NOT
----------------------------
It is a thin, auditable adapter.  One `step()` is exactly one
`Simulator.advance_control_interval()`; the action is exactly one
`set_target_replicas()`; the observation is exactly a normalised
`ObservationSpec.build(sim.hist)` plus one element for episode progress; the
reward is exactly `-CostModel` applied to that interval.  No physics, no hidden
state, no shaping beyond the churn term the cost model declares.

It is NOT a place where the environment gets easier.  Two properties are
load-bearing and every change to this file must preserve them:

1.  **The policy sees no more than a hand-written controller.**  Observation
    features are keys of `kubegym.core.state.FEATURES`, each derived only from a
    `ClusterState`; normalisation uses a-priori constants only (see
    `kubegym.gym.obs`).  The one non-`FEATURES` element is episode progress
    `t / horizon`, which is fair because the core's controller interface is
    `decide(hist, t)` -- an analytic baseline is handed `t` too.
2.  **The episode is fresh.**  `Simulator.reset` is the only episode start and
    the only place requests come from, and the core refuses to reuse request
    objects.  This env never caches a request list, never reuses a `Simulator`
    across configs, and constructs its own `WorkloadSource` instance so two envs
    cannot share one.

EPISODE STRUCTURE
-----------------
    reset()                      -> t=0 scrape, 1 entry in sim.hist
    step() x n_control_steps     -> one control interval each
    at the last step             -> post-horizon drain, then the episode closes

`n_control_steps = horizon_s / control_interval_s` and the division must be
exact.  At the last step the env runs `Simulator.drain_all`, exactly as the
core's own `run_controller` does, so an episode's conservation quantities
(`episode_summary`) mean the same thing here as they do for a scripted
controller.

`terminated` vs `truncated`.  Reaching the horizon sets `terminated=True`: the
workload is exhausted, the drain has closed the episode out, and there is no
continuation to bootstrap into.  `truncated=True` is reserved for the one case
where the episode genuinely did not finish -- the post-horizon drain hit its cap
with work still in flight, which the core reports as `truncated_drain`.  With
the default cap of `10 x horizon` that does not happen on the shipped tasks, but
when it does the distinction is real information and not a formality.

WHY THE POST-HORIZON DRAIN IS CHARGED TO THE LAST REWARD
--------------------------------------------------------
Work that is still in the system at the horizon keeps costing replica-seconds
and keeps missing its SLO during the drain.  If that cost were dropped, the
optimal end-of-episode policy would be to scale to `n_replicas_min` a few steps
early and push the backlog into an unpriced window -- a boundary artefact that
an RL agent finds quickly and an analytic controller never looks for, i.e. an
artefact that would flatter RL specifically.  So the drain's replica-seconds,
its completed-request SLO violations, and any request left unfinished are
charged to the final step, as three separately reported terms.  This is delayed
reward, not lookahead: the agent observes nothing about the drain before acting.
`include_drain_in_reward=False` restores the unpriced-tail behaviour for anyone
who wants to measure the size of the artefact; the canonical tasks set True.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from ..core.engine import Simulator
from ..core.state import FEATURES, ObservationSpec
from ..provenance import ProvenancedConfig
from .cost import TERMS, CostModel, CostWeights
from .obs import ObsNorm, normalise
from .tasks import TaskSpec, get_task, task_action_mode

ACTION_MODES = ("delta", "absolute")


class KubeGymEnv(gymnasium.Env):
    """Gymnasium environment over one KubeGym benchmark task.

    Construct it through `gymnasium.make("KubeGym/<Task>-<Mode>-v0")`, or
    directly for a task you have built but not registered:

        from kubegym.gym import KubeGymEnv, get_task
        env = KubeGymEnv(get_task("LLMServing-S1"), action_mode="delta", split="dev")

    Parameters
    ----------
    task
        A `TaskSpec`, or the name / Gymnasium id of a registered task.
    action_mode
        `"delta"`: `Discrete(len(delta_set))` over increments applied to the
        CURRENT target (default `(-2,-1,0,1,2)`).  `"absolute"`:
        `Discrete(k_max - k_min + 1)`, action `a` requests `k_min + a`.
        Both are supported everywhere; every registered task exists in both.
    split
        Which work-seed set `reset` may draw from: `"train"`, `"dev"`, `"test"`.
    allow_hypothetical_replicas
        Off by default.  On, the action space may request targets above the
        config's `n_replicas_max` up to `hypothetical_replicas_max`, the core
        sets `hypothetical=True` on the pool, and every result row is tainted.
        For `llm_serving_l40s.json` the ceiling of 4 is a MEASURED physical limit
        of the testbed, so this is not a knob to reach for; turning it on emits a
        warning at construction and sets `info["hypothetical_replicas_allowed"]`.
    include_drain_in_reward
        Defaults to the task's canonical setting.  See the module docstring.
    """

    metadata: Dict[str, Any] = {"render_modes": [], "kubegym_version": "0.1.0"}

    def __init__(self, task: Any, *, action_mode: str = "delta", split: str = "train",
                 allow_hypothetical_replicas: bool = False,
                 hypothetical_replicas_max: Optional[int] = None,
                 include_drain_in_reward: Optional[bool] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        spec = task if isinstance(task, TaskSpec) else get_task(str(task))
        if action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}")
        if split not in spec.splits():
            raise KeyError(f"task {spec.name!r} has no split {split!r}; have {spec.splits()}")
        if render_mode is not None:
            raise ValueError(
                "KubeGymEnv has no render modes: an autoscaling episode has no frame to draw. "
                "Plot info['cost'] and info['n_ready'] instead.")

        self.task_spec = spec
        self.task_name = spec.name
        self.task_id = spec.gym_id(action_mode)
        self.action_mode = action_mode
        self.split = split
        self.render_mode = None

        # -- config and physics ---------------------------------------
        self.cfg: ProvenancedConfig = spec.load_config()
        self.k_min = int(self.cfg.f("n_replicas_min"))
        self.k_max_physical = int(self.cfg.f("n_replicas_max"))
        self.control_interval_s = float(self.cfg.f("control_interval_s"))
        self.n_control_steps = spec.n_control_steps()
        self.horizon_s = float(spec.horizon_s)

        self.allow_hypothetical_replicas = bool(allow_hypothetical_replicas)
        if self.allow_hypothetical_replicas:
            hmax = int(hypothetical_replicas_max if hypothetical_replicas_max is not None
                       else self.k_max_physical + 4)
            if hmax <= self.k_max_physical:
                raise ValueError(
                    f"hypothetical_replicas_max={hmax} is not above n_replicas_max="
                    f"{self.k_max_physical}; nothing hypothetical would be reachable")
            warnings.warn(
                f"{self.task_id}: allow_hypothetical_replicas=True raises the action ceiling to "
                f"{hmax} replicas, above the config's n_replicas_max={self.k_max_physical}. For "
                f"{self.cfg.path} that ceiling is a MEASURED physical limit of the testbed, so "
                "every row produced by this env is now tainted: the core sets hypothetical=True "
                "on the pool and info['hypothetical_replica_count_used'] reports it. Results are "
                "not comparable with canonical-task results.",
                RuntimeWarning, stacklevel=2)
            self.k_max = hmax
        else:
            self.k_max = self.k_max_physical

        self.service_model = spec.service_model_factory(self.cfg)
        # One workload source per env instance: two envs must never share the
        # build counter or the record cache of one source.
        self.workload = spec.workload_factory()
        self.sim = Simulator(self.service_model, self.workload, self.cfg,
                            k0=spec.k0,
                            allow_hypothetical_replicas=self.allow_hypothetical_replicas)

        # -- objective -------------------------------------------------
        self.include_drain_in_reward = bool(
            spec.include_drain_in_reward if include_drain_in_reward is None
            else include_drain_in_reward)
        self.cost_model = CostModel(self.cfg, spec.weights, slo_profile=spec.slo_profile)
        self.drain_cap_s = float(spec.horizon_s * spec.drain_cap_multiple)

        # -- observation ----------------------------------------------
        self.obs_spec = ObservationSpec(features=tuple(spec.obs_features),
                                        history=int(spec.obs_history))
        self.obs_norm: ObsNorm = spec.obs_norm
        self._divisors = self.obs_norm.divisor_vector(spec.obs_features, spec.obs_history)
        self.obs_names: List[str] = list(self.obs_spec.names()) + ["episode_progress"]
        self.observation_space = spaces.Box(
            low=0.0, high=float(self.obs_norm.clip),
            shape=(len(self.obs_names),), dtype=np.float32)

        # -- action ----------------------------------------------------
        self.delta_set: Tuple[int, ...] = tuple(int(d) for d in spec.delta_set)
        if action_mode == "delta":
            self.action_space = spaces.Discrete(len(self.delta_set))
            self.action_meanings = [f"target {d:+d}" for d in self.delta_set]
        else:
            self.action_space = spaces.Discrete(self.k_max - self.k_min + 1)
            self.action_meanings = [f"target = {self.k_min + a}"
                                    for a in range(self.k_max - self.k_min + 1)]

        # -- episode state --------------------------------------------
        self._episode: int = -1
        self._work_seed: int = -1
        self._step_i: int = 0
        self._scored: Set[int] = set()
        self._t_prev: float = 0.0
        self._rs_prev: float = 0.0
        self._cum: Dict[str, float] = {}
        self._n_viol_cum: int = 0
        self._closed_episode: bool = True

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def observe_from_history(self, hist: Sequence[Any], step_i: int) -> np.ndarray:
        """The observation as a pure function of the scrape history and step index.

        Public because it is what makes the no-privileged-information claim
        checkable: `kubegym/tests/test_gym_observation.py` calls this with a
        history whose requests have been mutated and asserts the result does not
        move.  Nothing else in the env contributes to the observation.
        """
        raw = self.obs_spec.build(hist)
        vec = normalise(raw, self._divisors, self.obs_norm.clip)
        progress = np.float32(min(1.0, max(0.0, step_i / float(self.n_control_steps))))
        return np.concatenate([vec, np.asarray([progress], dtype=np.float32)])

    def _observe(self) -> np.ndarray:
        return self.observe_from_history(self.sim.hist, self._step_i)

    # ------------------------------------------------------------------
    # action
    # ------------------------------------------------------------------
    def _action_to_target(self, action: int) -> int:
        """Map an action to a requested replica target (pre-clamp)."""
        if self.action_mode == "delta":
            return int(self.sim.target) + int(self.delta_set[int(action)])
        return self.k_min + int(action)

    # ------------------------------------------------------------------
    # episode lifecycle
    # ------------------------------------------------------------------
    def _select_pair(self, seed: Optional[int], options: Optional[Dict[str, Any]]
                     ) -> Tuple[int, int]:
        """Choose `(episode, work_seed)` for this episode.

        Precedence: explicit `options`, then `seed`, then a draw from
        `self.np_random`.  `reset(seed=s)` is a total, deterministic function of
        `s` -- `pairs[s % len(pairs)]` -- which is what makes byte-identical
        replay possible through the Gym API alone.
        """
        pairs = self.task_spec.pairs(self.split)
        opts = dict(options or {})
        ep = opts.pop("episode", None)
        ws = opts.pop("work_seed", None)
        if opts:
            raise KeyError(f"unknown reset options {sorted(opts)}; supported: "
                           "'episode', 'work_seed'")
        if ep is not None or ws is not None:
            if ep is None or ws is None:
                raise ValueError(
                    "pass both 'episode' and 'work_seed' in reset options, or neither: a "
                    "half-specified episode would silently mix an explicit choice with a "
                    "drawn one.")
            ep, ws = int(ep), int(ws)
            allowed = set(self.task_spec.split_work_seeds[self.split])
            if ws not in allowed:
                raise ValueError(
                    f"work_seed {ws} is not in split {self.split!r} of task "
                    f"{self.task_spec.name!r}. Using a seed from another split would put a test "
                    "episode in a training run under another name; construct the env with "
                    f"split=... instead. Allowed: {sorted(allowed)}")
            if not (0 <= ep < self.task_spec.n_episodes):
                raise ValueError(f"episode {ep} out of range for task {self.task_spec.name!r} "
                                 f"(n_episodes={self.task_spec.n_episodes})")
            return ep, ws
        if seed is not None:
            return pairs[int(seed) % len(pairs)]
        return pairs[int(self.np_random.integers(len(pairs)))]

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        episode, work_seed = self._select_pair(seed, options)
        self._episode = int(episode)
        self._work_seed = int(work_seed)
        # The core rebuilds every Request here; it raises if the workload source
        # hands back an object it already issued. That guard is the reason
        # thousands of resets do not silently degrade into no-op episodes.
        self.sim.reset(episode=self._episode, seed=self._work_seed, k0=self.task_spec.k0)
        self._step_i = 0
        self._scored = set()
        self._t_prev = float(self.sim.t)
        self._rs_prev = float(self.sim.total_replica_seconds(self.sim.t))
        self._cum = {k: 0.0 for k in TERMS}
        self._cum["total"] = 0.0
        self._n_viol_cum = 0
        self._closed_episode = False
        obs = self._observe()
        return obs, self._info_common(action=None, target_before=self.sim.target,
                                      target_requested=self.sim.target,
                                      cost=None, slo=None)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._closed_episode:
            raise RuntimeError(
                "step() called after the episode ended. Call reset() first; the core's "
                "requests are consumed and stepping on would mutate a finished episode.")
        a = int(action)
        if not self.action_space.contains(a):
            raise ValueError(f"action {action!r} is not in {self.action_space}")

        target_before = int(self.sim.target)
        target_requested = self._action_to_target(a)
        # `set_target_replicas` clamps to [n_replicas_min, n_replicas_max] inside
        # the core (or to the hypothetical ceiling if that was enabled). We read
        # the post-clamp value back rather than tracking our own, so the env can
        # never disagree with the pool about what was actuated.
        self.sim.set_target_replicas(target_requested)
        target_after = int(self.sim.target)
        churn = abs(target_after - target_before)

        t_prev = float(self.sim.t)
        rs_prev = float(self.sim.total_replica_seconds(t_prev))
        self.sim.advance_control_interval()
        t_now = float(self.sim.t)
        rs_now = float(self.sim.total_replica_seconds(t_now))
        replica_seconds_step = rs_now - rs_prev

        n_viol, newly, reasons = self.cost_model.score_window(
            self.sim.requests, t_prev, t_now, self._scored)
        self._scored.update(newly)
        self._n_viol_cum += n_viol
        self._step_i += 1

        at_horizon = self._step_i >= self.n_control_steps
        drain: Dict[str, Any] = {}
        n_viol_drain = 0
        rs_drain = 0.0
        unfinished = 0
        truncated_drain = False
        if at_horizon:
            cap = t_now + self.drain_cap_s
            truncated_drain = bool(self.sim.drain_all(cap))
            t_drained = float(self.sim.t)
            rs_drain = float(self.sim.total_replica_seconds(t_drained)) - rs_now
            n_viol_drain, newly_d, reasons_d = self.cost_model.score_window(
                self.sim.requests, t_now, t_drained, self._scored, score_pending=False)
            self._scored.update(newly_d)
            unfinished = self.cost_model.count_unfinished(self.sim.requests, self._scored)
            drain = {"t_end": t_drained, "duration_s": t_drained - t_now,
                     "replica_seconds": rs_drain, "slo_violations": n_viol_drain,
                     "slo_reasons": reasons_d, "unfinished_requests": unfinished,
                     "truncated_drain": truncated_drain, "cap_s": cap,
                     "charged_to_reward": self.include_drain_in_reward}

        charge_drain = at_horizon and self.include_drain_in_reward
        cost = self.cost_model.decompose(
            slo_violations=n_viol,
            replica_seconds=replica_seconds_step,
            churn_replicas=churn,
            slo_violations_drain=(n_viol_drain if charge_drain else 0),
            replica_seconds_drain=(rs_drain if charge_drain else 0.0),
            unfinished_requests=(unfinished if charge_drain else 0),
        )
        reward = -float(cost["total"])
        for k in TERMS:
            self._cum[k] += float(cost[k])
        self._cum["total"] += float(cost["total"])

        terminated = bool(at_horizon and not truncated_drain)
        truncated = bool(at_horizon and truncated_drain)
        if at_horizon:
            self._closed_episode = True

        obs = self._observe()
        info = self._info_common(action=a, target_before=target_before,
                                 target_requested=target_requested,
                                 cost=cost,
                                 slo={"violations_step": n_viol,
                                      "reasons_step": reasons,
                                      "violations_cumulative": self._n_viol_cum,
                                      "targets": dict(self.cost_model.slo),
                                      "profile": self.cost_model.slo_profile})
        info["replica_seconds_step"] = replica_seconds_step
        if at_horizon:
            info["post_horizon_drain"] = drain
            info["episode_summary"] = self.sim.episode_summary(self.horizon_s)
            info["episode_cost"] = self._episode_cost_row()
        self._t_prev = t_now
        self._rs_prev = rs_now
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # info assembly
    # ------------------------------------------------------------------
    def _info_common(self, *, action: Optional[int], target_before: int,
                     target_requested: int, cost: Optional[Dict[str, float]],
                     slo: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        st = self.sim.hist[-1]
        info: Dict[str, Any] = {
            # identity: enough to re-run this episode from scratch
            "task": self.task_name,
            "task_id": self.task_id,
            "action_mode": self.action_mode,
            "split": self.split,
            # NOT "episode": Gymnasium's RecordEpisodeStatistics and SB3's
            # Monitor both reserve info["episode"] for their own statistics DICT,
            # and SB3's logger then indexes it as info["episode"]["r"]. Publishing
            # the corpus episode INDEX under that key makes the first logger dump
            # raise TypeError, so PPO/DQN cannot be trained through the standard
            # stable_baselines3 helpers at all. The index is ours to name; the
            # key is not.
            "episode_index": self._episode,
            "work_seed": self._work_seed,
            "config": self.cfg.provenance()["config_path"],
            # the claims gate, on every single row
            "calibrated": bool(self.cfg.calibrated),
            "uncalibrated_required_fields": list(self.cfg.uncalibrated_required),
            # actuation
            "t": float(self.sim.t),
            "step": int(self._step_i),
            "n_control_steps": int(self.n_control_steps),
            "action": action,
            "target_before": int(target_before),
            "target_requested": int(target_requested),
            "target": int(self.sim.target),
            "action_clamped": bool(target_requested != int(self.sim.target)),
            "n_replicas_min": int(self.k_min),
            "n_replicas_max": int(self.k_max),
            "n_replicas_max_physical": int(self.k_max_physical),
            "hypothetical_replicas_allowed": bool(self.allow_hypothetical_replicas),
            "hypothetical_replica_count_used": bool(self.sim.pool.hypothetical),
            # cluster state, as scraped
            "n_ready": int(st.n_ready),
            "n_booting": int(st.n_booting),
            "n_draining": int(st.n_draining),
            "n_replicas_effective": int(st.n_replicas_effective),
            "arrivals_since_last": int(st.arrivals_since_last),
            "replica_seconds_total": float(self.sim.total_replica_seconds(self.sim.t)),
        }
        if cost is not None:
            info["cost"] = dict(cost)
            info["cost"]["weights"] = self.cost_model.weights.to_dict()
            info["cost"]["units"] = {
                "slo": "reward per violating request",
                "resource": "reward per replica-second",
                "churn": "reward per replica added or removed",
            }
            info["cost"]["sign_convention"] = "reward = -cost['total']"
            info["cost_cumulative"] = dict(self._cum)
        if slo is not None:
            info["slo"] = slo
        return info

    def _episode_cost_row(self) -> Dict[str, Any]:
        """Episode-level cost row, provenance-stamped.

        `cfg.stamp` is applied here for the same reason the core applies it to
        `episode_summary`: a return is a result, and a `calibrated=False` result
        must not be able to travel without its label.
        """
        row: Dict[str, Any] = {
            "task": self.task_name, "task_id": self.task_id,
            "action_mode": self.action_mode, "split": self.split,
            "episode": self._episode, "work_seed": self._work_seed,
            "n_control_steps": int(self.n_control_steps),
            "return": -float(self._cum["total"]),
            "cost_total": float(self._cum["total"]),
            "slo_violations_total": int(self._n_viol_cum),
            "include_drain_in_reward": self.include_drain_in_reward,
        }
        for k in TERMS:
            row[f"cost_{k}"] = float(self._cum[k])
        row.update({f"weight_{k}": v for k, v in self.cost_model.weights.to_dict().items()})
        return self.cfg.stamp(row)

    # ------------------------------------------------------------------
    # documentation surface
    # ------------------------------------------------------------------
    def observation_description(self) -> List[Dict[str, Any]]:
        """Element-by-element description of the observation vector."""
        rows = self.obs_norm.describe(self.task_spec.obs_features, self.task_spec.obs_history)
        for r in rows:
            r["source"] = "kubegym.core.state.FEATURES[%r](ClusterState)" % r["feature"]
        rows.append({
            "name": "episode_progress", "feature": "(env)", "lag_intervals": 0,
            "scale_kind": "unit", "divisor": float(self.n_control_steps),
            "source": "control steps taken / n_control_steps; the core's own controller "
                      "interface decide(hist, t) is handed t, so this is not privileged",
        })
        return rows

    def spec_card(self) -> Dict[str, Any]:
        """Everything an env card needs, as data."""
        cfg = self.cfg
        return {
            "id": self.task_id,
            "task": self.task_name,
            "action_mode": self.action_mode,
            "action_space": str(self.action_space),
            "action_meanings": list(self.action_meanings),
            "observation_space": str(self.observation_space),
            "observation_dim": int(self.observation_space.shape[0]),
            "observation_elements": self.observation_description(),
            "observation_norm": self.obs_norm.to_dict(),
            "service_model": self.task_spec.service_model_name,
            "service_model_provenance": self.service_model.provenance(),
            "workload": self.task_spec.workload_name,
            "workload_description": self.task_spec.workload_description,
            "workload_manifest": self.workload.manifest(),
            "config": cfg.provenance()["config_path"],
            "control_interval_s": self.control_interval_s,
            "horizon_s": self.horizon_s,
            "n_control_steps": self.n_control_steps,
            "n_episodes": self.task_spec.n_episodes,
            "k0": self.task_spec.k0,
            "n_replicas_min": self.k_min,
            "n_replicas_max": self.k_max,
            "n_replicas_max_is_measured_physical_limit":
                bool(cfg.is_calibrated("n_replicas_max")),
            "splits": {s: list(self.task_spec.split_work_seeds[s]) for s in self.task_spec.splits()},
            "split_pairs": {s: len(self.task_spec.pairs(s)) for s in self.task_spec.splits()},
            "drain_cap_s": self.drain_cap_s,
            "include_drain_in_reward": self.include_drain_in_reward,
            "cost": self.cost_model.provenance(),
            "calibrated": bool(cfg.calibrated),
            "uncalibrated_required_fields": list(cfg.uncalibrated_required),
            "calibration_note": cfg.calibration_note,
            "flagged_optimistic_fields": cfg.provenance()["flagged_optimistic_fields"],
            "notes": list(self.task_spec.notes),
            "provenance": self.sim.provenance(),
        }

    # ------------------------------------------------------------------
    def render(self):
        return None

    def close(self) -> None:
        # Nothing external is held: no files, no sockets, no subprocess. The
        # simulator and its requests are plain Python objects and are dropped
        # with the env. `close` exists to satisfy the Gymnasium contract and to
        # make the episode unsteppable.
        self._closed_episode = True

    def __repr__(self) -> str:
        return (f"KubeGymEnv({self.task_id!r}, split={self.split!r}, "
                f"obs_dim={self.observation_space.shape[0]}, "
                f"actions={self.action_space.n}, calibrated={self.cfg.calibrated})")


# ---------------------------------------------------------------------------
# Gymnasium entry point
# ---------------------------------------------------------------------------
def make_task_env(task: str, action_mode: str = "delta", **kwargs: Any) -> KubeGymEnv:
    """Entry point used by `gymnasium.register`.  Also fine to call directly."""
    return KubeGymEnv(task, action_mode=action_mode, **kwargs)
