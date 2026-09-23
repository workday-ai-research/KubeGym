"""Named benchmark tasks: what makes two papers' numbers comparable.

A `TaskSpec` freezes every choice that changes the numbers: service model,
config file, workload, horizon, initial replica count, observation features and
their scales, the canonical cost weights, the SLO profile, and the
train/dev/test seed separation.  `KubeGymEnv` reads a `TaskSpec` and nothing
else, so "the task" is a single auditable object rather than a pile of
constructor arguments.

ID SCHEME
---------
    KubeGym/<ServiceModel><Workload>-<ActionMode>-v0

`<ServiceModel><Workload>` is the task name (e.g. `LLMServing-S1`),
`<ActionMode>` is `Delta` or `Abs`, and `-v0` is the *task* version.  Bump it
whenever anything in the `TaskSpec` changes the numbers -- a new weight, a new
observation feature, a different horizon.  Never mutate a registered `-v0` in
place: a reader who compares their `-v0` result against a published `-v0`
result must be comparing the same task.  The split (`train` / `dev` / `test`) is
a constructor keyword, not part of the id, because it selects seeds rather than
redefining the task; it is echoed in `info["split"]` and in the env card.

WHICH WORKLOADS THIS MODULE MAY USE
-----------------------------------
Only what the core ships: the two reference traces in `kubegym/tests/data/`
(`S1_dev.jsonl`, `S3_dev.jsonl`) and `poisson_arrivals`, via the core's own
`LLMTraceSource` / `RequestServiceSource`.  Those traces are TEST FIXTURES, not
the benchmark's workload corpus -- their own README says so, and they are 300 s
long, which is 20 control steps.  They are here so the RL layer has something
registered and testable on day one.

The workload corpus is a separate artifact built by a separate track.  It
registers its tasks by calling `register_task` (or `register_workload_task`)
with its own workload factory; this module never imports it, and it never
imports this module's internals.  `register_task` is the whole contract between
the two.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import config_path
from ..provenance import ProvenancedConfig
from .cost import CostWeights
from .obs import ObsNorm

#: Gymnasium entry point for every registered task.  A string so that
#: `gymnasium.register` does not import `kubegym.gym.env` at registration time.
ENTRY_POINT = "kubegym.gym.env:make_task_env"

#: Action-mode name -> id suffix.
ACTION_MODES: Dict[str, str] = {"delta": "Delta", "absolute": "Abs"}

#: Work seeds per split.  DISJOINT BY CONSTRUCTION -- that disjointness is the
#: train/dev/test separation, and `test_gym_registry.py` asserts it for every
#: registered task rather than trusting this comment.
#:
#: `seed` in `reset(seed=...)` indexes into the split's (episode, work_seed)
#: pairs; it is not itself the work seed.  So a policy trained with
#: `split="train"` has never seen a `dev` or `test` work realisation, whatever
#: seeds the training loop passed.
DEFAULT_SPLIT_WORK_SEEDS: Dict[str, Tuple[int, ...]] = {
    "train": tuple(range(0, 64)),
    "dev": tuple(range(1000, 1016)),
    "test": tuple(range(2000, 2016)),
}


def fixture_trace_path(name: str) -> str:
    """Absolute path to a reference trace shipped in the core's test fixtures.

    These are fixtures, not the corpus.  Provenance (sha256, request count,
    horizon) is in `kubegym/tests/data/README.md`.
    """
    import kubegym as _kg
    p = os.path.join(os.path.dirname(os.path.abspath(_kg.__file__)), "tests", "data", name)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"reference trace fixture {name!r} not found at {p}. The Gym task registry depends "
            "on the traces shipped in kubegym/tests/data/; if the package was installed without "
            "its package-data, reinstall it.")
    return p


# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    """Every choice that changes a task's numbers, in one frozen object.

    Parameters
    ----------
    name
        Task name without the action-mode suffix or version, e.g.
        `"LLMServing-S1"`.  Appears in every id built from this spec.
    service_model_factory
        `cfg -> ServiceModel`.  Called once per env instance.
    workload_factory
        `() -> WorkloadSource`.  Called once per env instance, so two envs never
        share a source object.  It must return a source whose `build()` is a
        pure function of `(episode, seed)` -- the core enforces the freshness
        half of that, not the purity half.
    config_file
        File name of a config shipped in `kubegym/configs/`, or an absolute
        path.  A task never carries physical constants of its own: they all come
        from the config, so the claims gate covers them.
    n_episodes
        How many distinct `episode` indices the workload source defines.  For a
        multi-file trace source this is the number of files.
    horizon_s
        Episode horizon in seconds.  Divided by the config's
        `control_interval_s` to get the number of control steps; the division
        must be exact.
    obs_features
        Keys of `kubegym.core.state.FEATURES`, in observation order.
    delta_set
        The increments of the `delta` action mode, e.g. `(-2,-1,0,1,2)`.  Must
        contain 0, so "hold" is always available.
    drain_cap_multiple
        The post-horizon drain cap is `horizon_s * drain_cap_multiple`.
    include_drain_in_reward
        Charge post-horizon drain cost to the final step's reward.  Default True;
        see `KubeGymEnv` for why the alternative is gameable.
    notes
        Task-specific caveats, copied verbatim into the env card.
    """

    name: str
    service_model_name: str
    config_file: str
    workload_name: str
    workload_description: str
    workload_factory: Callable[[], Any]
    service_model_factory: Callable[[ProvenancedConfig], Any]
    n_episodes: int
    horizon_s: float
    k0: int
    obs_features: Tuple[str, ...]
    obs_history: int
    obs_norm: ObsNorm
    weights: CostWeights
    slo_profile: str
    delta_set: Tuple[int, ...] = (-2, -1, 0, 1, 2)
    drain_cap_multiple: float = 10.0
    include_drain_in_reward: bool = True
    split_work_seeds: Mapping[str, Tuple[int, ...]] = field(
        default_factory=lambda: dict(DEFAULT_SPLIT_WORK_SEEDS))
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if 0 not in self.delta_set:
            raise ValueError(
                f"task {self.name!r}: delta_set {self.delta_set} does not contain 0, so the "
                "policy cannot hold the current replica count. Every autoscaling baseline "
                "needs a no-op action.")
        if self.n_episodes < 1:
            raise ValueError(f"task {self.name!r}: n_episodes must be >= 1")
        if self.obs_history < 1:
            raise ValueError(f"task {self.name!r}: obs_history must be >= 1")
        splits = set(self.split_work_seeds)
        if not splits:
            raise ValueError(f"task {self.name!r}: no splits declared")
        seen: Dict[int, str] = {}
        for split in sorted(splits):
            for s in self.split_work_seeds[split]:
                if s in seen and seen[s] != split:
                    raise ValueError(
                        f"task {self.name!r}: work seed {s} appears in both {seen[s]!r} and "
                        f"{split!r}. Splits must be disjoint in work seed, or a 'test' episode "
                        "is a training episode under another name.")
                seen[s] = split

    # -- derived -------------------------------------------------------
    def config_abspath(self) -> str:
        if os.path.isabs(self.config_file):
            return self.config_file
        return config_path(self.config_file)

    def load_config(self) -> ProvenancedConfig:
        return ProvenancedConfig.load(self.config_abspath())

    def control_interval_s(self) -> float:
        return float(self.load_config().f("control_interval_s"))

    def n_control_steps(self) -> int:
        dt = self.control_interval_s()
        n = self.horizon_s / dt
        n_int = int(round(n))
        if abs(n - n_int) > 1e-9:
            raise ValueError(
                f"task {self.name!r}: horizon_s={self.horizon_s} is not an integer multiple of "
                f"control_interval_s={dt}. A partial final control interval would make the "
                "episode length depend on rounding.")
        return n_int

    def splits(self) -> List[str]:
        return sorted(self.split_work_seeds)

    def pairs(self, split: str) -> Tuple[Tuple[int, int], ...]:
        """The `(episode, work_seed)` pairs a split may draw from.

        `reset(seed=s)` selects `pairs[s % len(pairs)]`, so the mapping from a
        Gymnasium seed to a workload realisation is total and deterministic.
        """
        if split not in self.split_work_seeds:
            raise KeyError(f"task {self.name!r} has no split {split!r}; have {self.splits()}")
        return tuple((e, int(s)) for s in self.split_work_seeds[split]
                     for e in range(self.n_episodes))

    def obs_dim(self) -> int:
        """Feature block plus the one episode-progress element appended by the env."""
        return len(self.obs_features) * self.obs_history + 1

    def gym_id(self, action_mode: str) -> str:
        if action_mode not in ACTION_MODES:
            raise KeyError(f"unknown action_mode {action_mode!r}; have {sorted(ACTION_MODES)}")
        return f"KubeGym/{self.name}-{ACTION_MODES[action_mode]}-v0"

    def gym_ids(self) -> Dict[str, str]:
        return {m: self.gym_id(m) for m in ACTION_MODES}


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
_TASKS: Dict[str, TaskSpec] = {}
_IDS: Dict[str, Tuple[str, str]] = {}          # gym id -> (task name, action mode)


def register_task(spec: TaskSpec, *, action_modes: Sequence[str] = tuple(ACTION_MODES),
                  override: bool = False) -> List[str]:
    """Register a `TaskSpec` with Gymnasium under one id per action mode.

    THIS IS THE PUBLIC EXTENSION POINT.  A downstream paper, or the workload
    corpus track, adds tasks by calling this with its own `workload_factory`;
    nothing in `kubegym.gym` needs to import the caller's module, and the caller
    needs nothing from `kubegym.gym` beyond `TaskSpec`, `ObsNorm`, `CostWeights`
    and this function.

    Returns the ids registered.  Re-registering an existing name raises unless
    `override=True`, because silently replacing a task would make two runs of
    the same id incomparable.
    """
    import gymnasium

    if spec.name in _TASKS and not override:
        raise ValueError(
            f"task {spec.name!r} is already registered. Pass override=True if you really mean "
            "to replace it, but prefer a new name or a version bump: two runs of the same id "
            "must be the same task.")
    modes = list(action_modes)
    bad = [m for m in modes if m not in ACTION_MODES]
    if bad:
        raise KeyError(f"unknown action modes {bad}; have {sorted(ACTION_MODES)}")
    spec.n_control_steps()          # fail now, not at first make()
    ids: List[str] = []
    for mode in modes:
        gid = spec.gym_id(mode)
        gymnasium.register(
            id=gid,
            entry_point=ENTRY_POINT,
            kwargs={"task": spec.name, "action_mode": mode},
            # The env implements its own horizon and reports `terminated` at it,
            # so no TimeLimit wrapper: a second, wrapper-level horizon would be
            # a way for the episode length to disagree with the task spec.
            max_episode_steps=None,
            disable_env_checker=False,
        )
        _IDS[gid] = (spec.name, mode)
        ids.append(gid)
    _TASKS[spec.name] = spec
    return ids


def register_workload_task(*, name: str, workload_factory: Callable[[], Any],
                           n_episodes: int, horizon_s: float,
                           service_model: str = "llm_serving",
                           config_file: Optional[str] = None,
                           workload_name: str = "",
                           workload_description: str = "",
                           weights: Optional[CostWeights] = None,
                           obs_norm: Optional[ObsNorm] = None,
                           k0: int = 1,
                           notes: Sequence[str] = (),
                           action_modes: Sequence[str] = tuple(ACTION_MODES),
                           override: bool = False,
                           **spec_kwargs: Any) -> List[str]:
    """Register a task from a workload factory, reusing a builtin's conventions.

    A convenience over `register_task` for the common case: same service model,
    same observation features and scales, same canonical weights as the builtin
    task of that service model, different workload.  Anything not passed is
    inherited from the reference spec for `service_model` and recorded in the
    env card, so an inherited choice is visible rather than implicit.

    Use `register_task` directly when the task changes the observation, the
    weights or the SLO profile -- those are not conventions to inherit.
    """
    ref = REFERENCE_SPEC.get(service_model)
    if ref is None:
        raise KeyError(f"no reference spec for service_model {service_model!r}; have "
                       f"{sorted(REFERENCE_SPEC)}. Build a TaskSpec and call register_task.")
    spec = TaskSpec(
        name=name,
        service_model_name=ref.service_model_name,
        config_file=config_file or ref.config_file,
        workload_name=workload_name or name,
        workload_description=workload_description or "(no description supplied)",
        workload_factory=workload_factory,
        service_model_factory=ref.service_model_factory,
        n_episodes=int(n_episodes),
        horizon_s=float(horizon_s),
        k0=int(k0),
        obs_features=ref.obs_features,
        obs_history=ref.obs_history,
        obs_norm=obs_norm or ref.obs_norm,
        weights=weights or ref.weights,
        slo_profile=ref.slo_profile,
        delta_set=ref.delta_set,
        drain_cap_multiple=ref.drain_cap_multiple,
        include_drain_in_reward=ref.include_drain_in_reward,
        split_work_seeds=dict(ref.split_work_seeds),
        notes=tuple(notes) + (
            f"Registered via register_workload_task, inheriting observation features, scales, "
            f"cost weights and SLO profile from the reference task {ref.name!r}.",),
        **spec_kwargs,
    )
    return register_task(spec, action_modes=action_modes, override=override)


def get_task(name: str) -> TaskSpec:
    """Look a task up by name, or by full Gymnasium id."""
    if name in _TASKS:
        return _TASKS[name]
    if name in _IDS:
        return _TASKS[_IDS[name][0]]
    raise KeyError(f"no registered task {name!r}; have {sorted(_TASKS)} "
                   f"(ids: {sorted(_IDS)})")


def task_action_mode(gym_id: str) -> str:
    if gym_id not in _IDS:
        raise KeyError(f"no registered id {gym_id!r}; have {sorted(_IDS)}")
    return _IDS[gym_id][1]


def task_names() -> List[str]:
    return sorted(_TASKS)


def task_ids() -> List[str]:
    return sorted(_IDS)


def list_tasks() -> List[Dict[str, Any]]:
    """One row per registered Gymnasium id, with the fields that identify a task.

    Deliberately includes `calibrated`: the flag travels with the task listing,
    not only with a result row, so nobody selects a task without seeing that its
    physics are uncalibrated.
    """
    rows: List[Dict[str, Any]] = []
    for gid in task_ids():
        tname, mode = _IDS[gid]
        spec = _TASKS[tname]
        cfg = spec.load_config()
        rows.append({
            "id": gid,
            "task": tname,
            "action_mode": mode,
            "service_model": spec.service_model_name,
            "workload": spec.workload_name,
            "config": os.path.basename(spec.config_abspath()),
            "horizon_s": spec.horizon_s,
            "control_interval_s": float(cfg.f("control_interval_s")),
            "n_control_steps": spec.n_control_steps(),
            "n_episodes": spec.n_episodes,
            "n_replicas_min": int(cfg.f("n_replicas_min")),
            "n_replicas_max": int(cfg.f("n_replicas_max")),
            "obs_dim": spec.obs_dim(),
            "splits": spec.splits(),
            "weights": spec.weights.to_dict(),
            "calibrated": bool(cfg.calibrated),
            "uncalibrated_required_fields": cfg.uncalibrated_required,
        })
    return rows


# ---------------------------------------------------------------------------
# Builtin tasks
# ---------------------------------------------------------------------------
#: Observation features used by every builtin task.  A subset of
#: `kubegym.core.state.FEATURES`, so the fairness guarantee of INTERFACE.md
#: section 6 -- every feature derives only from a `ClusterState` -- holds by
#: construction.  `target` is included because it is the policy's own actuation
#: state: without it, `delta` actions act on a quantity the policy cannot see.
BUILTIN_FEATURES: Tuple[str, ...] = (
    "n_ready", "n_booting", "n_draining", "target",
    "n_running", "n_waiting", "sat_mean", "sat_max", "arrivals",
)
BUILTIN_HISTORY = 4


def _llm_service_model(cfg: ProvenancedConfig) -> Any:
    from ..models.llm_serving import LLMServingModel
    from ..models.length_model import load_length_model
    # No measured_lengths.json ships with the package, so this is the explicitly
    # labelled PLACEHOLDER length model and `calibrated` stays False. It is not
    # silently upgraded; see kubegym/models/length_model.py.
    return LLMServingModel(cfg, length_model=load_length_model("measured_lengths.json"))


def _llm_trace_source(paths: Sequence[str], name: str, horizon_s: float) -> Callable[[], Any]:
    def factory() -> Any:
        from ..models.llm_serving import LLMTraceSource
        from ..models.length_model import load_length_model
        return LLMTraceSource([str(p) for p in paths],
                              length_model=load_length_model("measured_lengths.json"),
                              name=name, horizon_s=horizon_s)
    return factory


def _rs_service_model(cfg: ProvenancedConfig) -> Any:
    from ..models.request_service import RequestServiceModel
    return RequestServiceModel(cfg)


def _rs_poisson_source(rate_rps: float, horizon_s: float, gen_seed: int,
                       name: str) -> Callable[[], Any]:
    def factory() -> Any:
        from ..models.request_service import RequestServiceSource, poisson_arrivals
        cfg = ProvenancedConfig.load(config_path("request_service_default.json"))
        recs = poisson_arrivals(rate_rps=rate_rps, horizon_s=horizon_s, seed=gen_seed,
                               cls="generic")
        return RequestServiceSource(
            recs, service_time_distribution=cfg.f("rs_service_time_distribution"),
            name=name, horizon_s=horizon_s,
            manifest={"arrival_process": "homogeneous Poisson",
                      "rate_rps": rate_rps,
                      "arrival_generator_seed": gen_seed,
                      "note": "Arrival times are FIXED by arrival_generator_seed, so this task "
                              "has one episode; per-episode variation comes from the work "
                              "realisation (service times drawn at build time from the "
                              "placeholder lognormal)."})
    return factory


#: Canonical cost weights per builtin task.  DESIGN CHOICES; see cost.py.
#:
#: `resource` is set to `1 / (control_interval_s * n_replicas_max)`, so running
#: at the physical ceiling for a whole control interval costs exactly 1.0.  That
#: gives the resource term the same meaning across tasks with different ceilings,
#: which is the only reason to prefer it to a round number.  `slo` and `churn`
#: are round numbers picked so that, at these tasks' scales, a handful of
#: violations or a one-replica move is commensurate with a fraction of the
#: ceiling cost. They are not derived from anything.
_LLM_WEIGHTS = CostWeights(slo=0.10, resource=1.0 / (15.0 * 4), churn=0.05)
_RS_WEIGHTS = CostWeights(slo=0.05, resource=1.0 / (15.0 * 20), churn=0.05)

#: Observation scales.  Every divisor below is an a-priori constant; see obs.py
#: for why nothing episode-derived may appear here.
#:
#: `concurrency_scale = n_replicas_max x 8` for the LLM tasks: batch 8 is the
#: top of the MEASURED decode-throughput range (`decode_step_time_s` was fitted
#: at batch 1/4/8), so it is the largest per-replica concurrency the physics are
#: measured at rather than extrapolated at. That makes the observation read 1.0
#: at the edge of the calibrated region -- a documented coincidence of
#: convenience, not a claim that 8 is a capacity limit.
_LLM_NORM = ObsNorm(replica_scale=4.0, concurrency_scale=4.0 * 8, per_replica_concurrency=8.0,
                    arrivals_scale=16.0, preemptions_scale=100.0, clip=10.0)
_RS_NORM = ObsNorm(replica_scale=20.0, concurrency_scale=20.0 * 4, per_replica_concurrency=4.0,
                   arrivals_scale=64.0, preemptions_scale=100.0, clip=10.0)

_FIXTURE_NOTE = (
    "WORKLOAD IS A TEST FIXTURE, NOT THE BENCHMARK CORPUS. The 300 s reference traces in "
    "kubegym/tests/data/ are the core's test fixtures (their README says so explicitly). At a "
    "15 s control interval an episode is 20 control steps, which is short for RL: credit for a "
    "31.9 s cold start spans two steps, so roughly a tenth of the episode is spent paying for "
    "any single scale-up. Treat these tasks as conformance and smoke-test targets; the workload "
    "corpus track supplies the episodes a learning-curve claim should use.")

_PLACEHOLDER_NOTE = (
    "CALIBRATION: the config is calibrated=false, so no return, reward or cost from this task "
    "is an empirical result. The SLO targets the reward is evaluated against are themselves a "
    "labelled placeholder (config field `slo`, required_for_claims=true).")

_SLO_CLIFF_NOTE = (
    "THE SLO TERM IS NEARLY BINARY ON THIS WORKLOAD. Measured over three work seeds at fixed "
    "replica counts: 254 violations in total at k=1, and exactly 0 at k=2, k=3 and k=4. At "
    "k=1 the KV budget fills, recompute-preemption fires and admission is delayed past "
    "ttft_p95_s=2.0 s; from k=2 up, the placeholder prefill throughput (6000 tok/s) makes TTFT "
    "negligible and the fitted decode step time (about 17 ms at these batch sizes) is far "
    "below tbt_p95_s=0.05 s. So the SLO term behaves as an indicator on 'did the policy sit "
    "at one replica under load' rather than as a graded penalty, and the interesting part of "
    "the objective on this task is the cost/churn trade-off between k=2 and k=4. Both the "
    "targets and the prefill constant are placeholders, so this cliff is a property of "
    "uncalibrated inputs and would move under calibration.")


def builtin_specs() -> List[TaskSpec]:
    """The tasks registered by `register_builtin_tasks()`."""
    s1 = fixture_trace_path("S1_dev.jsonl")
    s3 = fixture_trace_path("S3_dev.jsonl")
    common = dict(
        service_model_name="llm_serving",
        config_file="llm_serving_l40s.json",
        service_model_factory=_llm_service_model,
        horizon_s=300.0,
        k0=1,
        obs_features=BUILTIN_FEATURES,
        obs_history=BUILTIN_HISTORY,
        obs_norm=_LLM_NORM,
        weights=_LLM_WEIGHTS,
        slo_profile="llm",
    )
    specs = [
        TaskSpec(
            name="LLMServing-S1",
            workload_name="S1_dev",
            workload_description=(
                "Reference trace S1_dev.jsonl: 225 arrivals over 300 s (0.7 rps nominal, "
                "p_long 0.5), sha256 2d8c19b0...c59f7. Output lengths are NOT in the trace; "
                "they are drawn per request at episode build time from the PLACEHOLDER length "
                "model, which is what the work seed varies."),
            workload_factory=_llm_trace_source([s1], "llm_trace_S1_dev", 300.0),
            n_episodes=1,
            notes=(_FIXTURE_NOTE, _PLACEHOLDER_NOTE, _SLO_CLIFF_NOTE,
                   "thinking_flip_prob = 0.0 in the generator that produced this trace, so the "
                   "observed class signal is a NOISELESS proxy for the length class. Nothing in "
                   "the observation vector exposes the class signal, so this task does not "
                   "benefit from it; a task that adds a class feature must repeat this caveat."),
            **common),
        TaskSpec(
            name="LLMServing-S3",
            workload_name="S3_dev",
            workload_description=(
                "Reference trace S3_dev.jsonl: 221 arrivals over 300 s, sha256 "
                "4c366a3e...79f007. Output lengths drawn at build time from the PLACEHOLDER "
                "length model."),
            workload_factory=_llm_trace_source([s3], "llm_trace_S3_dev", 300.0),
            n_episodes=1,
            notes=(_FIXTURE_NOTE, _PLACEHOLDER_NOTE, _SLO_CLIFF_NOTE),
            **common),
        TaskSpec(
            name="LLMServing-Mixed",
            workload_name="S1_dev+S3_dev",
            workload_description=(
                "Both reference traces as two episodes: `episode=0` is S1_dev, `episode=1` is "
                "S3_dev. The only builtin task where the `episode` index selects a different "
                "arrival process, so it is the one that exercises episode selection through the "
                "Gym API."),
            workload_factory=_llm_trace_source([s1, s3], "llm_trace_S1_S3", 300.0),
            n_episodes=2,
            notes=(_FIXTURE_NOTE, _PLACEHOLDER_NOTE, _SLO_CLIFF_NOTE,
                   "Two arrival traces is not workload diversity. It exercises episode indexing; "
                   "it does not make a generalisation claim."),
            **common),
        TaskSpec(
            name="RequestService-Poisson",
            service_model_name="request_service",
            config_file="request_service_default.json",
            workload_name="poisson_4rps_300s",
            workload_description=(
                "Homogeneous Poisson arrivals at 4 rps over 300 s (generator seed 7, ~1200 "
                "arrivals), service times drawn per request at build time from the PLACEHOLDER "
                "lognormal (mu=0, sigma=1). A deliberately uninteresting reference process, as "
                "the core's own docstring for `poisson_arrivals` says."),
            workload_factory=_rs_poisson_source(4.0, 300.0, 7, "rs_poisson_4rps"),
            service_model_factory=_rs_service_model,
            n_episodes=1,
            horizon_s=300.0,
            k0=1,
            obs_features=BUILTIN_FEATURES,
            obs_history=BUILTIN_HISTORY,
            obs_norm=_RS_NORM,
            weights=_RS_WEIGHTS,
            slo_profile="request_service",
            notes=(
                "EVERY CONSTANT IN THIS TASK'S CONFIG IS A PLACEHOLDER. request_service exists "
                "to exercise the ServiceModel seam with physics structurally unlike LLM "
                "decoding; nothing it produces is an empirical result about any real platform.",
                "The arrival times are fixed by the generator seed, so this task has ONE "
                "episode. Per-episode variation comes only from the service-time realisation, "
                "which the work seed selects. A task whose arrivals also vary needs a workload "
                "source that indexes arrivals by episode -- that is the corpus track's job.",
                "n_replicas_max = 20 here is a placeholder ceiling, NOT a measured physical "
                "limit (unlike the LLM config's 4), so a target near it is not hypothetical.",
                "THE ACTUATION LAG IS INVISIBLE ON THIS TASK. The placeholder cold start is "
                "2.0 s and the control interval is 15.0 s, so a scale-up completes entirely "
                "inside one control step and `n_booting` is zero at every scrape -- measured "
                "over 1400 steps across constant, ramp, alternating and hold-then-drop "
                "policies in both action modes. The four `n_booting[t-k]` elements of the "
                "observation are therefore constant zero, and the policy cannot see a "
                "scale-up in flight because there is nothing in flight to see. This makes the "
                "task a no-lag control problem, structurally easier than the LLM tasks, whose "
                "measured 31.9 s cold start spans 2.13 control intervals. It is an artefact "
                "of a placeholder constant, not a property of any real platform.",
                "THE SLO TARGET IS UNREACHABLE FOR ROUGHLY HALF THE REQUESTS. The placeholder "
                "service-time lognormal has median 1.0 s and the placeholder `latency_p95_s` "
                "is also 1.0 s, so about half of all requests miss the target on service time "
                "alone, before any queueing. Measured at the ceiling k=20 (80 concurrency "
                "slots against an offered load of about 6.6 requests in service): 582/580/600 "
                "violations out of 1231 requests on seeds 0/1/2, i.e. 47-49%. No policy can "
                "drive the SLO term below that floor, so on this task the term is largely a "
                "constant offset and the reward is dominated by the resource term.",
                _PLACEHOLDER_NOTE),
        ),
    ]
    return specs


#: Per service model, the spec whose conventions `register_workload_task`
#: inherits.  Populated by `register_builtin_tasks`.
REFERENCE_SPEC: Dict[str, TaskSpec] = {}


def register_builtin_tasks(*, override: bool = False) -> List[str]:
    """Register the builtin tasks.  Idempotent; called on `import kubegym.gym`."""
    ids: List[str] = []
    for spec in builtin_specs():
        if spec.name in _TASKS and not override:
            ids.extend(spec.gym_ids().values())
            REFERENCE_SPEC.setdefault(spec.service_model_name, _TASKS[spec.name])
            continue
        ids.extend(register_task(spec, override=override))
        # First registered task of a service model is its reference for
        # `register_workload_task`; later ones do not silently take over.
        REFERENCE_SPEC.setdefault(spec.service_model_name, spec)
    return sorted(ids)
