"""`kubegym.gym` -- the RL-facing layer: a Gymnasium env, a task registry, cards.

    import kubegym.gym                      # registers the builtin tasks
    import gymnasium

    for row in kubegym.gym.list_tasks():
        print(row["id"], row["calibrated"])

    env = gymnasium.make("KubeGym/LLMServing-S1-Delta-v0", split="dev")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(info["cost"], info["calibrated"])

Importing this module registers the builtin tasks with Gymnasium.  It is
idempotent, so importing it twice (or importing it after `gymnasium.make` has
already imported it lazily) does not duplicate a registration.

What lives where:

    env.py      KubeGymEnv, the Gymnasium adapter and the entry point
    tasks.py    TaskSpec, the registry, register_task (the extension point)
    cost.py     CostModel / CostWeights: the objective and its units
    obs.py      ObsNorm: observation scaling, and the rules that constrain it
    cards.py    env-card rendering from a live env
    smoke.py    the worked smoke run and the per-1000-step timing

THE CALIBRATION FLAG IS NOT OPTIONAL
------------------------------------
Both shipped configs are `calibrated: false`, so every reward, return and cost
this layer produces is non-reportable.  `info["calibrated"]` carries the flag on
every step, `list_tasks()` carries it per task, `info["episode_cost"]` is passed
through `ProvenancedConfig.stamp`, and the env cards say it in prose.  Do not
strip it.
"""
from __future__ import annotations

from .cost import SLO_PROFILES, TERMS, CostModel, CostWeights
from .env import KubeGymEnv, make_task_env
from .obs import FEATURE_SCALE_KIND, ObsNorm
from .tasks import (ACTION_MODES, DEFAULT_SPLIT_WORK_SEEDS, TaskSpec, builtin_specs,
                    fixture_trace_path, get_task, list_tasks, register_builtin_tasks,
                    register_task, register_workload_task, task_action_mode, task_ids,
                    task_names)

__all__ = [
    "KubeGymEnv", "make_task_env",
    "TaskSpec", "register_task", "register_workload_task", "register_builtin_tasks",
    "get_task", "list_tasks", "task_names", "task_ids", "task_action_mode",
    "builtin_specs", "fixture_trace_path", "ACTION_MODES", "DEFAULT_SPLIT_WORK_SEEDS",
    "CostModel", "CostWeights", "SLO_PROFILES", "TERMS",
    "ObsNorm", "FEATURE_SCALE_KIND",
    "env_card_markdown", "all_env_cards_markdown",
]

# Register on import. Idempotent: `register_builtin_tasks` skips names already in
# the registry rather than raising, so a second import is a no-op.
_REGISTERED_IDS = register_builtin_tasks()


def __getattr__(name: str):
    # cards.py imports env.py, so it is loaded lazily to keep `import
    # kubegym.gym` from pulling in the card renderer for every RL run.
    if name in ("env_card_markdown", "all_env_cards_markdown"):
        from . import cards
        return getattr(cards, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
