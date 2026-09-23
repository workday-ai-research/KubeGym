"""Workload corpus for KubeGym: Azure Functions 2019 replay + parametrized generator.

`register_corpus_tasks` is the join to the Gym layer and is imported lazily so
that importing this package never requires `gymnasium`.
"""

from __future__ import annotations

__all__ = ["register_corpus_tasks", "CorpusTraceSource", "find_corpus_dir",
           "FAMILIES", "SPLITS"]


def __getattr__(name):
    if name in __all__:
        from . import gym_tasks
        return getattr(gym_tasks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
