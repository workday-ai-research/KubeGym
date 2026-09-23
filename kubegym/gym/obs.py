"""Observation normalisation for the Gymnasium layer.

WHY NORMALISATION LIVES HERE AND NOT IN THE CORE
------------------------------------------------
`kubegym.core.state.ObservationSpec` assembles a raw, named, history-stacked
float32 vector from `ClusterState` scrapes and deliberately does not scale it:
per INTERFACE.md section 6, "normalisation is left to the wrapper, which knows
the action space and the replica ceiling".  This module is that wrapper's
normaliser.

THE RULE THAT CONSTRAINS EVERY CHOICE BELOW
-------------------------------------------
A divisor may only be a constant that is knowable *before the episode starts*:
a config field (`n_replicas_max`), a documented design constant of the task, or
a measured range boundary.  It must never be a statistic of the episode being
run -- no running mean, no per-episode max, no min-max over the rollout.  Two
reasons, and the second is the load-bearing one:

  1. A quantity computed from the whole episode is information from the future,
     so an observation scaled by it is not a function of the past.
  2. The benchmark's fairness guarantee is that an RL policy sees no more than a
     hand-written controller does.  A hand-written controller reading
     `ClusterState` has no episode statistics either.  Any adaptive
     normalisation would break the guarantee silently, because the resulting
     observation still *looks* like a scaled cluster metric.

Consequence, stated so nobody has to rediscover it: the scales here are round
design constants, not calibrated quantities.  They change the conditioning of
the learning problem, not its physics, and they are recorded in the env card.
A study that reports learning curves is reporting them under these scales.

CLIPPING
--------
Scaled features are clipped into `[0, clip]` (default `clip = 10.0`), which is
what makes the observation space a bounded `Box`.  Clipping is lossy at the top
end: under extreme overload, `n_waiting / concurrency_scale` saturates and the
policy stops seeing the queue grow.  With the default scales that needs roughly
`10 x concurrency_scale` queued requests, which does not occur in the shipped
300 s fixtures (asserted in `kubegym/tests/test_gym_observation.py`), but a
longer or heavier workload can reach it.  `clip` is in the env card for that
reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

#: How each `kubegym.core.state.FEATURES` key is scaled.  The value names which
#: of `ObsNorm`'s scales divides it; `"unit"` means the feature is already a
#: fraction in [0, 1] and is passed through.
FEATURE_SCALE_KIND: Dict[str, str] = {
    "n_ready": "replica",
    "n_booting": "replica",
    "n_draining": "replica",
    "n_effective": "replica",
    "target": "replica",
    "n_running": "concurrency",
    "n_waiting": "concurrency",
    "n_inflight": "concurrency",
    "running_per_ready": "per_replica_concurrency",
    "waiting_per_ready": "per_replica_concurrency",
    "sat_mean": "unit",
    "sat_max": "unit",
    "arrivals": "arrivals",
    "preemptions_total": "preemptions",
}


@dataclass(frozen=True)
class ObsNorm:
    """Divisors for the observation vector.  All are a-priori constants.

    Parameters
    ----------
    replica_scale
        Divides every replica count.  Set it to the config's `n_replicas_max`,
        so a saturated cluster reads 1.0 on `n_ready`.
    concurrency_scale
        Divides cluster-wide request counts (`n_running`, `n_waiting`,
        `n_inflight`).  A documented design constant, conventionally
        `n_replicas_max x per_replica_concurrency`.
    per_replica_concurrency
        Divides the already-per-replica ratios (`running_per_ready`,
        `waiting_per_ready`).
    arrivals_scale
        Divides the arrival count observed over one control interval.  A round
        constant near the task's nominal arrival rate times the control
        interval -- a design constant of the task, never measured from the
        episode in progress.
    preemptions_scale
        Divides cumulative preemption counts.
    clip
        Upper clip after scaling; the observation space is `Box(0, clip)`.
    """

    replica_scale: float
    concurrency_scale: float
    per_replica_concurrency: float
    arrivals_scale: float
    preemptions_scale: float = 100.0
    clip: float = 10.0

    def __post_init__(self) -> None:
        for k in ("replica_scale", "concurrency_scale", "per_replica_concurrency",
                  "arrivals_scale", "preemptions_scale", "clip"):
            v = float(getattr(self, k))
            if not (v > 0.0) or not np.isfinite(v):
                raise ValueError(f"ObsNorm.{k} must be finite and > 0, got {v!r}")

    def divisor(self, feature: str) -> float:
        """The constant that divides `feature`.  Raises on an unknown feature."""
        kind = FEATURE_SCALE_KIND.get(feature)
        if kind is None:
            raise KeyError(
                f"no normalisation declared for observation feature {feature!r}; "
                f"add it to kubegym.gym.obs.FEATURE_SCALE_KIND. Features are not given a "
                "default divisor on purpose: an undeclared scale is how an unnormalised or "
                "silently mis-scaled element enters the observation.")
        return {
            "replica": float(self.replica_scale),
            "concurrency": float(self.concurrency_scale),
            "per_replica_concurrency": float(self.per_replica_concurrency),
            "arrivals": float(self.arrivals_scale),
            "preemptions": float(self.preemptions_scale),
            "unit": 1.0,
        }[kind]

    def divisor_vector(self, features: Sequence[str], history: int) -> np.ndarray:
        """Divisors laid out to match `ObservationSpec.build` element order.

        `ObservationSpec` stacks history oldest-first with features as the inner
        loop, so the divisor vector is the per-feature vector tiled `history`
        times.  Element order is asserted against `ObservationSpec.names()` in
        `kubegym/tests/test_gym_observation.py`.
        """
        base = np.asarray([self.divisor(f) for f in features], dtype=np.float64)
        return np.tile(base, int(history)).astype(np.float32)

    def describe(self, features: Sequence[str], history: int) -> List[Dict[str, object]]:
        """Row-per-element description, for the env card and GYM_API.md."""
        out: List[Dict[str, object]] = []
        for k in range(int(history) - 1, -1, -1):
            for f in features:
                out.append({
                    "name": f if history == 1 else f"{f}[t-{k}]",
                    "feature": f,
                    "lag_intervals": k,
                    "scale_kind": FEATURE_SCALE_KIND[f],
                    "divisor": self.divisor(f),
                })
        return out

    def to_dict(self) -> Dict[str, float]:
        return {"replica_scale": self.replica_scale,
                "concurrency_scale": self.concurrency_scale,
                "per_replica_concurrency": self.per_replica_concurrency,
                "arrivals_scale": self.arrivals_scale,
                "preemptions_scale": self.preemptions_scale,
                "clip": self.clip}


def normalise(raw: np.ndarray, divisors: np.ndarray, clip: float) -> np.ndarray:
    """Elementwise `clip(raw / divisors, 0, clip)` as float32.

    Negative values cannot arise from `FEATURES` (every feature is a count, a
    ratio of counts, or a fraction in [0, 1]), so the lower clip at 0 is a
    guard, not a transformation.
    """
    out = np.asarray(raw, dtype=np.float32) / divisors
    return np.clip(out, np.float32(0.0), np.float32(clip), out=out)
