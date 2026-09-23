"""Decode a KubeGym observation vector back into cluster quantities.

WHY THIS MODULE EXISTS
----------------------
The analytic controllers ported here must consume **exactly** the observation an
RL policy consumes -- otherwise a controller-vs-policy comparison is confounded
by an information asymmetry.  The source project's controllers read a
`ClusterState` directly (`decide(hist, t)`), which gives them the whole scrape
history and the deployed system's native metric names.  The Gym observation is
a 37-element `float32` vector: 9 features x 4 lags, each divided by an a-priori
constant and clipped into `[0, clip]`, plus one appended `episode_progress`
element.

`ObsView` is the documented inverse of that transform.  It is a *decoder*, not
a source of extra information: every quantity it returns is recovered from the
observation vector alone.  Three consequences, all of which are real and all of
which are reported rather than hidden:

1.  **The history is four lags, not the whole episode.**  A controller with a
    memory longer than four control intervals (the Kubernetes HPA's 300 s
    scale-down stabilization window is 20 intervals at the 15 s cadence) cannot
    reconstruct it from the observation.  The resolution used here is that such
    a window operates on the controller's **own past decisions**, which are
    controller-internal state and not observation -- a recurrent policy could
    accumulate the same thing.  `HPAGym` therefore keeps its own
    `_desired_hist`, exactly as the source implementation does, and the port is
    decision-identical to the source on that axis (verified in
    `parity_port.py`).

2.  **Counts are recovered by rounding.**  `n_ready`, `n_waiting`, ... are
    integers divided by an exact constant and stored as `float32`.
    `decode()` multiplies back and rounds; `assert_lossless()` checks the
    round-trip against the env's own scrape on a fixed trace.

3.  **Clipping is genuinely lossy and is flagged.**  If a scaled element equals
    `clip`, the decoded value is a *lower bound* and the scrape is marked
    `saturated`.  Under that condition the ported controller is blind in exactly
    the way the RL policy is blind, which is the point -- but a run in which it
    happens is not a run in which either method saw the queue.  The evaluation
    counts saturated steps and reports them.

WHAT IS *NOT* IN THE OBSERVATION, AND WHAT THE PORT DOES ABOUT IT
-----------------------------------------------------------------
| source-side signal | in the 37-element observation? | port's substitute |
|---|---|---|
| `mean vllm:kv_cache_usage_perc` / `svc:concurrency_usage_perc` | yes, `sat_mean` (divisor 1, exact) | used directly |
| `n_running`, `n_waiting`, `n_ready`, `n_booting`, `n_draining`, `target` | yes | used directly |
| `n_replicas_effective` | derived: ready + booting + draining | used directly |
| `arrivals_since_last` | yes, `arrivals` | used directly |
| `vllm:generation_tokens_total` (token throughput) | **no** | arrivals per second; see `gym_controllers.LeadingIndicatorGym` |
| `router_inflight` per-request token counts | **no** | not used by any ported controller |
| `t` (wall time in the episode) | yes, via `episode_progress x horizon_s` | used directly |
| whole scrape history | **no** (4 lags) | controller-internal decision memory only |

The two token-based signals are unavailable on the `request_service` service
model in any case (it exposes `svc:*` metrics and has no token stream), so their
absence from the observation costs nothing on the corpus tasks.  It does change
`LeadingIndicator` from a token-velocity controller into an arrival-velocity
controller, which is an adaptation and is labelled as one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

#: Features whose raw value is an integer count (recovered by rounding).
INTEGER_FEATURES = frozenset({
    "n_ready", "n_booting", "n_draining", "n_effective", "target",
    "n_running", "n_waiting", "n_inflight", "arrivals",
})

_SCRAPE_FIELDS = ("n_ready", "n_booting", "n_draining", "target",
                  "n_running", "n_waiting", "sat_mean", "sat_max", "arrivals")


@dataclass
class Scrape:
    """One decoded control-interval scrape, in cluster units."""

    n_ready: float = 0.0
    n_booting: float = 0.0
    n_draining: float = 0.0
    target: float = 0.0
    n_running: float = 0.0
    n_waiting: float = 0.0
    sat_mean: float = 0.0
    sat_max: float = 0.0
    arrivals: float = 0.0
    #: True if ANY element of this lag was at the clip ceiling, i.e. the decoded
    #: values for those elements are lower bounds.
    saturated: bool = False
    #: Names of the elements that were at the ceiling.
    saturated_features: Tuple[str, ...] = ()

    @property
    def n_effective(self) -> float:
        """What the Kubernetes HPA calls `currentReplicas`: everything billed."""
        return self.n_ready + self.n_booting + self.n_draining

    @property
    def inflight(self) -> float:
        return self.n_running + self.n_waiting


class ObsView:
    """The documented inverse of `KubeGymEnv.observe_from_history`.

    Construct from a `TaskSpec` (or from an env's `task_spec`).  Call
    `decode(obs)` to get `(scrapes_oldest_first, t_seconds, progress)`.
    """

    def __init__(self, task_spec):
        self.features: Tuple[str, ...] = tuple(task_spec.obs_features)
        self.history: int = int(task_spec.obs_history)
        self.norm = task_spec.obs_norm
        self.clip = float(self.norm.clip)
        self.horizon_s = float(task_spec.horizon_s)
        self.n_control_steps = int(task_spec.n_control_steps())
        self.dt = self.horizon_s / self.n_control_steps
        self.divisors = np.asarray(
            [self.norm.divisor(f) for f in self.features], dtype=np.float64)
        self.dim = len(self.features) * self.history + 1
        missing = [f for f in _SCRAPE_FIELDS if f not in self.features]
        if missing:
            raise KeyError(
                f"task observation lacks features {missing}; the ported controllers read "
                "them and would otherwise silently see zero. Register the task with the "
                "canonical obs_features tuple or extend Scrape.")

    # -- decoding ------------------------------------------------------
    def decode(self, obs) -> Tuple[List[Scrape], float, float]:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        if obs.shape != (self.dim,):
            raise ValueError(f"observation shape {obs.shape} != expected ({self.dim},)")
        block = obs[: len(self.features) * self.history].reshape(
            self.history, len(self.features))
        progress = float(obs[-1])
        scrapes: List[Scrape] = []
        for lag in range(self.history):                     # oldest first
            row = block[lag]
            sat_names = tuple(f for f, v in zip(self.features, row)
                              if v >= self.clip - 1e-6)
            vals: Dict[str, float] = {}
            for f, v, d in zip(self.features, row, self.divisors):
                raw = v * d
                vals[f] = float(round(raw)) if f in INTEGER_FEATURES else float(raw)
            scrapes.append(Scrape(saturated=bool(sat_names),
                                  saturated_features=sat_names,
                                  **{k: vals[k] for k in _SCRAPE_FIELDS}))
        # `episode_progress` is `step_i / n_control_steps` stored as float32, so
        # `progress * horizon_s` is only accurate to ~1e-5 s. Recovering the step
        # index first and multiplying by the control interval makes the episode
        # time exact. This matters: several controllers compare an elapsed time
        # against a window boundary (`t - t_last <= 300.0`, `t - t_change <
        # cooldown_s`), and a 2e-5 s error at exactly the boundary flips the
        # comparison. It was the entire cause of the first port-parity
        # disagreements measured for `hpa`, `queue` and `leading`; see
        # `port_parity.json`. Rounding a value that is known to be quantised is
        # a decode, not extra information.
        step_i = int(round(progress * self.n_control_steps))
        t = step_i * self.dt
        return scrapes, t, progress

    # -- checks --------------------------------------------------------
    def assert_lossless(self, obs, cluster_state) -> None:
        """Assert the most recent decoded lag reproduces a live `ClusterState`.

        Used by the port-verification harness: the decoder is only a legitimate
        stand-in for `ClusterState` if the round-trip is exact whenever no
        element is clipped.
        """
        scrapes, _, _ = self.decode(obs)
        st = scrapes[-1]
        if st.saturated:
            return                                          # lower bounds only
        checks = {
            "n_ready": (st.n_ready, float(cluster_state.n_ready)),
            "n_booting": (st.n_booting, float(cluster_state.n_booting)),
            "n_draining": (st.n_draining, float(cluster_state.n_draining)),
            "target": (st.target, float(cluster_state.target)),
            "n_running": (st.n_running, float(cluster_state.n_running)),
            "n_waiting": (st.n_waiting, float(cluster_state.n_waiting)),
            "arrivals": (st.arrivals, float(cluster_state.arrivals_since_last)),
            "sat_mean": (st.sat_mean, float(cluster_state.kv_mean)),
            "sat_max": (st.sat_max, float(cluster_state.kv_max)),
        }
        for name, (got, want) in checks.items():
            tol = 1e-4 if name.startswith("sat") else 1e-9
            if abs(got - want) > tol:
                raise AssertionError(
                    f"observation decode lost {name}: decoded {got!r}, scrape {want!r}. "
                    "The port cannot claim to consume the same information as the policy "
                    "while the decode is lossy on an unclipped element.")
