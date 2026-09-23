"""The reward: a configurable cost model over SLO violation, resource and churn.

The core does not define reward, SLO or cost -- INTERFACE.md section 1 says so
explicitly, because those are experiment choices rather than physics.  This
module makes the choice, once, per benchmark task, so that two papers reporting
`KubeGym/LLMServing-S1-Delta-v0` are reporting the same number.

SIGN CONVENTION
---------------
`cost >= 0` always, and `reward = -cost`.  Higher reward is better; the best
achievable reward is bounded above by 0 and is not attainable, because holding
`n_replicas_min` replicas for the whole horizon already accrues resource cost.
Nothing here is scaled to a "return of 1.0 = solved" convention: there is no
known optimum for these tasks and inventing a normaliser would imply one.

THE TERMS, WITH UNITS
---------------------
    slo         violating requests            x  w_slo    [reward / request]
    resource    replica-seconds billed        x  w_res    [reward / replica-s]
    churn       |change in target replicas|   x  w_churn  [reward / replica]

and, added once at the end of the episode:

    slo_drain        violating requests that completed during the post-horizon
                     drain
    resource_drain   replica-seconds billed during the post-horizon drain
    slo_unfinished   requests that never completed (drain hit its cap)

`total = slo + resource + churn + slo_drain + resource_drain + slo_unfinished`,
and `reward = -total`.  The env reports every term separately in `info["cost"]`
and `kubegym/tests/test_gym_cost.py` asserts the terms sum to the reported
total and that the reported total is exactly `-reward`.

NO LOOKAHEAD, AND MORE THAN THAT: NO PRIVILEGED INFORMATION EITHER
------------------------------------------------------------------
Each term at control step `i` is a function of what happened in
`(t_{i-1}, t_i]` only.  Requests are scored once, at the first control boundary
at which their outcome is decided -- either they completed inside the window, or
they have already been waiting longer than the hard SLO threshold, which is
observable at `t_i` without knowing when they will finish.

Every quantity the cost model reads is one a real deployment could compute from
router and orchestrator telemetry: `ttft`, `mean_tbt`, `latency`,
`queue_delay`, billed replica-seconds, and the controller's own actions.  The
cost model never reads `Request.demand` (the ground-truth total work), nor its
aliases `remaining` / `output_tokens_true`.  That is not just a convention:
`kubegym/tests/test_gym_cost.py::test_cost_model_source_reads_no_ground_truth`
greps this module's own source for those attribute names.  The consequence worth
stating is that the benchmark's objective is *implementable on the real cluster
the physics were measured on*, so a controller that does well here is optimising
a deployable objective and not a simulator artefact.

The drain terms are delayed, not lookahead: they are charged at the final step
of the episode, after the horizon, and the agent never observes them before
acting.  They exist so that pushing unfinished work past the horizon is not
free -- see `KubeGymEnv` and GYM_API.md for why that boundary effect would
otherwise be a gameable artefact.

CALIBRATION STATUS OF THE WEIGHTS
---------------------------------
The weights are a DESIGN CHOICE, not a measurement, and they are not physical
constants of the config, so the claims gate does not cover them: a config could
in principle be fully calibrated while the weights remain arbitrary.  They are
recorded in `unverified_gym.md`, surfaced in `info["cost"]["weights"]` and in
every env card, and chosen so each term is O(1) per control step at the task's
scale.  The SLO *targets* they are evaluated against do come from the config's
`slo` field, which is `required_for_claims` and is a labelled placeholder in
both shipped configs -- so the SLO term of every reward produced today is
placeholder-derived.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from ..provenance import ProvenancedConfig

#: Which `slo` sub-keys each profile requires, and which of them is the "hard"
#: deadline used to score a request that has not finished yet.
SLO_PROFILES: Dict[str, Dict[str, Any]] = {
    "llm": {
        "required_keys": ("ttft_p95_s", "tbt_p95_s", "ttft_hard_s"),
        "hard_key": "ttft_hard_s",
        "description": (
            "LLM token-streaming SLO. A completed request violates if its time-to-first-token "
            "exceeded `ttft_p95_s` or its mean time-between-tokens exceeded `tbt_p95_s`. An "
            "unfinished request violates if it has produced no first token `ttft_hard_s` after "
            "arrival."),
    },
    "request_service": {
        "required_keys": ("latency_p95_s", "latency_hard_s", "queue_delay_p95_s"),
        "hard_key": "latency_hard_s",
        "description": (
            "Request/response SLO. A completed request violates if its end-to-end latency "
            "exceeded `latency_p95_s` or its queueing delay exceeded `queue_delay_p95_s`. An "
            "unfinished request violates if it is still in the system `latency_hard_s` after "
            "arrival."),
    },
}

#: Names of the reward terms, in the order they appear in `info["cost"]`.
TERMS: Tuple[str, ...] = ("slo", "resource", "churn",
                          "slo_drain", "resource_drain", "slo_unfinished")


@dataclass(frozen=True)
class CostWeights:
    """Non-negative weights, one per cost term.  Units in the module docstring.

    Non-negativity is enforced rather than assumed: it is exactly what makes the
    cost monotone in each term, which is the property
    `kubegym/tests/test_gym_cost.py` pins (more SLO violation must never improve
    reward; more replica-seconds must never improve reward).
    """

    slo: float
    resource: float
    churn: float

    def __post_init__(self) -> None:
        for k in ("slo", "resource", "churn"):
            v = float(getattr(self, k))
            if v < 0.0 or v != v:
                raise ValueError(
                    f"CostWeights.{k} must be >= 0 (got {v!r}). A negative weight would make "
                    "the objective reward a controller for violating the SLO or for burning "
                    "replica-seconds, and would silently break the monotonicity the benchmark "
                    "asserts.")

    def to_dict(self) -> Dict[str, float]:
        return {"slo_per_violating_request": self.slo,
                "resource_per_replica_second": self.resource,
                "churn_per_replica_changed": self.churn}


class CostModel:
    """Turns per-interval telemetry into a cost decomposition.

    Stateless with respect to the episode: the caller (`KubeGymEnv`) owns the
    set of already-scored request sequence numbers and passes it in.  That keeps
    the model directly testable -- the monotonicity tests call `decompose` with
    synthetic numbers and never build a simulator.
    """

    def __init__(self, cfg: ProvenancedConfig, weights: CostWeights, *,
                 slo_profile: str):
        if slo_profile not in SLO_PROFILES:
            raise KeyError(f"unknown slo_profile {slo_profile!r}; have "
                           f"{sorted(SLO_PROFILES)}")
        self.cfg = cfg
        self.weights = weights
        self.slo_profile = slo_profile
        prof = SLO_PROFILES[slo_profile]
        slo = cfg.f("slo")
        if not isinstance(slo, dict):
            raise TypeError(f"config field 'slo' must be an object, got {type(slo).__name__}")
        missing = [k for k in prof["required_keys"] if k not in slo]
        if missing:
            raise KeyError(
                f"config {cfg.path} field 'slo' is missing {missing} required by slo_profile "
                f"{slo_profile!r}; have {sorted(slo)}")
        self.slo: Dict[str, float] = {k: float(slo[k]) for k in prof["required_keys"]}
        self.hard_key: str = str(prof["hard_key"])
        self.hard_s: float = float(self.slo[self.hard_key])

    # ------------------------------------------------------------------
    # SLO scoring
    # ------------------------------------------------------------------
    def violates_completed(self, req: Any) -> Tuple[bool, str]:
        """Score a request that has finished.  Returns (violated, reason)."""
        if self.slo_profile == "llm":
            ttft = req.ttft
            if ttft is not None and ttft > self.slo["ttft_p95_s"]:
                return True, "ttft"
            mtbt = req.mean_tbt
            if mtbt is not None and mtbt > self.slo["tbt_p95_s"]:
                return True, "tbt"
            return False, ""
        lat = req.latency
        if lat is not None and lat > self.slo["latency_p95_s"]:
            return True, "latency"
        qd = req.queue_delay
        if qd is not None and qd > self.slo["queue_delay_p95_s"]:
            return True, "queue_delay"
        return False, ""

    def violates_pending(self, req: Any, t: float) -> bool:
        """Score a request still in the system at `t` against the hard deadline.

        For the LLM profile the deadline is on first token, so a long-running
        request that is streaming normally is NOT a pending violation.  For the
        request-service profile there is no token stream, so the deadline is on
        being in the system at all.
        """
        if req.t_arrival > t:
            return False
        if self.slo_profile == "llm":
            if req.t_first_token is not None:
                return False
        return (t - req.t_arrival) > self.hard_s

    def score_window(self, requests: Sequence[Any], t_prev: float, t_now: float,
                     already_scored: Set[int], *,
                     score_pending: bool = True) -> Tuple[int, List[int], Dict[str, int]]:
        """Count SLO violations decided in `(t_prev, t_now]`.

        Returns `(n_violations, newly_scored_seqs, reason_counts)`.  A request is
        scored at most once per episode; `newly_scored_seqs` is what the caller
        must add to `already_scored`.  `reason_counts` also carries
        `n_scored` -- the number of requests whose outcome was decided in this
        window, violating or not -- so a downstream analysis can form a rate
        without the env having to guess which denominator it wants.

        `score_pending=False` scores only requests that completed inside the
        window.  The env uses it for the post-horizon drain window, where a
        request that never completes is charged once as `slo_unfinished`
        instead; scoring it as a pending violation there would move the same
        request between two terms depending on the drain cap.
        """
        n_viol = 0
        newly: List[int] = []
        reasons: Dict[str, int] = {}
        n_scored = 0
        for r in requests:
            if r.seq in already_scored:
                continue
            td = r.t_done
            if td is not None and t_prev < td <= t_now:
                bad, why = self.violates_completed(r)
                newly.append(r.seq)
                n_scored += 1
                if bad:
                    n_viol += 1
                    reasons[why] = reasons.get(why, 0) + 1
            elif td is None and score_pending and self.violates_pending(r, t_now):
                newly.append(r.seq)
                n_scored += 1
                n_viol += 1
                reasons["pending_hard"] = reasons.get("pending_hard", 0) + 1
        reasons["n_scored"] = n_scored
        return n_viol, newly, reasons

    def count_unfinished(self, requests: Iterable[Any], already_scored: Set[int]) -> int:
        """Requests that never completed and were never scored: episode-end term.

        Reached only when `Simulator.drain_all` hit its cap, which the core
        reports as `truncated_drain`.  Charging these keeps a truncated drain
        from looking cheaper than a completed one.
        """
        return sum(1 for r in requests
                   if r.t_done is None and r.seq not in already_scored)

    # ------------------------------------------------------------------
    # the decomposition
    # ------------------------------------------------------------------
    def decompose(self, *, slo_violations: int = 0, replica_seconds: float = 0.0,
                  churn_replicas: int = 0, slo_violations_drain: int = 0,
                  replica_seconds_drain: float = 0.0,
                  unfinished_requests: int = 0) -> Dict[str, float]:
        """Weighted terms plus their sum.  Every value is >= 0.

        Raw (unweighted) counts are returned alongside under `raw_*` so a result
        row records both what happened and what it was charged, and a reader can
        re-weight an episode after the fact without re-running it.
        """
        w = self.weights
        d = {
            "slo": w.slo * float(slo_violations),
            "resource": w.resource * float(replica_seconds),
            "churn": w.churn * float(churn_replicas),
            "slo_drain": w.slo * float(slo_violations_drain),
            "resource_drain": w.resource * float(replica_seconds_drain),
            "slo_unfinished": w.slo * float(unfinished_requests),
        }
        d["total"] = sum(d[k] for k in TERMS)
        d["raw_slo_violations"] = float(slo_violations)
        d["raw_replica_seconds"] = float(replica_seconds)
        d["raw_churn_replicas"] = float(churn_replicas)
        d["raw_slo_violations_drain"] = float(slo_violations_drain)
        d["raw_replica_seconds_drain"] = float(replica_seconds_drain)
        d["raw_unfinished_requests"] = float(unfinished_requests)
        return d

    # ------------------------------------------------------------------
    def provenance(self) -> Dict[str, Any]:
        """What the env card and every result row record about the objective."""
        return {
            "sign_convention": "reward = -cost; cost >= 0; higher reward is better",
            "terms": list(TERMS),
            "units": {
                "slo": "reward per violating request",
                "resource": "reward per replica-second",
                "churn": "reward per replica added or removed",
            },
            "weights": self.weights.to_dict(),
            "weights_calibrated": False,
            "weights_note": (
                "DESIGN CHOICE, not a measurement. Chosen so each term is O(1) per control "
                "step at this task's scale. Not covered by the config's claims gate because "
                "they are not physical constants; recorded in unverified_gym.md."),
            "slo_profile": self.slo_profile,
            "slo_profile_description": SLO_PROFILES[self.slo_profile]["description"],
            "slo_targets": dict(self.slo),
            "slo_targets_calibrated": self.cfg.is_calibrated("slo"),
            "slo_targets_source": self.cfg.source("slo"),
            "reads_ground_truth_demand": False,
        }
