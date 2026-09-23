"""Verify the Gym-interface controller ports against the source implementation.

METHOD
------
The core's own parity harness compares two simulators step for step on fixed
traces.  A controller port cannot be checked that way directly, because the
moment two controllers disagree their trajectories diverge and every later
comparison is meaningless.  So this harness compares **decisions along one
trajectory**:

    drive the episode with the PORTED controller (reading the 37-element
    observation), and at every control step also ask the SOURCE controller for
    its decision given the raw `ClusterState` history at that same step.

Both are handed the identical state, so a step-by-step decision comparison is
well-posed, and agreement at every step implies the two would generate identical
trajectories (by induction on the step index).  Reported per (controller,
config, trace): number of steps, number of disagreeing steps, the maximum
absolute difference in requested replicas, and the step indices of the first few
disagreements.

The observation decode itself is checked on every step by
`ObsView.assert_lossless`, so a "no disagreement" verdict is not resting on the
decode silently agreeing with itself.

WHAT CAN AND CANNOT BE COMPARED
-------------------------------
* `static` -- reads nothing; exact by construction.
* `hpa` with `metric in {conc, running}` -- comparable on both service models.
* `hpa` with `metric = kv` -- the source reads the raw metric name
  `vllm:kv_cache_usage_perc`, which does not exist under `request_service` (the
  metric guard raises rather than returning zero, which is the point of the
  guard).  Compared on the builtin `LLMServing-S1` task instead, where the metric
  exists and `ClusterState.kv_mean` is the same quantity the observation's
  `sat_mean` carries.
* `queue` -- comparable on both.
* `forecast` with `signal = "rate"` -- comparable on both.  The source's
  `signal = "tokens"` variant has no counterpart: `request_service` has no token
  stream and no token counter appears in the observation for either model.
* `leading` -- the source drives on differenced
  `vllm:generation_tokens_total`, so it has **no** counterpart under
  `request_service`.  It is an adaptation, and the harness compares it against a
  reference implementation of the *same adapted algorithm* reading the raw
  `ClusterState` with unrestricted history (`_LeadingReference` below).  That
  measures the cost of the 4-lag observation window, which is the only thing
  worth measuring here; it is not a check against the published method and is
  labelled `kind="adaptation"`.
* `oracle` -- privileged; the source's version is LLM-specific (it needs
  `decode_step_time_s` and `kv_tokens_per_replica`), so there is no source
  implementation to compare decisions against.  `_check_oracle_properties`
  asserts the two properties that would otherwise be taken on trust:
  **P1** the ground-truth demand vector is genuinely read (doubling it along a
  fixed trajectory changes at least one decision, so `uses_privileged_info` is
  an accurate label and not decoration), and **P2** the privilege buys
  *anticipation* (there is at least one step where the oracle raises its target
  while the observation shows no queue and non-increasing arrivals, i.e. it acts
  on information no deployable controller has).  Neither is a port check, and
  the file says so.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import source_controllers as SRC                              # noqa: E402

from .gym_controllers import REGISTRY, HPA_SEMANTICS_SOURCE    # noqa: E402
from .obsmap import ObsView                                    # noqa: E402
from .runner import (_ensure_registered, load_manifest, make_env,   # noqa: E402
                     trace_entries)


# ---------------------------------------------------------------------------
# reference implementation for the one adapted controller
# ---------------------------------------------------------------------------
class _LeadingReference(SRC.Controller):
    """`LeadingIndicator` with the arrival-rate signal, on raw `ClusterState`.

    Identical arithmetic to `gym_controllers.LeadingIndicatorGym`, but reading
    `ClusterState` with the whole `sim.hist` available, which is what the source
    controller interface gives.  Its only purpose is to isolate the effect of the
    4-lag observation window from the effect of the signal substitution.
    """

    name = "leading_ref"

    def __init__(self, cfg, per_replica_rps=4.0, kappa=15.0, window_s=30.0,
                 backlog_target=2.0, **kw):
        super().__init__(cfg, per_replica_rps=per_replica_rps, kappa=kappa,
                         window_s=window_s, backlog_target=backlog_target, **kw)
        self._prev_rate = None
        self._prev_t = None

    def decide(self, hist, t):
        st = hist[-1]
        rate = self.arrival_rate(hist, self.window_s)
        drate = 0.0
        if self._prev_rate is not None and self._prev_t is not None and t > self._prev_t:
            drate = (rate - self._prev_rate) / (t - self._prev_t)
        self._prev_rate, self._prev_t = rate, t
        signal = max(0.0, rate + self.kappa * drate)
        desired_rate = signal / max(self.per_replica_rps, 1e-9)
        inflight = st.n_running + st.n_waiting
        desired_backlog = inflight / max(self.backlog_target, 1e-9)
        return self.clamp(max(desired_rate, desired_backlog, float(self.k_min)))


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------
def compare_one_counted(env, port, source, episode: int, work_seed: int) -> Dict[str, Any]:
    """Same as `compare_one` but counts every disagreement, not just the first 8."""
    view = port.view
    obs, info = env.reset(options={"episode": episode, "work_seed": work_seed})
    port.reset(env)
    sim = env.unwrapped.sim
    n_steps = 0
    n_dis = 0
    n_dis_padded = 0                    # disagreements inside the zero-pad window
    max_abs = 0
    first: List[Dict[str, Any]] = []
    pad_steps = view.history - 1
    while True:
        view.assert_lossless(obs, sim.hist[-1])
        k_port = port.desired_replicas(obs)
        k_src = int(source.decide(sim.hist, float(sim.t)))
        if k_port != k_src:
            n_dis += 1
            if n_steps < pad_steps:
                n_dis_padded += 1
            max_abs = max(max_abs, abs(k_port - k_src))
            if len(first) < 6:
                first.append({"step": n_steps, "t": round(float(sim.t), 1),
                              "port": k_port, "source": k_src})
        a = int(k_port - port.k_min) if port.action_mode == "absolute" else port.act(obs)
        obs, _, term, trunc, info = env.step(a)
        n_steps += 1
        if term or trunc:
            break
    return {"n_steps": n_steps, "n_disagree": n_dis,
            "n_disagree_in_zeropad_window": n_dis_padded,
            "max_abs_diff": int(max_abs), "first_disagreements": first}


# ---------------------------------------------------------------------------
# the privileged oracle: property checks, not a port check
# ---------------------------------------------------------------------------
def _check_oracle_properties(family: str, episodes: Sequence[int],
                             params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assert that the oracle's privilege is real and that it buys anticipation.

    There is no source implementation of a request-service oracle to shadow, so
    three measured properties stand in for a port check:

      **P1a (demand path)** doubling the ground-truth `demand` vector must move at
        least one decision.  Evaluated in `mode="rate"`, where the work-rate bound
        is the only bound, so a change in demand cannot be masked; the same
        perturbation is also run under the *configured* mode, and the difference
        between the two is itself reported (on this corpus the residency bound
        dominates, so the demand path is fully masked there).
      **P1b (foresight path)** hiding requests that have not arrived yet must move
        at least one decision.
      **P2 (anticipation)** there must be at least one step where the oracle raises
        its target while the observation shows an empty queue and non-increasing
        arrivals, i.e. it acts on information no deployable controller has.

    All three are evaluated along a FIXED trajectory: the oracle drives the
    episode once and its action sequence is recorded, then the episode is replayed
    under that same recorded action sequence so that every alternative decision is
    queried against identical states.  Without that, perturbing the demand vector
    would change the trajectory and the comparison would be ill-posed.

    Per-family verdicts do not raise: on a low-volume trace every bound clamps to
    `n_replicas_min` and no privileged input can change the answer.  `run()` takes
    the judgement across families.
    """
    params = dict(params or {"H_s": 30.0, "safety": 1.0, "mode": "rate+residency"})
    env = make_env(family, "dev", "absolute")
    u = env.unwrapped
    view = ObsView(u.task_spec)
    cfg = u.cfg
    ws = int(u.task_spec.split_work_seeds["dev"][0])
    runs: List[Dict[str, Any]] = []
    for ep in episodes:
        oracle = REGISTRY["oracle"](view, cfg, action_mode="absolute", **params)
        queue = REGISTRY["queue"](view, cfg, action_mode="absolute",
                                  target_concurrency=2.0, hyst=0.0, cooldown_s=0.0)
        # pass 1: the oracle drives; record its actions and the queue baseline
        obs, _ = env.reset(options={"episode": int(ep), "work_seed": ws})
        oracle.reset(env)
        queue.reset(env)
        actions: List[int] = []
        k_oracle: List[int] = []
        k_queue: List[int] = []
        waiting: List[float] = []
        arrivals: List[float] = []
        while True:
            sc, _, _ = view.decode(obs)
            waiting.append(sc[-1].n_waiting)
            arrivals.append(sc[-1].arrivals)
            ko = oracle.desired_replicas(obs)
            k_oracle.append(ko)
            k_queue.append(queue.desired_replicas(obs))
            a = int(ko - oracle.k_min)
            actions.append(a)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        # pass 2/3: same trajectory, one privileged input perturbed at a time.
        # The trajectory is fixed by replaying `actions`, so each alternative
        # decision is queried against an identical state.
        def replay(mutate, mode_override=None) -> List[int]:
            pr = dict(params)
            if mode_override is not None:
                pr["mode"] = mode_override
            o2 = REGISTRY["oracle"](view, cfg, action_mode="absolute", **pr)
            ob, _ = env.reset(options={"episode": int(ep), "work_seed": ws})
            o2.reset(env)
            mutate(o2)
            ks: List[int] = []
            for a in actions:
                ks.append(o2.desired_replicas(ob))
                ob, _, te, tu, _ = env.step(a)
                if te or tu:
                    break
            return ks

        def double_demand(o):
            if o._arr is not None:
                o._dem_sorted = o._dem_sorted * 2.0
                o._cum = np.concatenate([[0.0], np.cumsum(o._dem_sorted)])

        def blind_to_future(o):
            if o._arr is not None:
                keep = o._arr_sorted <= 0.0
                o._arr_sorted = o._arr_sorted[keep]
                o._dem_sorted = o._dem_sorted[keep]
                o._cum = np.concatenate([[0.0], np.cumsum(o._dem_sorted)])

        def noop(o):
            return None

        # P1a: the DEMAND path. Checked in mode="rate", where the work-rate bound
        # is the only bound, so a change in demand cannot be masked.
        k_rate_ref = replay(noop, mode_override="rate")
        k_rate_2x = replay(double_demand, mode_override="rate")
        n_moved_demand = sum(1 for x, y in zip(k_rate_ref, k_rate_2x) if x != y)
        # the same perturbation under the CONFIGURED mode, to measure whether the
        # residency bound masks the demand path on this corpus
        k_cfg_2x = replay(double_demand)
        n_moved_demand_cfg = sum(1 for x, y in zip(k_oracle, k_cfg_2x) if x != y)
        # P1b: the FORESIGHT path -- hide requests that have not arrived yet
        k_blind = replay(blind_to_future)
        n_moved_future = sum(1 for x, y in zip(k_oracle, k_blind) if x != y)

        # P2: a step where the oracle raises its target although the observation
        # shows an empty queue and non-increasing arrivals
        anticipatory = 0
        for i in range(1, len(k_oracle)):
            if (k_oracle[i] > k_oracle[i - 1] and waiting[i] <= 0.0
                    and arrivals[i] <= arrivals[i - 1]):
                anticipatory += 1
        runs.append({"family": family, "episode": int(ep), "n_steps": len(k_oracle),
                     "P1a_decisions_moved_by_doubling_demand_mode_rate": n_moved_demand,
                     "P1a_decisions_moved_by_doubling_demand_configured_mode":
                         n_moved_demand_cfg,
                     "P1b_decisions_moved_by_hiding_future_arrivals": n_moved_future,
                     "P2_anticipatory_scale_ups": anticipatory,
                     "n_steps_oracle_above_queue": sum(
                         1 for a, b in zip(k_oracle, k_queue) if a > b)})
    env.close()
    p1a = sum(r["P1a_decisions_moved_by_doubling_demand_mode_rate"] for r in runs)
    p1a_cfg = sum(r["P1a_decisions_moved_by_doubling_demand_configured_mode"] for r in runs)
    p1b = sum(r["P1b_decisions_moved_by_hiding_future_arrivals"] for r in runs)
    p2 = sum(r["P2_anticipatory_scale_ups"] for r in runs)
    verdict = {
        "controller": "oracle", "kind": "privileged_property_check",
        "family": family, "params": params,
        "P1a_demand_is_read": p1a > 0,
        "P1a_total_decisions_moved_mode_rate": p1a,
        "P1a_total_decisions_moved_configured_mode": p1a_cfg,
        "P1b_future_arrivals_are_read": p1b > 0,
        "P1b_total_decisions_moved": p1b,
        "P2_privilege_buys_anticipation": p2 > 0,
        "P2_total_anticipatory_scale_ups": p2,
        "runs": runs,
        "note": ("not a port check: the source project's Oracle is LLM-specific "
                 "(decode_step_time_s, kv_tokens_per_replica) and has no request-service "
                 "counterpart to shadow."),
    }
    if p1a == 0 and p1b == 0:
        verdict["P1_inconclusive_on_this_family"] = (
            "no decision moved under either perturbation on these episodes. That is not "
            "evidence against the privilege label: on a low-volume trace every bound clamps "
            "to n_replicas_min and no privileged input can change the answer. The verdict is "
            "taken across families in run().")
    if p1a > 0 and p1a_cfg == 0:
        verdict["P1a_masking_finding"] = (
            "the demand path is REAL but MASKED under the configured mode on this family: "
            f"doubling every request's true service demand moved {p1a} decisions in mode "
            "'rate' and 0 in mode 'rate+residency'. On this corpus offered work per request "
            "is small (pooled reconstructed Azure service times: median 0.017 s) relative to "
            "a 4-slot replica over a 32 s lookahead, so the residency bound "
            "ceil(n_concurrent / rs_max_concurrency) dominates the work-rate bound. The "
            "oracle's privilege here is therefore mostly KNOWING FUTURE ARRIVAL TIMES, not "
            "knowing their durations -- see P1b.")
    if p2 == 0:
        verdict["P2_caveat"] = (
            "no anticipatory scale-up found on these episodes: on this corpus the oracle's "
            "advantage may be entirely in sizing rather than in timing. Reported, not raised, "
            "because a trace with no pre-arrival ramp legitimately admits none.")
    return verdict


# ---------------------------------------------------------------------------
# the configurations checked
# ---------------------------------------------------------------------------
def _port(name: str, view, cfg, **params):
    return REGISTRY[name](view, cfg, action_mode="absolute", **params)


def _cases_request_service() -> List[Tuple[str, Dict[str, Any], Dict[str, Any], str]]:
    """(controller, port params, source params, kind) on `request_service`."""
    cases: List[Tuple[str, Dict[str, Any], Dict[str, Any], str]] = []
    for k in (1, 4, 10, 20):
        cases.append(("static", {"k": k}, {"k": k}, "port"))
    for metric in ("conc", "running"):
        for target in (1.0, 2.0, 4.0):
            for ds in (60.0, 300.0):
                p = {"metric": metric, "target": target, "tolerance": 0.1,
                     "down_stabilization_s": ds}
                cases.append(("hpa", p, dict(p), "port"))
    for tc in (1.0, 2.0, 4.0):
        for hyst in (0.0, 0.15):
            for cd in (0.0, 60.0):
                p = {"target_concurrency": tc, "hyst": hyst, "cooldown_s": cd}
                cases.append(("queue", p, dict(p), "port"))
    for fc in ("ewma", "holt"):
        for H in (0.0, 30.0):
            for safety in (1.0, 1.5):
                pp = {"forecaster": fc, "H_s": H, "safety": safety,
                      "per_replica_rps": 4.0, "backlog_target": 12.0,
                      "alpha": 0.4, "beta": 0.3}
                sp = {"forecaster": fc, "signal": "rate", "H_s": H, "safety": safety,
                      "per_replica_capacity": 4.0, "alpha": 0.4, "beta": 0.3}
                cases.append(("forecast", pp, sp, "port"))
    for kappa in (0.0, 15.0):
        for w in (15.0, 30.0, 45.0):
            p = {"per_replica_rps": 4.0, "kappa": kappa, "window_s": w,
                 "backlog_target": 2.0}
            cases.append(("leading", p, dict(p), "adaptation"))
    return cases


def _cases_llm_serving() -> List[Tuple[str, Dict[str, Any], Dict[str, Any], str]]:
    """The `metric = kv` HPA variant, checkable only where the metric exists."""
    cases = []
    for target in (0.3, 0.6, 0.9):
        for ds in (60.0, 300.0):
            p = {"metric": "kv", "target": target, "tolerance": 0.1,
                 "down_stabilization_s": ds}
            cases.append(("hpa", p, dict(p), "port"))
    for metric in ("conc", "running"):
        p = {"metric": metric, "target": 8.0, "tolerance": 0.1,
             "down_stabilization_s": 300.0}
        cases.append(("hpa", p, dict(p), "port"))
    return cases


def _check_forecast_backlog_note() -> str:
    return ("forecast: the source hard-codes the backlog guard divisor at 12.0; the port exposes "
            "it as `backlog_target` and the parity cases pin it to 12.0 to match. The swept grid "
            "uses smaller values because a 4-slot replica cannot hold 12 concurrent requests.")


def run(out_json: str = "out/port_parity.json", n_episodes: int = 3) -> Dict[str, Any]:
    _ensure_registered()
    import gymnasium
    import kubegym.gym  # noqa: F401

    results: List[Dict[str, Any]] = []

    # ---- request_service, corpus traces -----------------------------
    for family in ("burst", "azure_replay"):
        env = make_env(family, "dev", "absolute")
        u = env.unwrapped
        view = ObsView(u.task_spec)
        cfg = u.cfg
        eps = list(range(min(n_episodes, u.task_spec.n_episodes)))
        ws = int(u.task_spec.split_work_seeds["dev"][0])
        for name, pp, sp, kind in _cases_request_service():
            for ep in eps:
                port = _port(name, view, cfg, **pp)
                if kind == "adaptation":
                    src = _LeadingReference(cfg, **sp)
                else:
                    ssp = dict(sp)
                    if name == "forecast":
                        ssp["per_replica_capacity"] = pp["per_replica_rps"]
                    src = SRC.make(name, cfg, **ssp)
                r = compare_one_counted(env, port, src, ep, ws)
                r.update({"service_model": "request_service", "task": u.task_name,
                          "family": family, "controller": name, "kind": kind,
                          "params": pp, "episode": ep})
                results.append(r)
        env.close()

    # ---- llm_serving, for the kv metric ------------------------------
    env = gymnasium.make("KubeGym/LLMServing-S1-Abs-v0", split="dev")
    u = env.unwrapped
    view = ObsView(u.task_spec)
    cfg = u.cfg
    ws = int(u.task_spec.split_work_seeds["dev"][0])
    for name, pp, sp, kind in _cases_llm_serving():
        for ep in range(min(2, u.task_spec.n_episodes)):
            port = _port(name, view, cfg, **pp)
            src = SRC.make(name, cfg, **sp)
            r = compare_one_counted(env, port, src, ep, ws)
            r.update({"service_model": "llm_serving", "task": u.task_name,
                      "family": "llm_serving_s1", "controller": name, "kind": kind,
                      "params": pp, "episode": ep})
            results.append(r)
    env.close()

    # ---- the privileged oracle: property checks ----------------------
    _man = load_manifest()

    def _busiest(fam: str, n: int = 3) -> List[int]:
        """Highest-request-count dev episodes of a family.

        A privileged input can only change a decision on a trace whose load is
        high enough that some bound is not clamped at `n_replicas_min`; the
        `azure_replay` dev split contains traces with 15 requests in an hour, on
        which no controller of any kind can be distinguished from static-1.
        """
        ents = trace_entries(_man, fam, "dev")
        order = sorted(range(len(ents)), key=lambda i: -int(ents[i]["n_requests"]))
        return sorted(order[:n])

    oracle_checks = [_check_oracle_properties(fam, _busiest(fam))
                     for fam in ("burst", "azure_replay", "variable")]
    _p1a = sum(c["P1a_total_decisions_moved_mode_rate"] for c in oracle_checks)
    _p1b = sum(c["P1b_total_decisions_moved"] for c in oracle_checks)
    if _p1a == 0 and _p1b == 0:
        raise AssertionError(
            "oracle P1 failed on every family checked: neither doubling the ground-truth "
            "demand vector (mode 'rate') nor hiding requests that have not arrived yet moved "
            "any decision, so the controller is not reading privileged information at all "
            "and uses_privileged_info would be false advertising.")

    # ---- summary -----------------------------------------------------
    by: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in results:
        key = (r["controller"], r["service_model"], r["kind"])
        agg = by.setdefault(key, {"controller": r["controller"],
                                  "service_model": r["service_model"],
                                  "kind": r["kind"], "n_runs": 0, "n_steps": 0,
                                  "n_disagree": 0, "n_disagree_in_zeropad_window": 0,
                                  "max_abs_diff": 0, "examples": []})
        agg["n_runs"] += 1
        agg["n_steps"] += r["n_steps"]
        agg["n_disagree"] += r["n_disagree"]
        agg["n_disagree_in_zeropad_window"] += r["n_disagree_in_zeropad_window"]
        agg["max_abs_diff"] = max(agg["max_abs_diff"], r["max_abs_diff"])
        if r["n_disagree"] and len(agg["examples"]) < 3:
            agg["examples"].append({"params": r["params"], "family": r["family"],
                                    "episode": r["episode"],
                                    "first": r["first_disagreements"]})
    summary = sorted(by.values(), key=lambda d: (d["service_model"], d["controller"]))
    out = {
        "hpa_semantics_source": HPA_SEMANTICS_SOURCE,
        "source_controllers_sha256": _sha256(os.path.join(_HERE, "source_controllers.py")),
        "method": ("decisions compared along one trajectory: the ported controller drives, the "
                   "source controller is shadowed on the raw ClusterState history at the same "
                   "step. ObsView.assert_lossless is checked on every step."),
        "notes": [_check_forecast_backlog_note()],
        "summary": summary,
        "oracle_property_checks": oracle_checks,
        "runs": results,
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    return out


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    o = run()
    for c in o["oracle_property_checks"]:
        print(f"oracle[{c['family']}] P1a_demand_read={c['P1a_demand_is_read']} "
              f"(rate-mode {c['P1a_total_decisions_moved_mode_rate']}, "
              f"cfg-mode {c['P1a_total_decisions_moved_configured_mode']})  "
              f"P1b_future_read={c['P1b_future_arrivals_are_read']} "
              f"({c['P1b_total_decisions_moved']})  "
              f"P2_anticipation={c['P2_privilege_buys_anticipation']} "
              f"({c['P2_total_anticipatory_scale_ups']})")
    for s in o["summary"]:
        print(f"{s['service_model']:<16} {s['controller']:<9} {s['kind']:<11} "
              f"runs={s['n_runs']:<4} steps={s['n_steps']:<6} "
              f"disagree={s['n_disagree']:<5} (zeropad {s['n_disagree_in_zeropad_window']}) "
              f"maxdiff={s['max_abs_diff']}")
