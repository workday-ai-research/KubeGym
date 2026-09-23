"""A guard against a non-terminating event loop in the shipped core engine.

THE DEFECT
----------
`kubegym.models.request_service.RequestServiceEngine` completes a request when

    r.progress >= r.demand - 1e-12                      # ABSOLUTE tolerance

and schedules the next service event at

    next_step_t = t + min(demand - progress for r in running) / rate

`progress` is accumulated by repeated `+=` over tens of thousands of events, so
its floating-point residual against `demand` can settle *above* the 1e-12
completion tolerance while being far below the spacing of the double-precision
time axis. Measured instance, `azure_replay` test episode 4 (trace
`azure-009f72f7-d11-m0840-s200004`, 23,286 requests) under `static-k1`:

    t                = 16391.100897071403
    demand           = 0.40464384787069557
    progress         = 0.40464384786901064
    residual         = 1.6849299733223688e-12      >  1e-12, so NOT completed
    next_step_t      = 16391.100897071403          == t   (spacing at 1e4 is ~3.6e-12)
    => dt            = 0.0

The event loop in `Simulator.step_to` then advances time by exactly zero on
every iteration and never returns. `Simulator.drain_all`'s own no-progress guard
(`if self.t <= before + 1e-12: break`) cannot fire, because control never comes
back to it. The process spins until it is killed -- which is what happened to the
first `azure_replay` evaluation run, silently, with no traceback.

The trigger needs three things at once: a long post-horizon drain (so `t` grows
to 10^4 and the time-axis spacing exceeds 1e-12), many service events per request
slot (so `progress` accumulates rounding), and a large backlog (so the drain is
long). Under-provisioned policies on heavy traces hit all three, so it is exactly
the trivial `Static-k` reference at small `k` that cannot be evaluated without
this guard -- the defect selectively removes the cheapest end of the reference
front, which is the end a benchmark most needs.

THE GUARD
---------
One line, applied to `_reschedule` only:

    next_step_t = t + max(rem / rate, MIN_QUANTUM_S)

with `MIN_QUANTUM_S = 1e-9`, which strictly exceeds the double spacing anywhere
in the simulated range (`t < 4 x 10^4 s`, spacing < 1e-11). Time therefore
advances, the near-complete request receives `1e-9 * rate` more work on the next
event -- vastly more than any residual that can stall the loop -- and completes.

WHY THIS IS SAFE FOR THE COMPARISON
-----------------------------------
1. It is installed **once, globally**, before any env is built, so every method
   -- analytic and learned, tuning, training and test -- runs under identical
   physics. It cannot favour one method over another.
2. It is inert on any episode that already terminated: the floor binds only when
   `rem / rate < 1e-9`, i.e. when less than a nanosecond of service work is
   outstanding. `verify_inert()` checks this empirically by recomputing the
   dev objectives of the tuned configurations, which were measured *before* the
   guard existed, and asserting they are bit-identical.
3. It does not change the completion rule, the cost model, the queueing
   discipline, or any billed quantity beyond `<= 1e-9 s` of simulated time per
   affected event.

It is a workaround in the baselines package, not a change to the environment.
The fix belongs in the core, by making the completion tolerance relative to
`demand` rather than absolute. Until that lands, this module is the authoritative
record of the defect: `INFO` is serialised into every run's metadata, and the
write-up (`BASELINES.md` section on environment defects found, and
`unverified_baselines.md`) must carry it forward.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

MIN_QUANTUM_S = 1e-9
_INSTALLED = False
INFO: Dict[str, Any] = {
    "installed": False,
    "min_quantum_s": MIN_QUANTUM_S,
    "target": "kubegym.models.request_service.RequestServiceEngine._reschedule",
    "reason": ("shipped engine can schedule next_step_t == t when a request's "
               "floating-point residual exceeds the 1e-12 absolute completion "
               "tolerance but is below the double spacing of the time axis, which "
               "makes Simulator.step_to spin forever"),
}


def install() -> Dict[str, Any]:
    """Idempotently install the guard.  Safe to call from every entry point."""
    global _INSTALLED
    if _INSTALLED:
        return INFO
    from kubegym.models.request_service import RequestServiceEngine as RSE

    INF = float("inf")

    def _reschedule(self, t: float) -> None:
        if not self.running:
            self.next_step_t = INF
            return
        rate = self.rate()
        rem = min(max(0.0, r.demand - r.progress) for r in self.running)
        if rate <= 0:
            self.next_step_t = INF
            return
        self.next_step_t = t + max(rem / rate, MIN_QUANTUM_S)

    RSE._reschedule = _reschedule
    _INSTALLED = True
    INFO["installed"] = True
    return INFO


def verify_inert(out_dir: str = "out", families: Optional[List[str]] = None
                 ) -> Dict[str, Any]:
    """Assert the guard changes nothing that was already measured without it.

    Recomputes each family's dev objective for every tuned configuration and
    compares against the value recorded in `tuned_<family>.json`, which was
    produced before the guard existed.  Bit-identical agreement is the evidence
    that the floor never binds on an episode that terminated.
    """
    import glob
    import json
    import os

    install()
    from .gym_controllers import REGISTRY
    from .obsmap import ObsView
    from .runner import make_env, objective, rollout
    from .tune import dev_tuning_episodes
    from .runner import load_manifest

    manifest = load_manifest()
    checks: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(out_dir, "tuned_*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        fam = d["family"]
        if families and fam not in families:
            continue
        eps = d["dev_tuning_episodes"]
        assert eps == dev_tuning_episodes(fam, manifest), (
            f"{fam}: dev tuning episode set changed since tuning; the comparison "
            "would not be like-for-like")
        env = make_env(fam, "dev", "absolute")
        u = env.unwrapped
        view = ObsView(u.task_spec)
        ws = int(u.task_spec.split_work_seeds["dev"][0])
        for m, md in d["methods"].items():
            ctrl = REGISTRY[m](view, u.cfg, action_mode="absolute",
                               **md["selected_params"])
            got = objective([rollout(env, ctrl, ep, ws) for ep in eps])
            want = float(md["selected_objective_dev"])
            checks.append({"family": fam, "method": m, "recorded": want,
                           "recomputed": got, "abs_diff": abs(got - want),
                           "bit_identical": got == want})
        env.close()
    n_bad = sum(1 for c in checks if not c["bit_identical"])
    worst = max((c["abs_diff"] for c in checks), default=0.0)
    return {"n_checked": len(checks), "n_not_bit_identical": n_bad,
            "max_abs_diff": worst, "checks": checks,
            "inert": n_bad == 0}
