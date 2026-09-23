"""Env cards, rendered from a live env rather than written by hand.

A card that is typed into a markdown file drifts from the code the first time a
weight changes.  `env_card_markdown` builds the card by constructing the env and
reading `KubeGymEnv.spec_card()`, so every number in `env_cards.md` came out of
the object a user will actually instantiate.  Regenerate with:

    python -m kubegym.gym.cards > env_cards.md

The inherited-bias section is copied from INTERFACE.md section 8 verbatim rather
than paraphrased: a benchmark paper using this core has to repeat those biases,
and a paraphrase is how a direction of bias gets softened by accident.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

#: INTERFACE.md section 8, "Direction of every known bias", transcribed. Each
#: row is (choice, direction, note). Any card built on a config whose fields
#: these describe inherits every row.
INHERITED_BIASES: List[Dict[str, str]] = [
    {"choice": "decode step time extrapolated affinely above batch 8",
     "direction": "optimistic",
     "note": "real engines saturate; makes under-provisioning look cheap. flagged_optimistic",
     "applies_to": "llm_serving"},
    {"choice": "prefill_tokens_per_s = 6000, never measured",
     "direction": "unknown sign",
     "note": "every TTFT number is placeholder-derived",
     "applies_to": "llm_serving"},
    {"choice": "chunked prefill not modelled", "direction": "mixed",
     "note": "overstates the inter-token spike on admission; understates long-prompt TTFT "
             "under load",
     "applies_to": "llm_serving"},
    {"choice": "teardown not billed (0.54 s, weak evidence n=2)", "direction": "optimistic",
     "note": "< 1 % of a scale-down cycle", "applies_to": "both"},
    {"choice": "parallel bring-up (source simulator; not the default here)",
     "direction": "optimistic",
     "note": "scale-up looks faster than the verified sequential path",
     "applies_to": "request_service"},
    {"choice": "metric staleness modelled as zero",
     "direction": "optimistic about the controller's information",
     "note": "real scrape interval is 15 s", "applies_to": "both"},
    {"choice": "rs_contention_slowdown = 0.0", "direction": "optimistic",
     "note": "a loaded replica is as fast as an idle one", "applies_to": "request_service"},
    {"choice": "no load shedding / drops in request_service",
     "direction": "pessimistic on latency, optimistic on success rate",
     "note": "queueing delay is the only symptom of overload",
     "applies_to": "request_service"},
    {"choice": "output-length distribution is a placeholder lognormal",
     "direction": "sets the difficulty of the whole problem",
     "note": "dominant uncalibrated input", "applies_to": "llm_serving"},
]

#: The two circularity warnings INTERFACE.md section 8 requires every paper
#: using this core to repeat.
CIRCULARITY_WARNINGS = [
    "A controller whose belief is over the SAME distribution the simulator samples from is "
    "handed a perfectly specified prior, so a belief-vs-point gap is close to guaranteed by "
    "construction and its magnitude is uninformative.",
    "In the shipped traces thinking_flip_prob = 0.0, so the class signal is a NOISELESS proxy "
    "for the length class; set class_predictor_accuracy < 1.0 to inject predictor error.",
]


def _table(rows: List[List[str]], header: List[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def env_card_markdown(gym_id: str, *, dead_elements: Optional[List[str]] = None) -> str:
    """One env card, as markdown, built from a live env."""
    import gymnasium

    from . import tasks  # noqa: F401  (ensures registration)

    env = gymnasium.make(gym_id, disable_env_checker=True).unwrapped
    c = env.spec_card()
    L: List[str] = []
    A = L.append

    A(f"## `{c['id']}`")
    A("")
    A(f"**Calibration status: `calibrated = {str(c['calibrated']).lower()}`.** "
      f"Blocked by `{'`, `'.join(c['uncalibrated_required_fields'])}`. "
      "No return, reward or cost from this task is an empirical result.")
    A("")
    A(_table([
        ["service model", f"`{c['service_model']}`"],
        ["config", f"`{c['config']}`"],
        ["workload", f"`{c['workload']}`"],
        ["episode length", f"{c['n_control_steps']} control steps "
                           f"x {c['control_interval_s']} s = {c['horizon_s']} s"],
        ["distinct episodes", str(c["n_episodes"])],
        ["initial replicas `k0`", str(c["k0"])],
        ["replica range", f"[{c['n_replicas_min']}, {c['n_replicas_max']}]"
                          + ("  (ceiling is a MEASURED physical limit of the testbed)"
                             if c["n_replicas_max_is_measured_physical_limit"]
                             else "  (ceiling is a placeholder, not a measured limit)")],
        ["action space", f"`{c['action_space']}`  ({c['action_mode']} mode)"],
        ["observation space", f"`{c['observation_space']}`"],
        ["post-horizon drain cap", f"{c['drain_cap_s']} s"],
        ["drain charged to reward", str(c["include_drain_in_reward"])],
    ], ["field", "value"]))
    A("")

    A("### Workload")
    A("")
    A(c["workload_description"])
    A("")
    man = c["workload_manifest"]
    A(_table([[f"`{k}`", f"`{v}`"] for k, v in sorted(man.items())], ["manifest key", "value"]))
    A("")

    A("### Action space")
    A("")
    A(f"`{c['action_space']}`, mode `{c['action_mode']}`.")
    A("")
    A(_table([[str(i), m] for i, m in enumerate(c["action_meanings"])], ["action", "meaning"]))
    A("")
    A(f"Targets are clamped to `[{c['n_replicas_min']}, {c['n_replicas_max']}]` by the core. "
      + ("Exceeding the ceiling requires `allow_hypothetical_replicas=True`, which sets "
         "`hypothetical=True` on the pool and taints every result row; no registered task "
         "enables it."
         if c["n_replicas_max_is_measured_physical_limit"] else
         "The ceiling here is a placeholder rather than a measured limit, so a target near it "
         "is not automatically hypothetical."))
    A("")

    A("### Observation vector")
    A("")
    A(f"{c['observation_dim']} elements, `float32`, each scaled then clipped into "
      f"`[0, {c['observation_norm']['clip']}]`. History is stacked oldest-first; `[t-0]` is "
      "the most recent scrape. Every element but the last is "
      "`kubegym.core.state.FEATURES[...](ClusterState)`, so the policy sees no more than a "
      "hand-written controller reading the same scrape.")
    A("")
    A("Normalisation divisors are a-priori constants -- config fields or documented design "
      "constants of the task. None is derived from the episode in progress, because a "
      "statistic of the episode is information from the future and would break the fairness "
      "guarantee silently.")
    A("")
    rows = []
    for i, e in enumerate(c["observation_elements"]):
        note = ""
        if dead_elements and e["name"].split("[")[0] in dead_elements:
            note = " **identically zero on this task, see caveats**"
        rows.append([str(i), f"`{e['name']}`", e["scale_kind"], f"{e['divisor']:g}",
                     (e["source"] if e["feature"] == "(env)" else "ClusterState") + note])
    A(_table(rows, ["idx", "element", "scale kind", "divisor", "source"]))
    A("")
    A(_table([[f"`{k}`", f"{v:g}"] for k, v in sorted(c["observation_norm"].items())],
             ["scale", "value"]))
    A("")

    A("### Reward")
    A("")
    cost = c["cost"]
    A(f"`{cost['sign_convention']}`. Cost terms, all non-negative:")
    A("")
    A(_table([
        ["`slo`", cost["units"]["slo"], f"{cost['weights']['slo_per_violating_request']:g}"],
        ["`resource`", cost["units"]["resource"],
         f"{cost['weights']['resource_per_replica_second']:.6g}"],
        ["`churn`", cost["units"]["churn"],
         f"{cost['weights']['churn_per_replica_changed']:g}"],
    ], ["term", "unit", "canonical weight"]))
    A("")
    A("Charged once at the final step, in addition: `slo_drain` (violations completed during "
      "the post-horizon drain), `resource_drain` (replica-seconds billed during the drain), "
      "`slo_unfinished` (requests that never completed). Same weights; "
      f"`include_drain_in_reward = {c['include_drain_in_reward']}`.")
    A("")
    A(f"**SLO profile `{cost['slo_profile']}`.** {cost['slo_profile_description']}")
    A("")
    A(_table([[f"`{k}`", f"{v:g}"] for k, v in sorted(cost["slo_targets"].items())],
             ["SLO target", "value (s)"]))
    A("")
    A(f"SLO targets come from the config field `slo`, `calibrated = "
      f"{str(cost['slo_targets_calibrated']).lower()}`. The **weights are a design choice, not "
      "a measurement**, and are not covered by the config's claims gate; they are recorded in "
      "`unverified_gym.md`. Every term is computable from router and orchestrator telemetry a "
      "real deployment has -- the reward never reads a request's ground-truth work "
      f"(`reads_ground_truth_demand = {cost['reads_ground_truth_demand']}`).")
    A("")

    A("### Seeds and splits")
    A("")
    A("`reset(seed=s)` selects `(episode, work_seed)` deterministically as "
      "`pairs[s % len(pairs)]` within the env's split, so a rollout is reproducible from the "
      "Gym API alone. Splits are disjoint in work seed.")
    A("")
    A(_table([[f"`{s}`", f"{len(v)} seeds: {v[0]}..{v[-1]}", str(c["split_pairs"][s])]
              for s, v in sorted(c["splits"].items())],
             ["split", "work seeds", "(episode, seed) pairs"]))
    A("")

    A("### Calibration and inherited biases")
    A("")
    A(f"`{c['config']}`: {c['calibration_note']}")
    A("")
    if c["flagged_optimistic_fields"]:
        A(f"Fields flagged as biasing results favourably: "
          f"`{'`, `'.join(c['flagged_optimistic_fields'])}`.")
        A("")
    A("Inherited from INTERFACE.md section 8 (transcribed, not paraphrased):")
    A("")
    applic = {c["service_model"], "both"}
    A(_table([[b["choice"], f"**{b['direction']}**", b["note"]]
              for b in INHERITED_BIASES if b["applies_to"] in applic],
             ["choice", "direction", "note"]))
    A("")
    A("Known infidelities declared by the service model itself:")
    A("")
    for s in c["service_model_provenance"].get("known_infidelities", []):
        A(f"- {s}")
    A("")
    if c["service_model"] == "llm_serving":
        A("Circularity warnings any paper using this core must repeat:")
        A("")
        for s in CIRCULARITY_WARNINGS:
            A(f"- {s}")
        A("")

    A("### Task caveats")
    A("")
    for n in c["notes"]:
        A(f"- {n}")
    A("")
    return "\n".join(L)


def all_env_cards_markdown(dead_elements_by_task: Optional[Dict[str, List[str]]] = None) -> str:
    """Every registered card, with a summary table."""
    from .tasks import list_tasks

    rows = list_tasks()
    L = ["# KubeGym env cards", "",
         "Generated from the live environments by `python -m kubegym.gym.cards`; every number "
         "below was read off the object `gymnasium.make(<id>)` returns.", "",
         "**Every task here is `calibrated = false`.** Both shipped configs have at least one "
         "`required_for_claims` field that is a labelled placeholder, so no return, reward or "
         "cost on this page is an empirical result. The flag is on every `info` dict, every "
         "`list_tasks()` row, and every stamped episode row.", "",
         "## Registered tasks", ""]
    L.append(_table([[f"`{r['id']}`", r["service_model"], r["workload"], r["action_mode"],
                      str(r["n_control_steps"]), f"[{r['n_replicas_min']}, "
                      f"{r['n_replicas_max']}]", str(r["obs_dim"]),
                      str(r["calibrated"]).lower()]
                     for r in rows],
                    ["id", "service model", "workload", "action mode", "steps", "k range",
                     "obs dim", "calibrated"]))
    L.append("")
    dead = dead_elements_by_task or {}
    for r in rows:
        L.append(env_card_markdown(r["id"], dead_elements=dead.get(r["task"])))
    return "\n".join(L)


if __name__ == "__main__":
    sys.stdout.write(all_env_cards_markdown())
