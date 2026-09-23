"""A worked smoke run: the loop closes, and what it costs per 1000 env steps.

    python -m kubegym.gym.smoke                     # default task
    python -m kubegym.gym.smoke --task KubeGym/RequestService-Poisson-Delta-v0

Two things this establishes and one it deliberately does not.

Establishes: (1) a random policy and a trivial fixed policy both run end to end
through the Gymnasium API with a full cost decomposition per episode; (2) the
wall-clock cost of an env step, split into the core's simulation time and the
wrapper's overhead, measured back-to-back on the same workload in one process.

Does NOT establish: anything about which policy is better.  Three episodes of a
placeholder-calibrated 300 s fixture is a smoke test.  Every number it prints
carries `calibrated: false`.

ON COMPARING THE TIMING TO THE CORE'S PUBLISHED FIGURES
-------------------------------------------------------
`parity_report.md` reports 3.95 wall-seconds per simulated hour and 16.2 ms per
control step, measured on a fixture of 4680 requests at k=4 over one simulated
hour.  The tasks here are the 300 s reference traces: ~225 requests, mostly at
1-2 replicas.  A per-step time from this script is therefore NOT comparable to
16.2 ms -- the offered load per step is roughly an order of magnitude lower, and
the simulator's cost is dominated by per-token work.  The comparison that is
valid is the one this script actually makes: the same episodes, with and without
the wrapper, in the same process, which isolates what the RL layer adds.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_TASK = "KubeGym/LLMServing-S1-Delta-v0"


def _make(task_id: str):
    import gymnasium
    import kubegym.gym  # noqa: F401  (registers the builtin tasks)
    return gymnasium.make(task_id, disable_env_checker=True).unwrapped


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------
def random_policy(env, rng: np.random.Generator):
    def act(obs, info):
        return int(rng.integers(env.action_space.n))
    return act


def fixed_target_policy(env, k: int):
    """Hold a constant replica target, whatever the action mode.

    In `delta` mode that means moving toward `k` by the largest available
    increment and then holding, which is the trivial baseline a wrapper must be
    able to express. It reads only `info["target"]`, i.e. its own actuation
    state.
    """
    deltas = list(env.delta_set) if env.action_mode == "delta" else None

    def act(obs, info):
        cur = int(info["target"])
        if deltas is None:
            return int(min(max(k, env.k_min), env.k_max) - env.k_min)
        want = k - cur
        # the increment closest to `want` without overshooting its sign
        best = min(range(len(deltas)),
                   key=lambda i: (abs(deltas[i] - want), abs(deltas[i])))
        return best
    return act


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------
def run_episode(env, policy, seed: int) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    ret = 0.0
    n = 0
    targets: List[int] = []
    while True:
        a = policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(a)
        ret += reward
        n += 1
        targets.append(info["target"])
        if terminated or truncated:
            break
    row = dict(info["episode_cost"])
    row.update({
        "steps": n,
        "return_checked": ret,
        "targets": targets,
        "n_requests": info["episode_summary"]["n_requests"],
        "n_done": info["episode_summary"]["n_done"],
        "p95_latency_s": info["episode_summary"]["p95_latency_s"],
        "replica_seconds": info["episode_summary"]["replica_seconds"],
        "preemptions_total": info["episode_summary"]["preemptions_total"],
        "truncated_drain": info["episode_summary"]["truncated_drain"],
        "terminated": terminated,
        "truncated": truncated,
    })
    assert abs(row["return"] - ret) < 1e-9, (row["return"], ret)
    return row


def _fmt_episode(tag: str, r: Dict[str, Any]) -> str:
    return (f"  {tag:22s} return={r['return']:9.3f}  "
            f"slo={r['cost_slo']:7.3f} resource={r['cost_resource']:7.3f} "
            f"churn={r['cost_churn']:6.3f} "
            f"drain(res={r['cost_resource_drain']:6.3f} slo={r['cost_slo_drain']:5.3f} "
            f"unfin={r['cost_slo_unfinished']:5.3f})  "
            f"viol={r['slo_violations_total']:4d} k={min(r['targets'])}-{max(r['targets'])}  "
            f"calibrated={r['calibrated']}")


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def time_env_steps(task_id: str, n_steps: int, seed0: int = 0) -> Dict[str, Any]:
    """Wall time for `n_steps` env steps, including resets and end-of-episode drains.

    Resets and drains are counted in on purpose: an RL training loop pays them,
    and at 20 control steps per episode a reset lands every 20 steps, so
    excluding it would understate the real cost by the episode-build time.
    """
    env = _make(task_id)
    rng = np.random.default_rng(0)
    policy = random_policy(env, rng)
    obs, info = env.reset(seed=seed0)
    n_resets = 1
    t0 = time.perf_counter()
    for i in range(n_steps):
        obs, _, terminated, truncated, info = env.step(policy(obs, info))
        if terminated or truncated:
            obs, info = env.reset(seed=seed0 + n_resets)
            n_resets += 1
    wall = time.perf_counter() - t0
    return {"n_steps": n_steps, "n_resets": n_resets, "wall_s": wall,
            "s_per_1000_steps": 1000.0 * wall / n_steps,
            "ms_per_step": 1000.0 * wall / n_steps}


def time_bare_core(task_id: str, n_steps: int, seed0: int = 0) -> Dict[str, Any]:
    """The same episodes driven straight on `Simulator`, no Gym layer.

    Same action sequence, same seeds, same drain. The difference against
    `time_env_steps` is what the wrapper adds: observation assembly, cost
    scoring and `info` construction.
    """
    from ..core.engine import Simulator

    env = _make(task_id)                  # reuse the task's own factories
    spec = env.task_spec
    cfg = spec.load_config()
    sim = Simulator(spec.service_model_factory(cfg), spec.workload_factory(), cfg, k0=spec.k0)
    pairs = spec.pairs(env.split)
    rng = np.random.default_rng(0)
    n_ctrl = env.n_control_steps
    drain_cap = spec.horizon_s * spec.drain_cap_multiple

    ep, ws = pairs[seed0 % len(pairs)]
    sim.reset(episode=ep, seed=ws, k0=spec.k0)
    n_resets = 1
    i_in_ep = 0
    t0 = time.perf_counter()
    for i in range(n_steps):
        a = int(rng.integers(env.action_space.n))
        if env.action_mode == "delta":
            sim.set_target_replicas(sim.target + env.delta_set[a])
        else:
            sim.set_target_replicas(env.k_min + a)
        sim.advance_control_interval()
        i_in_ep += 1
        if i_in_ep >= n_ctrl:
            sim.drain_all(sim.t + drain_cap)
            ep, ws = pairs[(seed0 + n_resets) % len(pairs)]
            sim.reset(episode=ep, seed=ws, k0=spec.k0)
            n_resets += 1
            i_in_ep = 0
    wall = time.perf_counter() - t0
    return {"n_steps": n_steps, "n_resets": n_resets, "wall_s": wall,
            "s_per_1000_steps": 1000.0 * wall / n_steps,
            "ms_per_step": 1000.0 * wall / n_steps}


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--timing-steps", type=int, default=1000)
    ap.add_argument("--fixed-k", type=int, default=2)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)

    env = _make(args.task)
    print(f"task            {args.task}")
    print(f"config          {env.cfg.provenance()['config_path']}  "
          f"calibrated={env.cfg.calibrated}")
    print(f"blocked by      {', '.join(env.cfg.uncalibrated_required)}")
    print(f"obs             {env.observation_space}  ({len(env.obs_names)} elements)")
    print(f"action          {env.action_space}  mode={env.action_mode}  "
          f"k in [{env.k_min}, {env.k_max}]")
    print(f"episode         {env.n_control_steps} control steps x "
          f"{env.control_interval_s} s = {env.horizon_s} s")
    print(f"weights         {env.cost_model.weights.to_dict()}")
    print(f"slo targets     {env.cost_model.slo}  "
          f"(calibrated={env.cfg.is_calibrated('slo')})")
    print()

    out: Dict[str, Any] = {"task": args.task, "calibrated": bool(env.cfg.calibrated),
                           "uncalibrated_required_fields": env.cfg.uncalibrated_required}

    rng = np.random.default_rng(12345)
    print("random policy")
    rnd = [run_episode(env, random_policy(env, rng), seed=s) for s in range(args.episodes)]
    for i, r in enumerate(rnd):
        print(_fmt_episode(f"episode seed={i}", r))
    print(f"  mean return {np.mean([r['return'] for r in rnd]):.3f}")

    print(f"\nfixed policy (hold k={args.fixed_k})")
    fix = [run_episode(env, fixed_target_policy(env, args.fixed_k), seed=s)
           for s in range(args.episodes)]
    for i, r in enumerate(fix):
        print(_fmt_episode(f"episode seed={i}", r))
    print(f"  mean return {np.mean([r['return'] for r in fix]):.3f}")

    out["random_policy"] = [{k: v for k, v in r.items() if k != "targets"} for r in rnd]
    out["fixed_policy"] = [{k: v for k, v in r.items() if k != "targets"} for r in fix]
    out["fixed_k"] = args.fixed_k

    print(f"\ntiming, {args.timing_steps} env steps (resets and end-of-episode drains included)")
    wrapped = time_env_steps(args.task, args.timing_steps)
    bare = time_bare_core(args.task, args.timing_steps)
    overhead_s = wrapped["s_per_1000_steps"] - bare["s_per_1000_steps"]
    print(f"  Gym env            {wrapped['s_per_1000_steps']:7.3f} s / 1000 steps "
          f"({wrapped['ms_per_step']:.3f} ms/step, {wrapped['n_resets']} resets)")
    print(f"  bare Simulator     {bare['s_per_1000_steps']:7.3f} s / 1000 steps "
          f"({bare['ms_per_step']:.3f} ms/step)")
    print(f"  wrapper overhead   {overhead_s:7.3f} s / 1000 steps "
          f"({100.0 * overhead_s / bare['s_per_1000_steps']:.1f}% of core time)")
    out["timing"] = {"wrapped": wrapped, "bare_core": bare,
                     "wrapper_overhead_s_per_1000_steps": overhead_s,
                     "wrapper_overhead_pct_of_core": (
                         100.0 * overhead_s / bare["s_per_1000_steps"])}

    print("\nNOTE every number above is calibrated=false and is not an empirical result.")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"wrote {args.json_out}")
    return out


if __name__ == "__main__":
    main()
