#!/usr/bin/env python3
"""Parity + timing harness: kubegym's llm_serving vs the source sim_cluster.py.

Runs the SAME trace and the SAME seed through both implementations under the
SAME open-loop scale schedule, then compares per-request completion times,
first-token times, replica-seconds, preemption counts and generated tokens.

An open-loop schedule (not a controller) is used for the primary comparison so
that no controller code sits in the loop: any divergence is attributable to the
physics port, not to a controller reading a slightly different state.  A second
comparison drives BOTH implementations with the same `controllers.HPA` instance
class, to check that the state surface a controller sees is equivalent too.

`kubegym` is configured with `replica_bringup="parallel"` here, which reproduces
the source simulator's bring-up exactly.  The shipped default is "sequential"
(the verified real-cluster path); see INTERFACE.md.

Usage:  python parity_harness.py [--out parity_results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# source testbed
import sim_cluster as ORIG                                    # noqa: E402
import controllers as ORIG_CTRL                               # noqa: E402
from length_model import load_length_model as orig_load_lm     # noqa: E402

# kubegym
from kubegym import ProvenancedConfig, Simulator, config_path  # noqa: E402
from kubegym.models.length_model import load_length_model      # noqa: E402
from kubegym.models.llm_serving import LLMServingModel, LLMTraceSource  # noqa: E402

TRACES = ["S1_dev", "S3_dev", "S5_dev"]
SEEDS = [1, 2]
HORIZON_S = 900.0
CONTROL_S = 15.0
DRAIN_CAP_S = 20000.0

#: Deterministic open-loop schedules, in control-interval steps.
#: `thrash` exercises cold start, multi-step scale-up, drain, revival of a
#: draining replica, and a scale-down that leaves work behind.
#: `starve` pins one replica so KV fills and the recompute-preemption path runs;
#: without it, parity would never touch preemption (the other schedules provision
#: enough capacity that no eviction ever happens).
SCHEDULES = {
    "thrash": {0: 1, 4: 3, 12: 4, 24: 1, 30: 2, 40: 4, 50: 1},
    "starve": {0: 1},
}


def schedule_at(schedule: Dict[int, int], step: int):
    return schedule.get(step)


# ---------------------------------------------------------------------------
def make_bench_trace(src_path: str, out_path: str, *, tiles: int, tile_s: float) -> str:
    """Tile a 300 s reference trace into a one-hour timing fixture."""
    recs = [json.loads(l) for l in open(src_path) if l.strip()]
    out, seq = [], 0
    for k in range(tiles):
        for r in recs:
            q = dict(r)
            q["t_arrival"] = float(r["t_arrival"]) + k * tile_s
            q["seq"] = seq
            seq += 1
            out.append(q)
    out.sort(key=lambda x: x["t_arrival"])
    for i, q in enumerate(out):
        q["seq"] = i
    with open(out_path, "w") as fh:
        for q in out:
            fh.write(json.dumps(q, sort_keys=True) + "\n")
    return out_path


def run_original(trace_path: str, seed: int, cfg, lm, *, controller=None,
                 schedule=None, horizon_s: float = HORIZON_S, k0: int = 1) -> Dict[str, Any]:
    schedule = SCHEDULES["thrash"] if schedule is None else schedule
    trace = ORIG.load_trace(trace_path, lm, seed=seed)
    sim = ORIG.SimCluster(trace, cfg, k0=k0, seed=seed)
    t0 = time.perf_counter()
    hist: List[Any] = [sim.scrape(0.0)]
    n_steps = int(round(horizon_s / CONTROL_S))
    for i in range(n_steps):
        if controller is not None:
            k = controller.decide(hist, sim.t)
            if k is not None:
                sim.set_target_replicas(int(k), sim.t)
        else:
            k = schedule_at(schedule, i)
            if k is not None:
                sim.set_target_replicas(k, sim.t)
        sim.step_to(sim.t + CONTROL_S)
        hist.append(sim.scrape(sim.t))
    sim.drain_all(DRAIN_CAP_S)
    wall = time.perf_counter() - t0
    return collect(trace, sim.total_replica_seconds(sim.t), sim.t, wall,
                   [dict(e) for e in sim.events])


def run_kubegym(trace_path: str, seed: int, cfg, lm, *, controller=None,
                schedule=None, horizon_s: float = HORIZON_S, k0: int = 1) -> Dict[str, Any]:
    schedule = SCHEDULES["thrash"] if schedule is None else schedule
    model = LLMServingModel(cfg, length_model=lm)
    source = LLMTraceSource([trace_path], length_model=lm)
    sim = Simulator(model, source, cfg, k0=k0, replica_bringup="parallel")
    t0 = time.perf_counter()
    sim.reset(episode=0, seed=seed)
    n_steps = int(round(horizon_s / CONTROL_S))
    for i in range(n_steps):
        if controller is not None:
            k = controller.decide(sim.hist, sim.t)
            if k is not None:
                sim.set_target_replicas(int(k), sim.t)
        else:
            k = schedule_at(schedule, i)
            if k is not None:
                sim.set_target_replicas(k, sim.t)
        sim.advance_control_interval()
    sim.drain_all(DRAIN_CAP_S)
    wall = time.perf_counter() - t0
    return collect(sim.requests, sim.total_replica_seconds(sim.t), sim.t, wall,
                   [dict(e) for e in sim.pool.events])


def collect(reqs, replica_seconds: float, t_end: float, wall_s: float,
            events: List[Dict]) -> Dict[str, Any]:
    by_seq = {r.seq: r for r in reqs}
    return {
        "n_requests": len(reqs),
        "n_done": sum(1 for r in reqs if r.done),
        "t_done": {s: by_seq[s].t_done for s in sorted(by_seq)},
        "t_first_token": {s: by_seq[s].t_first_token for s in sorted(by_seq)},
        "demand": {s: float(by_seq[s].output_tokens_true) for s in sorted(by_seq)},
        "generated": sum(int(r.generated) for r in reqs),
        "preemptions": sum(int(r.n_preemptions) for r in reqs),
        "replica_seconds": float(replica_seconds),
        "t_end": float(t_end),
        "wall_s": float(wall_s),
        "events": events,
    }


def compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    seqs = sorted(set(a["t_done"]) & set(b["t_done"]))
    d_done, d_ttft = [], []
    n_none_mismatch = 0
    for s in seqs:
        x, y = a["t_done"][s], b["t_done"][s]
        if (x is None) != (y is None):
            n_none_mismatch += 1
        elif x is not None:
            d_done.append(abs(x - y))
        u, v = a["t_first_token"][s], b["t_first_token"][s]
        if (u is None) != (v is None):
            n_none_mismatch += 1
        elif u is not None:
            d_ttft.append(abs(u - v))
    return {
        "n_requests_a": a["n_requests"], "n_requests_b": b["n_requests"],
        "demand_identical": a["demand"] == b["demand"],
        "n_done_a": a["n_done"], "n_done_b": b["n_done"],
        "max_abs_t_done_diff_s": max(d_done) if d_done else 0.0,
        "mean_abs_t_done_diff_s": (sum(d_done) / len(d_done)) if d_done else 0.0,
        "n_t_done_nonzero_diff": sum(1 for x in d_done if x > 0.0),
        "max_abs_t_first_token_diff_s": max(d_ttft) if d_ttft else 0.0,
        "n_completion_none_mismatch": n_none_mismatch,
        "generated_a": a["generated"], "generated_b": b["generated"],
        "preemptions_a": a["preemptions"], "preemptions_b": b["preemptions"],
        "replica_seconds_a": a["replica_seconds"], "replica_seconds_b": b["replica_seconds"],
        "replica_seconds_abs_diff": abs(a["replica_seconds"] - b["replica_seconds"]),
        "t_end_abs_diff": abs(a["t_end"] - b["t_end"]),
        "events_identical": a["events"] == b["events"],
        "wall_s_a": a["wall_s"], "wall_s_b": b["wall_s"],
    }


def exact(c: Dict[str, Any]) -> bool:
    return (c["demand_identical"] and c["n_done_a"] == c["n_done_b"]
            and c["max_abs_t_done_diff_s"] == 0.0
            and c["max_abs_t_first_token_diff_s"] == 0.0
            and c["n_completion_none_mismatch"] == 0
            and c["generated_a"] == c["generated_b"]
            and c["preemptions_a"] == c["preemptions_b"]
            and c["replica_seconds_abs_diff"] == 0.0
            and c["events_identical"])


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="parity_results.json")
    ap.add_argument("--scenarios-dir", default=os.path.join(SRC, "scenarios"))
    a = ap.parse_args()

    orig_cfg = ORIG.load_config(os.path.join(SRC, "cluster_config.json"))
    kg_cfg = ProvenancedConfig.load(config_path("llm_serving_l40s.json"))
    lm_o = orig_load_lm("measured_lengths.json")
    lm_k = load_length_model("measured_lengths.json")

    results = {"schedules": SCHEDULES, "horizon_s": HORIZON_S,
               "control_interval_s": CONTROL_S, "traces": TRACES, "seeds": SEEDS,
               "original_config": orig_cfg.provenance(),
               "kubegym_config": kg_cfg.provenance(),
               "cases": [], "controller_cases": []}

    for sname, sched in SCHEDULES.items():
        for tr in TRACES:
            path = os.path.join(a.scenarios_dir, f"{tr}.jsonl")
            for seed in SEEDS:
                o = run_original(path, seed, orig_cfg, lm_o, schedule=sched)
                k = run_kubegym(path, seed, kg_cfg, lm_k, schedule=sched)
                c = compare(o, k)
                c.update({"trace": tr, "seed": seed, "driver": f"open_loop:{sname}",
                          "exact": exact(c)})
                results["cases"].append(c)
                print(f"[{sname}] {tr} seed={seed} exact={c['exact']} "
                      f"max|dt_done|={c['max_abs_t_done_diff_s']:.3e}s "
                      f"rs={c['replica_seconds_a']:.2f}/{c['replica_seconds_b']:.2f} "
                      f"preempt={c['preemptions_a']}/{c['preemptions_b']}")

    # controller-driven check: same controller class, same params, both backends
    for tr in ["S3_dev"]:
        path = os.path.join(a.scenarios_dir, f"{tr}.jsonl")
        for seed in SEEDS:
            co = ORIG_CTRL.make("hpa", orig_cfg, metric="kv", target=0.6)
            ck = ORIG_CTRL.make("hpa", orig_cfg, metric="kv", target=0.6)
            o = run_original(path, seed, orig_cfg, lm_o, controller=co)
            k = run_kubegym(path, seed, kg_cfg, lm_k, controller=ck)
            c = compare(o, k)
            c.update({"trace": tr, "seed": seed, "driver": "controllers.HPA(kv,0.6)",
                      "exact": exact(c)})
            results["controller_cases"].append(c)
            print(f"[hpa] {tr} seed={seed} exact={c['exact']} "
                  f"max|dt_done|={c['max_abs_t_done_diff_s']:.3e}s "
                  f"rs={c['replica_seconds_a']:.2f}/{c['replica_seconds_b']:.2f}")

    # ---- timing at the reference load -------------------------------
    # The stated budget is ~5.5 s wall per SIMULATED HOUR at 4 replicas and
    # ~4300 requests. The shipped reference traces are only 300 s long, so a
    # one-hour benchmark trace is built by tiling S2_dev (390 requests at
    # 1.31 rps) 12x with 300 s offsets: 4680 requests over 3600 s at the same
    # arrival rate and prompt mix. This is a TIMING FIXTURE, not a scenario.
    bench = os.path.join(os.path.dirname(os.path.abspath(a.out)) or ".",
                         "bench_1h.jsonl")
    make_bench_trace(os.path.join(a.scenarios_dir, "S2_dev.jsonl"), bench,
                     tiles=12, tile_s=300.0)
    n_bench = sum(1 for _ in open(bench))
    timing = []
    for impl, fn in (("original", run_original), ("kubegym", run_kubegym)):
        cfgx, lmx = ((orig_cfg, lm_o) if impl == "original" else (kg_cfg, lm_k))
        walls = []
        for rep in range(3):
            r = fn(bench, 1, cfgx, lmx, schedule={0: 4}, horizon_s=3600.0, k0=4)
            walls.append(r["wall_s"])
        timing.append({"impl": impl, "trace": "bench_1h (S2_dev tiled 12x)", "reps": 3,
                       "k_replicas": 4, "n_requests": n_bench,
                       "wall_s_min": min(walls), "wall_s_median": sorted(walls)[1],
                       "sim_hours": 1.0,
                       "wall_s_per_sim_hour": sorted(walls)[1],
                       "t_end_s": r["t_end"], "preemptions": r["preemptions"]})
    results["timing"] = timing
    for t in timing:
        print(f"[timing] {t['impl']}: {t['wall_s_median']:.3f} s wall for "
              f"{t['sim_hours']:.2f} simulated h, {t['n_requests']} reqs at k=4 "
              f"-> {t['wall_s_per_sim_hour']:.2f} s/sim-hour")
    results["timing_ratio_kubegym_over_original"] = (
        timing[1]["wall_s_per_sim_hour"] / timing[0]["wall_s_per_sim_hour"])

    n_exact = sum(1 for c in results["cases"] + results["controller_cases"] if c["exact"])
    n_tot = len(results["cases"]) + len(results["controller_cases"])
    results["summary"] = {"n_cases": n_tot, "n_exact": n_exact,
                          "all_exact": n_exact == n_tot}
    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n{n_exact}/{n_tot} cases bit-exact; wrote {a.out}")
    return 0 if n_exact == n_tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
