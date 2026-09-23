# Parity report — `kubegym.models.llm_serving` vs the source `sim_cluster.py`

Harness: `parity_harness.py` (in-suite equivalent: `kubegym/tests/test_parity.py`).
Raw output: `parity_results.json`, `timing_control_steps.json`.

## Result

**Exact parity, 14/14 cases.** Every compared
quantity is bit-identical — not "within tolerance". No tolerance was applied anywhere: the
comparisons below are `==` on floats and integers.

Compared, per case:

* per-request completion time `t_done` (all 225+ requests, matched by `seq`)
* per-request first-token time `t_first_token`
* the drawn output-length realisation (`output_tokens_true`) — checked first, since nothing
  else is comparable if the two implementations draw different work
* total generated tokens
* per-request and total preemption counts
* billed replica-seconds
* the scale-event log (`{t, from, to}` per actuation)
* simulated end time after the post-horizon drain

## Protocol

* Traces: `S1_dev`, `S3_dev`, `S5_dev` (300 s reference scenarios, 217–225 requests), seeds 1 and 2.
* Horizon 900 s at a 15 s control interval, then `drain_all(20000 s)`.
* Two open-loop scale schedules, so **no controller code is in the loop** for the primary
  comparison and any divergence is attributable to the physics port alone:
  * `thrash` = `{'0': 1, '4': 3, '12': 4, '24': 1, '30': 2, '40': 4, '50': 1}` (control-step index → target k): cold start,
    multi-step scale-up, drain, revival of a draining replica, scale-down under load.
  * `starve` = `{'0': 1}`: one replica throughout, so KV fills and the
    recompute-preemption path runs. **This case was added specifically because the first
    parity run recorded 0 preemptions on every case** — the `thrash` schedule provisions
    enough capacity that eviction never happens, so it does not test the preemption code at
    all. Under `starve` the same runs produce 77–306 preemptions, all matching exactly.
* One controller-driven case as a cross-check that the *state surface* is equivalent too:
  the source project's own `controllers.make("hpa", cfg, metric="kv", target=0.6)` driving
  both backends. This also confirms `ProvenancedConfig` is drop-in for the source
  `ClusterConfig` (the controller reads it via `cfg.f(...)` unmodified).
* `kubegym` is configured with `replica_bringup="parallel"`, which reproduces the source
  simulator's bring-up exactly. The shipped default is `"sequential"` — see "Documented
  divergence" below.

## Per-case results

| driver | trace | seed | reqs | done | max abs Δt_done (s) | max abs Δttft (s) | preemptions orig / kubegym | replica-s orig / kubegym | verdict |
|---|---|---|---|---|---|---|---|---|---|
| open_loop:thrash | S1_dev | 1 | 225 | 225 | 0.0 | 0.0 | 0 / 0 | 2735.38 / 2735.38 | exact |
| open_loop:thrash | S1_dev | 2 | 225 | 225 | 0.0 | 0.0 | 0 / 0 | 2983.11 / 2983.11 | exact |
| open_loop:thrash | S3_dev | 1 | 221 | 221 | 0.0 | 0.0 | 0 / 0 | 2811.00 / 2811.00 | exact |
| open_loop:thrash | S3_dev | 2 | 221 | 221 | 0.0 | 0.0 | 0 / 0 | 2753.84 / 2753.84 | exact |
| open_loop:thrash | S5_dev | 1 | 217 | 217 | 0.0 | 0.0 | 0 / 0 | 2820.03 / 2820.03 | exact |
| open_loop:thrash | S5_dev | 2 | 217 | 217 | 0.0 | 0.0 | 0 / 0 | 2635.08 / 2635.08 | exact |
| open_loop:starve | S1_dev | 1 | 225 | 225 | 0.0 | 0.0 | 271 / 271 | 900.00 / 900.00 | exact |
| open_loop:starve | S1_dev | 2 | 225 | 225 | 0.0 | 0.0 | 255 / 255 | 900.00 / 900.00 | exact |
| open_loop:starve | S3_dev | 1 | 221 | 221 | 0.0 | 0.0 | 247 / 247 | 900.00 / 900.00 | exact |
| open_loop:starve | S3_dev | 2 | 221 | 221 | 0.0 | 0.0 | 306 / 306 | 900.00 / 900.00 | exact |
| open_loop:starve | S5_dev | 1 | 217 | 217 | 0.0 | 0.0 | 77 / 77 | 900.00 / 900.00 | exact |
| open_loop:starve | S5_dev | 2 | 217 | 217 | 0.0 | 0.0 | 131 / 131 | 901.03 / 901.03 | exact |
| controllers.HPA(kv,0.6) | S3_dev | 1 | 221 | 221 | 0.0 | 0.0 | 0 / 0 | 2190.02 / 2190.02 | exact |
| controllers.HPA(kv,0.6) | S3_dev | 2 | 221 | 221 | 0.0 | 0.0 | 4 / 4 | 2505.02 / 2505.02 | exact |

Every `Δ` column is exactly `0.0`; `demand_identical` is `true` and
`n_completion_none_mismatch` is `0` in all 14 cases; the scale-event
logs compare equal.

## Documented divergence (a deliberate default change, not a parity failure)

The core's default `replica_bringup` is `"sequential"`, the source simulator's behaviour is
`"parallel"`. Concurrent vLLM engine init fails KV sizing on the testbed, so one-at-a-time
is the verified real-cluster path; parallel bring-up makes multi-replica scale-up look
faster than it is, i.e. **optimistic**. The two policies are provably identical whenever a
scale-up step adds one replica at a time (`test_replica_pool.py::
test_single_replica_scale_up_is_identical_under_both_policies`), which is the common case
for every controller in the suite. Parity above is measured with `"parallel"` so it isolates
the physics port; a run with the shipped default is a *different, more faithful* simulation
and is not expected to match the source simulator when k jumps by more than one.

Two smaller behavioural notes, both preserved rather than fixed:

* Teardown time (`replica_teardown_s = 0.54 s`, weak evidence n=2) is loaded but **not
  billed**, exactly as in the source. Direction: optimistic, magnitude < 1 % of a
  scale-down cycle. `bill_teardown=True` switches it on and then parity would break by
  0.54 s per release.
* The source simulator mixes `<= t` and `<= t + 1e-9` when deciding whether a replica is
  ready. The port uses the `+1e-9` form consistently. A divergence would require a replica's
  ready time to fall within 1 ns of a control tick; it did not occur in any case above, and
  no case is constructed to probe it. Flagged as unverified rather than claimed impossible.

## Timing

Budget: ~5.5 s wall per simulated hour at 4 replicas / ~4300 requests; 13–25 ms per 15 s
control step.

The shipped reference traces are only 300 s long, so the timing fixture is `S2_dev`
(390 requests at 1.31 rps) tiled 12× with 300 s offsets → **4680 requests over 3600 s** at
the same arrival rate and prompt mix, run at k=4. This is a timing fixture, not a scenario.
That it reproduces the original's stated ~5.5 s/simulated-hour (5.47 s
measured here) is the check that the fixture is comparable to whatever produced the budget.

| implementation | wall s / simulated hour (median of 3) | vs budget |
|---|---|---|
| source `sim_cluster.py` | 5.47 | at budget |
| `kubegym` | **3.95** | **28 % under** |

Ratio kubegym / original: **0.72×** — the port is
28 % faster than the code it replaced.

Per-control-step latency, same fixture (240 steps of 15 s at k=4,
4680 requests, 1317 preemptions):

| mean | median | p95 | max | post-horizon drain |
|---|---|---|---|---|
| 16.22 ms | 16.39 ms | 20.34 ms | 23.65 ms | 149 ms |

Inside the 13–25 ms reference band, so a control step remains cheap relative to an RL policy
forward pass and training feasibility is preserved.

### How the slowdown was removed

The first working port ran at 8.81 s/simulated hour — **1.62× slower** than the original,
which would have broken the performance contract even though it passed parity. Three causes,
all in code the event loop touches once per simulated event:

1. `Request.prompt_tokens` / `.generated` were `int()`-casting properties, called O(batch)
   times inside `kv_used()`, itself called on every admission and preemption check. Replaced
   with direct float arithmetic on `size`/`progress` — exact for integral values.
2. `Slot` was a wrapper class, so `len(slot)` and `slot.peek()` were Python-level calls in
   the per-replica admission check. `Slot` is now a `deque` subclass: `if slot:` and
   `slot[0]` are C-level.
3. The per-token bookkeeping (`_mark_output`) was a method call per token per running
   sequence — the single hottest path — and two counters were incremented per token where
   one increment per step suffices. Inlined and hoisted.

Parity was re-verified bit-exact after each change; the numbers above are post-optimisation.

## What this parity does and does not establish

It establishes that the port did not change the physics: the token/KV/preemption/prefill/
cold-start/drain-billing model in `kubegym.models.llm_serving` is the same model, to the
last floating-point bit, as the one in `sim_cluster.py` under both a provisioning-rich and
a KV-starved regime.

It does **not** establish that the physics are right. The source simulator is itself
`calibrated: false`: the per-class output-length distribution is a labelled placeholder,
prefill throughput was never measured, and decode step time above batch 8 is an unmeasured
affine extrapolation flagged optimistic. Parity with an uncalibrated model is fidelity of a
port, not validity of a simulator. Every result row remains stamped
`calibrated=false`.
