# KubeGym core — interface contract (Phase 0)

`kubegym` 0.1.0. Scope of this phase: the package skeleton, the provenance-tracked config
system, the service-model-agnostic discrete-event core, and two service models. **Not** in
this phase: the Gymnasium wrapper, the workload corpus, the RL baselines. This document is
the contract those layers build against.

Dependencies: standard library + numpy. `gymnasium`, `torch` and `stable-baselines3` are
deliberately absent.

**Calibration status of everything below: `calibrated = false`.** Both shipped configs
have at least one `required_for_claims` field that is a placeholder, so every result row
this core produces is stamped non-reportable. See `core_provenance.json` and
`unverified_core.md`.

---

## 1. Layer map

```
                 ┌───────────────────────────────────────────────┐
   later phase   │  Gym wrapper  ·  RL baselines  ·  controllers │
                 └───────┬──────────────────────┬───────────────┘
                         │ ClusterState          │ set_target_replicas(k)
                         │ ObservationSpec        │ Simulator.reset/step_to
                 ┌───────┴──────────────────────┴───────────────┐
   THIS PHASE    │  Simulator (engine.py)                        │
                 │  ├─ ReplicaPool (replica.py)  billing, boots  │
                 │  ├─ DispatchPolicy (router.py)                │
                 │  ├─ ClusterState (state.py)   metric guards   │
                 │  ├─ Slot (service.py)         queue storage   │
                 │  └─ ProvenancedConfig (provenance.py)         │
                 └───────┬──────────────────────┬───────────────┘
                         │ ServiceModel          │ WorkloadSource
                 ┌───────┴──────────┐   ┌────────┴──────────────┐
   THIS PHASE    │ llm_serving      │   │ LLMTraceSource        │
                 │ request_service  │   │ RequestServiceSource  │
                 └──────────────────┘   │ (corpus: later phase) │
                                        └───────────────────────┘
```

The division of labour is the whole design:

* **The engine owns time.** Event loop, control-interval stepping, episode lifecycle,
  replica lifecycle and billing, queue *storage*, dispatch, telemetry assembly.
* **The service model owns physics.** What work a request represents, when it may be
  admitted, what admission costs, when the next service event fires, what it advances,
  whether there is preemption, and which metrics exist.
* **The workload source owns the offered load**, and constructs fresh requests per episode.
* **Nobody in this phase owns reward, SLO or cost objectives.** Those are experiment
  choices and belong in the Gym/eval layer. The core reports conservation quantities only
  (`Simulator.episode_summary`).

---

## 2. `ProvenancedConfig` — the config system and the claims gate

`kubegym/provenance.py`. A config is a set of **fields**, each carrying its own provenance:

| key | meaning |
|---|---|
| `value` | the constant (scalar, string, or nested object) |
| `calibrated` | `true` **only** if the value came from a measurement |
| `source` | free text: how it was obtained — n, hardware, fit quality. **Required**; a field without one is refused at load time |
| `required_for_claims` | if `true`, a placeholder here makes the whole run non-reportable |
| `weak_evidence` | optional: measured, but from too few observations |
| `flagged_optimistic` | optional: known to bias results favourably; `source` must say which direction |

### The claims gate

```python
cfg = ProvenancedConfig.load(config_path("llm_serving_l40s.json"))
cfg.calibrated              # False: AND over every required_for_claims field
cfg.uncalibrated_required   # ['decode_batch_extrapolation_above_8',
                            #  'output_length_distribution', 'prefill_tokens_per_s', 'slo']
cfg.require_calibrated("Table 3")   # raises RuntimeError while any of those is a placeholder
row = cfg.stamp(row)        # adds calibrated / uncalibrated_required_fields / config /
                            # flagged_optimistic_fields / config_overridden
```

**Every downstream result row must pass through `stamp()`.** `Simulator.episode_summary()`
already does. A Gym wrapper reporting episode returns, a baseline sweep writing CSV, and a
figure script writing a caption must all carry the flag; a `calibrated=False` number is not
an empirical result.

### No silent upgrades

`override(name, value, source=..., attest=None)` returns a **copy**.

* Without `attest`, the field becomes `calibrated: false` — even if it was measured before,
  because the old measurement no longer describes the value in use.
* With `attest="<description of the new measurement>"`, the field is `calibrated: true` and
  the attestation is appended to `source`.
* Every override is recorded in `provenance()["overrides"]` and sets `config_overridden` on
  stamped rows.
* `overridden(**values)` is the sweep helper: always uncalibrated.

There is no other code path from placeholder to measured. Do not add one.

### Other surface

`f(name)` / `cfg[name]` / `get(name, default)`, `is_calibrated`, `source`, `field(name)`
(returns the `Field`, including `extra` keys such as `n`, `iqr_s`, `fit_points`),
`names()`, `subset(names)` (per-component provenance; raises on missing fields),
`merge(other)` (duplicate names are an error), `to_dict()`/`save()`, `report()` (human
table), `provenance()` (the block every output file embeds).

`f()` is name-compatible with the source testbed's `ClusterConfig.f`, so the existing
controller suite (`controllers.make(name, cfg, **params)`) accepts a `ProvenancedConfig`
unmodified. This is load-bearing for the baseline phase and is exercised by the parity
harness.

---

## 3. `ServiceModel` — adding a physics model

`kubegym/core/service.py`. Implement two objects.

### 3.1 The model (metadata + factory)

```python
class MyModel:
    name: str                                     # appears in every result row

    def config_fields(self) -> Sequence[str]:     # names consumed, for subsetting
    def metric_names(self) -> Sequence[str]:      # what ClusterState exposes
    def retired_metric_names(self) -> Dict[str, str]:   # name -> reason; get() raises
    def make_replica(self, rid: int, slot: Slot) -> ReplicaEngine
    def reset(self, seed: int) -> None            # re-seed model-owned randomness
    def provenance(self) -> Dict[str, Any]        # measured vs placeholder, infidelities
```

`provenance()` must include a `known_infidelities` list, each entry naming the
**direction** of the bias (optimistic / pessimistic) — both shipped models do.

Metric names should be prefixed with the deployed system's namespace (`vllm:`, `svc:`) and
must not collide with another model's names; a mismatch then fails loudly as a `KeyError`
listing what does exist, instead of silently reading zeros.

### 3.2 The per-replica engine

`BaseReplicaEngine` handles the running list, the service clock, the accepting flag, the
time-between-output accounting and `inflight()`. Subclass it and implement:

```python
def service_step(self, t) -> List[Request]   # advance service; return requests finished at t;
                                             # set t_done; update self.next_step_t
def on_tick(self, t) -> None                 # admit from self.slot, charge blocking work,
                                             # rebalance/preempt, set self.next_step_t
def metrics(self) -> Dict[str, float]        # one value per metric_names() entry
```

Contract details that matter:

* **`next_step_t` is the authoritative service clock**, an attribute (the event loop reads
  it directly — it is the hot path). `next_service_time()` is its accessor. It must be
  `inf` when there is no timed work, and must never be left in the past while
  `running` is non-empty, or the loop stalls.
* `self.running` is a list the core reads directly in the loop.
* `self.accepting` is set by the pool: `False` while draining. Honour it in admission.
* The engine may mutate only runtime fields of a `Request` (`t_admit`, `t_first_token`,
  `t_done`, `progress`, `n_preemptions`, the TBT accumulators). Descriptor fields
  (`demand`, `size`, `t_arrival`, `cls`, `attrs`) belong to the workload source.
* Call order per event, per serving replica: `service_step` (only if `running` and the
  clock is due) then `on_tick` — always.
* Draining replicas still run both hooks: they finish their work, they just refuse new
  admissions.

### 3.3 `Slot` — the queue

`Slot` is a `collections.deque` subclass owned by the core. Read `slot[0]`, take with
`slot.popleft()` / `slot.take()`, return a preemption victim to the head with
`slot.push_front(r)`. `Slot.pop()` **raises**: `deque.pop()` removes from the right, which
would silently serve the queue LIFO.

Recompute-style preemption is expressible precisely because the victim goes back to a
core-owned FIFO while keeping its `progress` — the core never learns what preemption means.

---

## 4. `WorkloadSource` — adding a workload

`kubegym/core/workload.py`. Subclass `WorkloadSource` and implement `records(episode)`
(immutable payloads), optionally `make_request(rec, i, rng)`, `begin_build(episode, seed,
rng)`, `horizon_s(episode)`, `n_episodes()`, `manifest()`.

**Do not override `build()`.** It is the freshness guarantee:

* it constructs new `Request` objects on every call,
* it stamps each one and raises `WorkloadReuseError` if a subclass hands back an object it
  already issued,
* `Simulator.reset()` is the only consumer, and there is no API anywhere that accepts a
  caller-supplied request list.

This exists because the source testbed's `load_trace()` returned mutable requests that the
simulator mutated in place: a second episode over the same list was a **silent no-op** —
wall time collapsed from 5.5 s to 0.02 s, every metric garbage, nothing raised. For an RL
benchmark calling `reset()` thousands of times that is fatal and invisible.
`tests/test_episode.py` asserts both determinism *and* non-triviality, so a regression
fails on the non-triviality half.

Determinism contract: `build(episode, seed)` must be a pure function of `(episode, seed)`
and the immutable records. All sampling from a `random.Random` seeded from those arguments
— never module-level `random`, never an unseeded numpy global.

**Sample per-request work at build time, not in the loop.** That makes controller
comparisons *paired*: for a given `(episode, seed)` every controller faces the identical
realisation, so a controller-vs-controller contrast carries no work-distribution sampling
variance. Both shipped sources do this. A corpus that draws inside the episode breaks
every paired comparison in the benchmark.

Shipped: `RecordListSource`, `JsonlTraceSource`, `LLMTraceSource`, `RequestServiceSource`.
`RequestServiceSource` uses a record's `service_time_s` verbatim when present, which is how
a replay corpus attaches measured durations without subclassing.

---

## 5. `Simulator` — the control surface

```python
sim = Simulator(service_model, workload_source, cfg, k0=1,
                dispatch=None,                    # default: cfg["router_policy"]
                allow_hypothetical_replicas=False,
                replica_bringup=None,             # default: cfg["replica_bringup"]
                bill_teardown=False)

sim.reset(episode=0, seed=0, k0=None) -> ClusterState   # the ONLY way to start an episode
sim.set_target_replicas(k, t=None)                      # the ONLY actuator
sim.step_to(t_end, advance_when_idle=True)
sim.advance_control_interval(dt=None) -> ClusterState    # steps, scrapes, appends to hist
sim.scrape(t=None) -> ClusterState
sim.drain_all(t_cap) -> bool                             # True if truncated by the cap
sim.requests -> List[Request]
sim.total_replica_seconds(t=None) -> float
sim.episode_summary(horizon_s=None) -> dict              # provenance-stamped
sim.provenance() -> dict
sim.hist -> List[ClusterState]                           # scrape history, oldest first
sim.target, sim.t, sim.pool, sim.control_interval_s
```

`run_controller(controller, horizon_s, drain_cap_s=None, episode=0, seed=0, schedule=None)`
runs a whole episode under a `decide(hist, t) -> Optional[int]` controller (the source
testbed's controller interface, unchanged) or under a fixed open-loop `{step: k}` schedule.

### Notes for the Gym wrapper

* One `env.step()` = one `advance_control_interval()`; the action maps to
  `set_target_replicas`. `sim.hist` is the scrape history an observation stacks over.
* `reset(episode, seed)` gives episode-level seeding: `episode` selects the trace,
  `seed` the work realisation. Both feed `WorkloadSource.build`.
* Reward is **not** the core's business. Build it from `ClusterState` and
  `episode_summary()`; the SLO targets live in the config (`slo`, a placeholder) so that
  every controller in a comparison is tuned against identical targets.
* `episode_summary()` is deliberately thin: `n_requests`, `n_done`, `work_done`,
  `work_demanded`, `preemptions_total`, `replica_seconds`, `mean/p95_latency_s`,
  `truncated_drain`, `wall_s`, `scale_events`, `hypothetical_replica_count_used`, plus the
  provenance stamp. Per-request SLO metrics belong to the eval layer, which has
  `sim.requests`.
* `truncated_drain=True` means the post-horizon drain hit its cap and unfinished requests
  were left; treat the episode as truncated rather than dropping them.

---

## 6. `ClusterState` — what a controller may see

`kubegym/core/state.py`. Per-replica metric lists cover **ready** replicas only, in
replica-id order: a booting replica has no metrics endpoint on a real cluster, so it
contributes no entry here either. What a controller can see beyond the metrics:

| attribute | meaning |
|---|---|
| `n_ready / n_booting / n_draining / target` | its own actuation state |
| `n_replicas_effective` | ready + booting + draining, i.e. everything being billed (what the Kubernetes HPA calls `currentReplicas`) |
| `arrivals_since_last`, `arrival_classes_since_last` | router-side arrival count and the class label the router itself observed |
| `router_inflight` | per-request `{seq, tokens_emitted, t_admit, prompt_tokens, observed_mode}` — legitimately observable because the router proxies the stream, and containing **no** information about a request's true total work |
| `calibrated` | the claims gate of the config in use |

Accessors: `get(name)` (list), `total`, `mean`, `max`, `has`; model-agnostic aggregates
`n_running`, `n_waiting`, `kv_mean`, `kv_max`, `preemptions_total`; `as_row()`.

A controller that reads a request's true `demand` is using privileged information and must
declare it (the source testbed's convention: `uses_privileged_info = True`, excluded from
Pareto fronts by default).

### The metric-name guard is load-bearing

`vllm:gpu_cache_usage_perc` **does not exist** on vLLM 0.25.1 — verified absent from all 66
`vllm:*` families on the live server. The correct KV metric is `vllm:kv_cache_usage_perc`.
A controller coded against the retired name reads KV pressure as a permanent zero: a
silently broken baseline that still produces plausible numbers.

`ClusterState.get()` **raises** on a retired name, and `has()` reports `False`.
`RETIRED_METRIC_NAMES` in `core/state.py` is an always-on floor applied whatever model is
active, so a new service model cannot reintroduce the mistake by forgetting to declare it;
a model may add more entries. **Never remove an entry to make a controller run.** Unknown
names also raise, listing what exists — returning 0.0 for a missing metric is exactly the
failure this class prevents.

Verified present: `num_requests_running`, `num_requests_waiting`, `kv_cache_usage_perc`,
`num_preemptions_total`, `generation_tokens_total`, `request_success_total`.

### `ObservationSpec` — observation assembly

```python
spec = ObservationSpec(features=("n_ready","n_booting","n_running","n_waiting",
                                 "sat_mean","sat_max","arrivals"), history=4)
spec.dim; spec.names(); spec.build(sim.hist)   # -> np.float32 vector, oldest first, zero-padded
```

Features are keys of the `FEATURES` dict; extend it to add one. Every feature is derived
only from a `ClusterState`, so an RL policy can never see more than a hand-written
controller. Normalisation is left to the wrapper, which knows the action space and the
replica ceiling.

---

## 7. Replica lifecycle, dispatch, and cost

`ReplicaPool` (`core/replica.py`): `BOOTING → SERVING → DRAINING → RELEASED`.

* **Booting** replicas accept no traffic, expose no metrics, and **are billed** from the
  moment they are requested.
* **Draining** replicas refuse new work, finish what they hold, and **keep billing**. This
  is what makes thrash expensive.
* Scale-down drains the **newest** replicas (they hold the least work). Scale-up **revives**
  a draining replica before starting a cold one, matching an orchestrator cancelling an
  in-flight scale-down.
* Targets are clamped to `[n_replicas_min, n_replicas_max]`. Above the ceiling requires
  `allow_hypothetical_replicas=True`, which sets `hypothetical=True` on the pool and on
  every result row — `n_replicas_max=4` is a **measured** physical limit of the testbed.
* `replica_seconds += dt * (booting + serving + draining)`.

### Bring-up is sequential by default

Concurrent vLLM engine init fails KV sizing on the source testbed; one replica at a time is
the verified path. So `replica_bringup="sequential"`: the i-th replica requested at `t`
becomes ready at `max(t, last_pending_ready) + cold_start`.

The source simulator booted in **parallel** (everything requested at `t` ready at
`t + cold_start`). That is an infidelity relative to the verified path and it makes
multi-replica scale-up look faster than it is — **optimistic**. `"parallel"` reproduces it
exactly and is what the parity harness uses. The two policies are **identical** when
scale-up adds one replica at a time, which is the common case for every controller in the
suite; `request_service_default.json` sets `"parallel"` because container platforms do start
replicas concurrently (a design choice there, not a measurement).

### Dispatch policies (`core/router.py`)

`least_running` (the source simulator's rule; the shipped default and a labelled
placeholder for a real router's policy), `least_outstanding_kv_rr` (matches the deployed
router of the source project: fewest outstanding, then lowest saturation, then
round-robin), `round_robin`, `random`. Add one with `register_policy(name, factory)`;
implement `reset(rng)` and `pick(candidates, req) -> Replica` over accepting replicas.

The router is a confound, not the object of study, which is why it is named and recorded in
provenance rather than hard-coded.

---

## 8. Direction of every known bias

For a benchmark paper the sign matters more than the magnitude.

| choice | direction | note |
|---|---|---|
| decode step time extrapolated affinely above batch 8 | **optimistic** | real engines saturate; makes under-provisioning look cheap. `flagged_optimistic` |
| `prefill_tokens_per_s = 6000`, never measured | **unknown sign** | every TTFT number is placeholder-derived |
| chunked prefill not modelled | mixed | overstates the inter-token spike on admission; understates long-prompt TTFT under load |
| teardown not billed (0.54 s, weak evidence n=2) | **optimistic** | < 1 % of a scale-down cycle |
| parallel bring-up (source simulator; not the default here) | **optimistic** | scale-up looks faster than the verified sequential path |
| metric staleness modelled as zero | **optimistic** about the controller's information | real scrape interval is 15 s |
| `rs_contention_slowdown = 0.0` | **optimistic** | a loaded replica is as fast as an idle one |
| no load shedding / drops in `request_service` | **pessimistic** on latency, optimistic on success rate | queueing delay is the only symptom of overload |
| output-length distribution is a placeholder lognormal | **sets the difficulty of the whole problem** | dominant uncalibrated input |

Two circularity warnings inherited from the source project's report, which any paper using
this core must repeat: (a) a controller whose belief is over the *same* distribution the
simulator samples from is handed a perfectly specified prior, so a belief-vs-point gap is
close to guaranteed by construction and its magnitude is uninformative; (b) in the shipped
traces `thinking_flip_prob = 0.0`, so the class signal is a **noiseless** proxy for the
length class — set `class_predictor_accuracy < 1.0` to inject predictor error.

---

## 9. Extension points, by downstream consumer

**Adaptive replica scaling (controllers / RL).** Implement `decide(hist, t) -> Optional[int]`
and read only `ClusterState` (metrics, `router_inflight`, arrival counts, own actuation
state). Declare `uses_privileged_info` if you read `Request.demand`.
Actuate with `set_target_replicas`. For RL, wrap `advance_control_interval` and build
observations with `ObservationSpec`. Nothing in the core needs changing.

**A second domain (serverless / microservice / batch).** Implement `ServiceModel` +
`ReplicaEngine` (§3) and a `WorkloadSource` (§4), and ship a config whose every field
carries provenance. `request_service.py` is the worked example, and
`tests/test_service_models.py::test_a_new_service_model_needs_only_the_protocol` is a
30-line third model proving the seam holds.

**A workload corpus.** Subclass `WorkloadSource`; put per-record durations in
`service_time_s` (or output lengths via a length model) and draw at build time. Ship a
manifest with file checksums; `manifest()` is embedded in `Simulator.provenance()`.

**Evaluation / reporting.** Consume `episode_summary()` and `sim.requests`; define SLO and
cost metrics there; pass every row through `cfg.stamp()`.

---

## 10. Deviations from the requested layout

One addition: **`kubegym/models/length_model.py`**, vendored **verbatim** (sha256
`7d35ce5e…60d34`, byte-identical to the source; verified in `core_provenance.json`). The
per-class output-length distribution is a `required_for_claims` placeholder that carries
its own calibration flag and its own labelled placeholder parameters, and it upgrades
itself the moment `measured_lengths.json` exists. Copying it verbatim keeps that provenance
intact; re-implementing it inside `llm_serving.py` would have forked it. No other file,
directory, or protocol name deviates from the requested layout.

Two conventions worth flagging as decisions rather than accidents:

* `Request` is generic (`demand` = total service work, `size` = admission footprint) with
  `generated` / `prompt_tokens` / `output_tokens_true` kept as read-only aliases so ported
  LLM code reads naturally. Aliases are meaningful only under a token-based model.
* For `request_service` there is no token stream, so `t_first_token` is set at admission:
  `Request.ttft` is the queueing delay and the TBT accumulators stay empty.

---

## 11. Performance contract

Measured on this machine (darwin, 12 cores), 4 replicas, 4680 requests over one simulated
hour, `llm_serving`:

| | source `sim_cluster.py` | `kubegym` |
|---|---|---|
| wall s per simulated hour | 5.47 | **3.95** |
| mean ms per 15 s control step | — | **16.2** (median 16.4, p95 20.3, max 23.7) |

Budget was ~5.5 s per simulated hour and 13–25 ms per control step; the port is 28 % faster
than the original and inside the control-step band. RL training over this core is therefore
no more expensive than over the original. Details in `parity_report.md`.

Hot-path rules for anyone touching the core: the event loop reads `engine.running` and
`engine.next_step_t` as attributes; `Slot` is a `deque` so admission checks are C-level;
`llm_serving` uses float arithmetic on `size`/`progress` (exact for integral values) and
inlines the per-token bookkeeping. Reintroducing property accessors or wrapper calls in
those three places is what made an earlier revision 1.6× slower.

---

## 12. Test suite

`pytest kubegym/tests -q` — 54 tests, all passing (8 parity tests skip unless
`KUBEGYM_SOURCE_TESTBED` points at the source testbed directory).

| file | what it pins |
|---|---|
| `test_parity.py` | bit-exact parity vs `sim_cluster.py`, 2 traces × 2 seeds × 2 schedules; step-time model vs the three measured throughput points |
| `test_episode.py` | episode independence and determinism, non-triviality, the reuse guard, `reset` refusing completed requests |
| `test_metric_guard.py` | retired names raise under every model, unknown names raise, booting replicas expose no metrics |
| `test_replica_pool.py` | cold start, sequential vs parallel bring-up, drain and boot billing, clamps, hypothetical flagging, revive-on-scale-up |
| `test_provenance.py` | the claims gate, no silent upgrades, `stamp`, roundtrip, the shipped configs' measured constants |
| `test_service_models.py` | interchangeability under one shared driver, protocol conformance, concurrency limits, a third model in 30 lines |
