# Unverified items and placeholder constants — KubeGym core (Phase 0)

Machine-readable companion: `core_provenance.json`. Nothing in this file is a result.

**Bottom line: both shipped configs are `calibrated: false`.** No number produced by this
core with either config may be reported as an empirical finding. The claims gate enforces
this mechanically (`ProvenancedConfig.calibrated`, stamped onto every row).

---

## 1. Placeholders that block reporting (`required_for_claims: true`)

### `llm_serving_l40s.json` — 4 fields

| field | value | why it blocks | direction of the bias |
|---|---|---|---|
| `output_length_distribution` | delegated to `length_model.py` | The per-class output-length distribution has never been measured on the served model. It is a labelled placeholder lognormal (`short\|no_think` median 100 tok, `long\|think` median 2000 tok, σ 0.5–1.0). **This is the dominant uncalibrated input**: it sets the offered token load, and therefore how hard capacity planning is at all. | sets the difficulty of the problem; sign undefined |
| `prefill_tokens_per_s` | 6000.0 | Never measured on the testbed. Governs TTFT and the decode stall an admission causes, so **every TTFT number this core produces is placeholder-derived**. | unknown |
| `decode_batch_extrapolation_above_8` | `"affine"` | Throughput was measured only at batch 1, 4, 8. Above 8 the affine step-time model predicts near-linear aggregate throughput (872 tok/s at b=16, 1591 at b=32). Real engines saturate. | **optimistic** — biases toward making under-provisioning look cheap. Carries `flagged_optimistic: true` |
| `slo` | ttft p95 2.0 s, tbt p95 0.05 s, ttft hard 5.0 s | Design choice pending calibration, not a measurement. Defines the objective every controller is tuned against. | n/a |

### `request_service_default.json` — every constant, 4 of them blocking

`replica_cold_start_s` (2.0 s), `rs_max_concurrency` (4), `rs_service_time_distribution`
(lognormal μ=0, σ=1 ⇒ median 1.0 s, CV ≈ 1.31), `slo`. Also placeholders but not blocking:
`n_replicas_max` (20), `replica_teardown_s`, `rs_contention_slowdown` (0.0, flagged
optimistic), `rs_queue_discipline`, `router_policy`, `metric_scrape_interval_s`,
`replica_bringup`, `cost_per_replica_hour_usd`.

**This model is a modelling scaffold, not a calibrated artifact.** It exists so a
serverless/microservice corpus has physics to land on and so the `ServiceModel` seam is
exercised by something structurally unlike LLM decoding. The three constants that need real
measurement (or a citation to a measured public dataset) before any claim: cold start,
per-replica concurrency, service-time distribution.

---

## 2. Non-blocking placeholders carried over from the source config

`max_num_seqs` (256, vLLM's documented default, not read off the live server — KV pressure
binds first at these sequence lengths); `chunked_prefill` (`false`, **not modelled**:
overstates the inter-token spike on admission, understates long-prompt TTFT under load);
`preemption_policy` (`recompute` — vLLM's exact victim-selection order was **not** read from
source; this model evicts the most-recently-admitted sequence); `router_policy`
(`least_running`, a placeholder for the deployed router's real policy);
`class_predictor_accuracy` (1.0, a sensitivity knob, not a measurement);
`gpu_cost_per_replica_hour_usd` (1.0, a normalised unit, not a price quote);
`metric_scrape_interval_s` (15.0, staleness currently modelled as **zero** — optimistic
about the controller's information).

## 3. Measured constants carried over — and what "measured" covers

Verbatim from the source config, values and `source` strings unchanged (verified by checksum
comparison in `core_provenance.json`):

* `n_replicas_max = 4` — 4 co-resident vLLM replicas on one L40S 46 GB at
  `--gpu-memory-utilization 0.21`, `--max-model-len 8192`. A **physical ceiling**; a target
  above it requires `allow_hypothetical_replicas` and stamps `hypothetical: true`.
* `kv_tokens_per_replica = 55200` — measured KV budget at that memory fraction.
* `replica_cold_start_s = 31.9` — median of n=9, IQR [31.82, 32.40].
* `decode_step_time_s` — least-squares fit `a0 = 0.01657539`, `a1 = 0.00011063` to measured
  60.3 / 232.6 / 460.2 tok/s at batch 1 / 4 / 8; max relative error 1.05 % on the fitted
  points. **Measured range is batch 1–8 only.**
* `replica_teardown_s = 0.54` — **weak evidence, n=2** (0.54 s for 1 replica, 1.31 s for 4
  concurrent); flagged `weak_evidence`, treated as an upper bound, and currently **not
  billed**.
* `class_signal = thinking_mode` — the router genuinely observes
  `chat_template_kwargs.enable_thinking`, so conditioning on it uses no privileged
  information. **Caveat:** in the shipped traces `thinking_flip_prob = 0.0`, so it is a
  **noiseless** proxy for the length class.
* `control_interval_s = 15.0` — matches the Kubernetes HPA default sync period.
* `n_replicas_min = 1` — a design choice, marked calibrated in the source config because it
  is definitional rather than estimated.

I did not re-derive or re-measure any of these; they are carried over on the authority of
the source testbed's own provenance strings, which I have not independently verified against
raw measurement logs (those logs were not provided to this phase).

## 4. Things I could not verify

1. **`replica_bringup = "sequential"` is marked `calibrated: true`, and that flag is a
   judgement call.** The underlying fact — concurrent vLLM engine init fails KV sizing, so
   replicas were brought up one at a time — was given to me as established. No timing was
   fitted, and I have not seen the failure logs. I recorded it as calibrated to mean "this is
   the path that was actually exercised", and said so in the field's `source`. A reviewer may
   reasonably want it demoted to a design choice; nothing depends on the flag because the
   field is `required_for_claims: false`.
2. **The 1 ns readiness-boundary case.** The source simulator mixes `<= t` and `<= t + 1e-9`
   when testing whether a replica is ready; the port uses `+1e-9` consistently. A divergence
   needs a replica's ready time within 1 ns of a control tick. It did not occur in any parity
   case, and I did not construct a case to force it. Unverified, not proven impossible.
3. **Whether the timing fixture matches the machine that produced the 5.5 s budget.** The
   budget (~5.5 s per simulated hour at 4 replicas / ~4300 requests) came from the source
   project. My fixture (4680 requests, k=4, one hour, tiled from `S2_dev`) measures the
   *original* code at 5.47 s/simulated hour on this machine, which is consistent — but the
   agreement could be coincidental across different hardware. Treat the ratio (0.72×,
   measured back-to-back on one machine) as the reliable number, not either absolute.
4. **`length_model.py` was vendored verbatim, not audited.** Byte-identical to the source
   (sha256 recorded). I read its interface and its placeholder parameters; I did not review
   the bootstrap path that activates when `measured_lengths.json` exists, because no such
   file exists to test against.
5. **The deployed router's real dispatch policy.** `least_outstanding_kv_rr` was written
   from reading `Router.pick()` in the source project's live router. That router is described
   in the source project's own report as *written but never executed* against a live server,
   so "matches the deployment" means "matches the deployment's code", not its observed
   behaviour.
6. **No claim is made that these two service models span cloud resource management.** Two
   models exercise the seam; they do not demonstrate generality.

## 5. A discrepancy found in the source config

`cluster_config.json`'s literal top-level `uncalibrated_required_fields` listed **three**
fields and omitted `slo`, which is itself `required_for_claims: true` and
`calibrated: false`. The source *code* (`ClusterConfig.uncalibrated_required`) computed four
and was correct — the stale literal list in the JSON was the error, and it under-reported the
number of blocking placeholders.

The port therefore **recomputes** the list instead of copying it, and
`llm_serving_l40s.json` lists all four. Any prose in a paper that quotes "three uncalibrated
required fields" from the source project is wrong; it is four. No stamped result changes,
because the gate was always computed from the fields, never from the literal list.

## 6. Out of scope for this phase (not "missing")

Gymnasium wrapper, workload corpus (Azure Functions 2019 replay + parametrized generator),
RL and heuristic baselines, reference results, SLO/reward definitions, `kubegym verify`. The
extension points for each are specified in `INTERFACE.md`.
