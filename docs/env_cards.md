# KubeGym env cards

Generated from the live environments by `python -m kubegym.gym.cards`; every number below was read off the object `gymnasium.make(<id>)` returns.

**Every task here is `calibrated = false`.** Both shipped configs have at least one `required_for_claims` field that is a labelled placeholder, so no return, reward or cost on this page is an empirical result. The flag is on every `info` dict, every `list_tasks()` row, and every stamped episode row.

## Registered tasks

| id | service model | workload | action mode | steps | k range | obs dim | calibrated |
|---|---|---|---|---|---|---|---|
| `KubeGym/LLMServing-Mixed-Abs-v0` | llm_serving | S1_dev+S3_dev | absolute | 20 | [1, 4] | 37 | false |
| `KubeGym/LLMServing-Mixed-Delta-v0` | llm_serving | S1_dev+S3_dev | delta | 20 | [1, 4] | 37 | false |
| `KubeGym/LLMServing-S1-Abs-v0` | llm_serving | S1_dev | absolute | 20 | [1, 4] | 37 | false |
| `KubeGym/LLMServing-S1-Delta-v0` | llm_serving | S1_dev | delta | 20 | [1, 4] | 37 | false |
| `KubeGym/LLMServing-S3-Abs-v0` | llm_serving | S3_dev | absolute | 20 | [1, 4] | 37 | false |
| `KubeGym/LLMServing-S3-Delta-v0` | llm_serving | S3_dev | delta | 20 | [1, 4] | 37 | false |
| `KubeGym/RequestService-Poisson-Abs-v0` | request_service | poisson_4rps_300s | absolute | 20 | [1, 20] | 37 | false |
| `KubeGym/RequestService-Poisson-Delta-v0` | request_service | poisson_4rps_300s | delta | 20 | [1, 20] | 37 | false |

## `KubeGym/LLMServing-Mixed-Abs-v0`

**Calibration status: `calibrated = false`.** Blocked by `decode_batch_extrapolation_above_8`, `output_length_distribution`, `prefill_tokens_per_s`, `slo`. No return, reward or cost from this task is an empirical result.

| field | value |
|---|---|
| service model | `llm_serving` |
| config | `llm_serving_l40s.json` |
| workload | `S1_dev+S3_dev` |
| episode length | 20 control steps x 15.0 s = 300.0 s |
| distinct episodes | 2 |
| initial replicas `k0` | 1 |
| replica range | [1, 4]  (ceiling is a MEASURED physical limit of the testbed) |
| action space | `Discrete(4)`  (absolute mode) |
| observation space | `Box(0.0, 10.0, (37,), float32)` |
| post-horizon drain cap | 3000.0 s |
| drain charged to reward | True |

### Workload

Both reference traces as two episodes: `episode=0` is S1_dev, `episode=1` is S3_dev. The only builtin task where the `episode` index selects a different arrival process, so it is the one that exercises episode selection through the Gym API.

| manifest key | value |
|---|---|
| `class` | `LLMTraceSource` |
| `class_predictor_accuracy` | `1.0` |
| `length_model_calibrated` | `False` |
| `length_model_label` | `PLACEHOLDER (not measured; per-class output-length distribution pending live-model measurement)` |
| `n_episodes` | `2` |
| `name` | `llm_trace_S1_S3` |
| `output_length_drawn_at` | `episode build time (paired across controllers)` |
| `paths` | `['S1_dev.jsonl', 'S3_dev.jsonl']` |

### Action space

`Discrete(4)`, mode `absolute`.

| action | meaning |
|---|---|
| 0 | target = 1 |
| 1 | target = 2 |
| 2 | target = 3 |
| 3 | target = 4 |

Targets are clamped to `[1, 4]` by the core. Exceeding the ceiling requires `allow_hypothetical_replicas=True`, which sets `hypothetical=True` on the pool and taints every result row; no registered task enables it.

### Observation vector

37 elements, `float32`, each scaled then clipped into `[0, 10.0]`. History is stacked oldest-first; `[t-0]` is the most recent scrape. Every element but the last is `kubegym.core.state.FEATURES[...](ClusterState)`, so the policy sees no more than a hand-written controller reading the same scrape.

Normalisation divisors are a-priori constants -- config fields or documented design constants of the task. None is derived from the episode in progress, because a statistic of the episode is information from the future and would break the fairness guarantee silently.

| idx | element | scale kind | divisor | source |
|---|---|---|---|---|
| 0 | `n_ready[t-3]` | replica | 4 | ClusterState |
| 1 | `n_booting[t-3]` | replica | 4 | ClusterState |
| 2 | `n_draining[t-3]` | replica | 4 | ClusterState |
| 3 | `target[t-3]` | replica | 4 | ClusterState |
| 4 | `n_running[t-3]` | concurrency | 32 | ClusterState |
| 5 | `n_waiting[t-3]` | concurrency | 32 | ClusterState |
| 6 | `sat_mean[t-3]` | unit | 1 | ClusterState |
| 7 | `sat_max[t-3]` | unit | 1 | ClusterState |
| 8 | `arrivals[t-3]` | arrivals | 16 | ClusterState |
| 9 | `n_ready[t-2]` | replica | 4 | ClusterState |
| 10 | `n_booting[t-2]` | replica | 4 | ClusterState |
| 11 | `n_draining[t-2]` | replica | 4 | ClusterState |
| 12 | `target[t-2]` | replica | 4 | ClusterState |
| 13 | `n_running[t-2]` | concurrency | 32 | ClusterState |
| 14 | `n_waiting[t-2]` | concurrency | 32 | ClusterState |
| 15 | `sat_mean[t-2]` | unit | 1 | ClusterState |
| 16 | `sat_max[t-2]` | unit | 1 | ClusterState |
| 17 | `arrivals[t-2]` | arrivals | 16 | ClusterState |
| 18 | `n_ready[t-1]` | replica | 4 | ClusterState |
| 19 | `n_booting[t-1]` | replica | 4 | ClusterState |
| 20 | `n_draining[t-1]` | replica | 4 | ClusterState |
| 21 | `target[t-1]` | replica | 4 | ClusterState |
| 22 | `n_running[t-1]` | concurrency | 32 | ClusterState |
| 23 | `n_waiting[t-1]` | concurrency | 32 | ClusterState |
| 24 | `sat_mean[t-1]` | unit | 1 | ClusterState |
| 25 | `sat_max[t-1]` | unit | 1 | ClusterState |
| 26 | `arrivals[t-1]` | arrivals | 16 | ClusterState |
| 27 | `n_ready[t-0]` | replica | 4 | ClusterState |
| 28 | `n_booting[t-0]` | replica | 4 | ClusterState |
| 29 | `n_draining[t-0]` | replica | 4 | ClusterState |
| 30 | `target[t-0]` | replica | 4 | ClusterState |
| 31 | `n_running[t-0]` | concurrency | 32 | ClusterState |
| 32 | `n_waiting[t-0]` | concurrency | 32 | ClusterState |
| 33 | `sat_mean[t-0]` | unit | 1 | ClusterState |
| 34 | `sat_max[t-0]` | unit | 1 | ClusterState |
| 35 | `arrivals[t-0]` | arrivals | 16 | ClusterState |
| 36 | `episode_progress` | unit | 20 | control steps taken / n_control_steps; the core's own controller interface decide(hist, t) is handed t, so this is not privileged |

| scale | value |
|---|---|
| `arrivals_scale` | 16 |
| `clip` | 10 |
| `concurrency_scale` | 32 |
| `per_replica_concurrency` | 8 |
| `preemptions_scale` | 100 |
| `replica_scale` | 4 |

### Reward

`reward = -cost; cost >= 0; higher reward is better`. Cost terms, all non-negative:

| term | unit | canonical weight |
|---|---|---|
| `slo` | reward per violating request | 0.1 |
| `resource` | reward per replica-second | 0.0166667 |
| `churn` | reward per replica added or removed | 0.05 |

Charged once at the final step, in addition: `slo_drain` (violations completed during the post-horizon drain), `resource_drain` (replica-seconds billed during the drain), `slo_unfinished` (requests that never completed). Same weights; `include_drain_in_reward = True`.

**SLO profile `llm`.** LLM token-streaming SLO. A completed request violates if its time-to-first-token exceeded `ttft_p95_s` or its mean time-between-tokens exceeded `tbt_p95_s`. An unfinished request violates if it has produced no first token `ttft_hard_s` after arrival.

| SLO target | value (s) |
|---|---|
| `tbt_p95_s` | 0.05 |
| `ttft_hard_s` | 5 |
| `ttft_p95_s` | 2 |

SLO targets come from the config field `slo`, `calibrated = false`. The **weights are a design choice, not a measurement**, and are not covered by the config's claims gate; they are recorded in `unverified_gym.md`. Every term is computable from router and orchestrator telemetry a real deployment has -- the reward never reads a request's ground-truth work (`reads_ground_truth_demand = False`).

### Seeds and splits

`reset(seed=s)` selects `(episode, work_seed)` deterministically as `pairs[s % len(pairs)]` within the env's split, so a rollout is reproducible from the Gym API alone. Splits are disjoint in work seed.

| split | work seeds | (episode, seed) pairs |
|---|---|---|
| `dev` | 16 seeds: 1000..1015 | 32 |
| `test` | 16 seeds: 2000..2015 | 32 |
| `train` | 64 seeds: 0..63 | 128 |

### Calibration and inherited biases

`llm_serving_l40s.json`: UNCALIBRATED OVERALL. Cold start, KV budget and the decode step-time model on batch<=8 are MEASURED on the GPU testbed (vLLM 0.25.1 / Qwen3-1.7B / single L40S 46GB). The per-class OUTPUT-LENGTH DISTRIBUTION is still a labelled placeholder from length_model.py, prefill throughput was never measured, and decode step time above batch 8 is an unmeasured affine extrapolation. No number produced by this simulator may be reported as an empirical result.

Fields flagged as biasing results favourably: `decode_batch_extrapolation_above_8`.

Inherited from INTERFACE.md section 8 (transcribed, not paraphrased):

| choice | direction | note |
|---|---|---|
| decode step time extrapolated affinely above batch 8 | **optimistic** | real engines saturate; makes under-provisioning look cheap. flagged_optimistic |
| prefill_tokens_per_s = 6000, never measured | **unknown sign** | every TTFT number is placeholder-derived |
| chunked prefill not modelled | **mixed** | overstates the inter-token spike on admission; understates long-prompt TTFT under load |
| teardown not billed (0.54 s, weak evidence n=2) | **optimistic** | < 1 % of a scale-down cycle |
| metric staleness modelled as zero | **optimistic about the controller's information** | real scrape interval is 15 s |
| output-length distribution is a placeholder lognormal | **sets the difficulty of the whole problem** | dominant uncalibrated input |

Known infidelities declared by the service model itself:

- prefill_tokens_per_s is a PLACEHOLDER (never measured); all TTFT numbers are placeholder-derived
- decode step time above batch 8 is an unmeasured affine extrapolation; real engines saturate, so this is OPTIMISTIC about large batches
- chunked prefill not modelled: OVERSTATES the inter-token spike on admission, UNDERSTATES long-prompt TTFT under load
- preemption victim order (most-recently-admitted) was not read from vLLM source

Circularity warnings any paper using this core must repeat:

- A controller whose belief is over the SAME distribution the simulator samples from is handed a perfectly specified prior, so a belief-vs-point gap is close to guaranteed by construction and its magnitude is uninformative.
- In the shipped traces thinking_flip_prob = 0.0, so the class signal is a NOISELESS proxy for the length class; set class_predictor_accuracy < 1.0 to inject predictor error.

### Task caveats

- WORKLOAD IS A TEST FIXTURE, NOT THE BENCHMARK CORPUS. The 300 s reference traces in kubegym/tests/data/ are the core's test fixtures (their README says so explicitly). At a 15 s control interval an episode is 20 control steps, which is short for RL: credit for a 31.9 s cold start spans two steps, so roughly a tenth of the episode is spent paying for any single scale-up. Treat these tasks as conformance and smoke-test targets; the workload corpus track supplies the episodes a learning-curve claim should use.
- CALIBRATION: the config is calibrated=false, so no return, reward or cost from this task is an empirical result. The SLO targets the reward is evaluated against are themselves a labelled placeholder (config field `slo`, required_for_claims=true).
- THE SLO TERM IS NEARLY BINARY ON THIS WORKLOAD. Measured over three work seeds at fixed replica counts: 254 violations in total at k=1, and exactly 0 at k=2, k=3 and k=4. At k=1 the KV budget fills, recompute-preemption fires and admission is delayed past ttft_p95_s=2.0 s; from k=2 up, the placeholder prefill throughput (6000 tok/s) makes TTFT negligible and the fitted decode step time (about 17 ms at these batch sizes) is far below tbt_p95_s=0.05 s. So the SLO term behaves as an indicator on 'did the policy sit at one replica under load' rather than as a graded penalty, and the interesting part of the objective on this task is the cost/churn trade-off between k=2 and k=4. Both the targets and the prefill constant are placeholders, so this cliff is a property of uncalibrated inputs and would move under calibration.
- Two arrival traces is not workload diversity. It exercises episode indexing; it does not make a generalisation claim.

## `KubeGym/LLMServing-Mixed-Delta-v0`

**Calibration status: `calibrated = false`.** Blocked by `decode_batch_extrapolation_above_8`, `output_length_distribution`, `prefill_tokens_per_s`, `slo`. No return, reward or cost from this task is an empirical result.

| field | value |
|---|---|
| service model | `llm_serving` |
| config | `llm_serving_l40s.json` |
| workload | `S1_dev+S3_dev` |
| episode length | 20 control steps x 15.0 s = 300.0 s |
| distinct episodes | 2 |
| initial replicas `k0` | 1 |
| replica range | [1, 4]  (ceiling is a MEASURED physical limit of the testbed) |
| action space | `Discrete(5)`  (delta mode) |
| observation space | `Box(0.0, 10.0, (37,), float32)` |
| post-horizon drain cap | 3000.0 s |
| drain charged to reward | True |

### Workload

Both reference traces as two episodes: `episode=0` is S1_dev, `episode=1` is S3_dev. The only builtin task where the `episode` index selects a different arrival process, so it is the one that exercises episode selection through the Gym API.

| manifest key | value |
|---|---|
| `class` | `LLMTraceSource` |
| `class_predictor_accuracy` | `1.0` |
| `length_model_calibrated` | `False` |
| `length_model_label` | `PLACEHOLDER (not measured; per-class output-length distribution pending live-model measurement)` |
| `n_episodes` | `2` |
| `name` | `llm_trace_S1_S3` |
| `output_length_drawn_at` | `episode build time (paired across controllers)` |
| `paths` | `['S1_dev.jsonl', 'S3_dev.jsonl']` |

### Action space

`Discrete(5)`, mode `delta`.

| action | meaning |
|---|---|
| 0 | target -2 |
| 1 | target -1 |
| 2 | target +0 |
| 3 | target +1 |
| 4 | target +2 |

Targets are clamped to `[1, 4]` by the core. Exceeding the ceiling requires `allow_hypothetical_replicas=True`, which sets `hypothetical=True` on the pool and taints every result row; no registered task enables it.

### Observation vector

37 elements, `float32`, each scaled then clipped into `[0, 10.0]`. History is stacked oldest-first; `[t-0]` is the most recent scrape. Every element but the last is `kubegym.core.state.FEATURES[...](ClusterState)`, so the policy sees no more than a hand-written controller reading the same scrape.

Normalisation divisors are a-priori constants -- config fields or documented design constants of the task. None is derived from the episode in progress, because a statistic of the episode is information from the future and would break the fairness guarantee silently.

| idx | element | scale kind | divisor | source |
|---|---|---|---|---|
| 0 | `n_ready[t-3]` | replica | 4 | ClusterState |
| 1 | `n_booting[t-3]` | replica | 4 | ClusterState |
| 2 | `n_draining[t-3]` | replica | 4 | ClusterState |
| 3 | `target[t-3]` | replica | 4 | ClusterState |
| 4 | `n_running[t-3]` | concurrency | 32 | ClusterState |
| 5 | `n_waiting[t-3]` | concurrency | 32 | ClusterState |
| 6 | `sat_mean[t-3]` | unit | 1 | ClusterState |
| 7 | `sat_max[t-3]` | unit | 1 | ClusterState |
| 8 | `arrivals[t-3]` | arrivals | 16 | ClusterState |
| 9 | `n_ready[t-2]` | replica | 4 | ClusterState |
| 10 | `n_booting[t-2]` | replica | 4 | ClusterState |
| 11 | `n_draining[t-2]` | replica | 4 | ClusterState |
| 12 | `target[t-2]` | replica | 4 | ClusterState |
| 13 | `n_running[t-2]` | concurrency | 32 | ClusterState |
| 14 | `n_waiting[t-2]` | concurrency | 32 | ClusterState |
| 15 | `sat_mean[t-2]` | unit | 1 | ClusterState |
| 16 | `sat_max[t-2]` | unit | 1 | ClusterState |
| 17 | `arrivals[t-2]` | arrivals | 16 | ClusterState |
| 18 | `n_ready[t-1]` | replica | 4 | ClusterState |
| 19 | `n_booting[t-1]` | replica | 4 | ClusterState |
| 20 | `n_draining[t-1]` | replica | 4 | ClusterState |
| 21 | `target[t-1]` | replica | 4 | ClusterState |
| 22 | `n_running[t-1]` | concurrency | 32 | ClusterState |
| 23 | `n_waiting[t-1]` | concurrency | 32 | ClusterState |
| 24 | `sat_mean[t-1]` | unit | 1 | ClusterState |
| 25 | `sat_max[t-1]` | unit | 1 | ClusterState |
| 26 | `arrivals[t-1]` | arrivals | 16 | ClusterState |
| 27 | `n_ready[t-0]` | replica | 4 | ClusterState |
| 28 | `n_booting[t-0]` | replica | 4 | ClusterState |
| 29 | `n_draining[t-0]` | replica | 4 | ClusterState |
| 30 | `target[t-0]` | replica | 4 | ClusterState |
| 31 | `n_running[t-0]` | concurrency | 32 | ClusterState |
| 32 | `n_waiting[t-0]` | concurrency | 32 | ClusterState |
| 33 | `sat_mean[t-0]` | unit | 1 | ClusterState |
| 34 | `sat_max[t-0]` | unit | 1 | ClusterState |
| 35 | `arrivals[t-0]` | arrivals | 16 | ClusterState |
| 36 | `episode_progress` | unit | 20 | control steps taken / n_control_steps; the core's own controller interface decide(hist, t) is handed t, so this is not privileged |

| scale | value |
|---|---|
| `arrivals_scale` | 16 |
| `clip` | 10 |
| `concurrency_scale` | 32 |
| `per_replica_concurrency` | 8 |
| `preemptions_scale` | 100 |
| `replica_scale` | 4 |

### Reward

`reward = -cost; cost >= 0; higher reward is better`. Cost terms, all non-negative:

| term | unit | canonical weight |
|---|---|---|
| `slo` | reward per violating request | 0.1 |
| `resource` | reward per replica-second | 0.0166667 |
| `churn` | reward per replica added or removed | 0.05 |

Charged once at the final step, in addition: `slo_drain` (violations completed during the post-horizon drain), `resource_drain` (replica-seconds billed during the drain), `slo_unfinished` (requests that never completed). Same weights; `include_drain_in_reward = True`.

**SLO profile `llm`.** LLM token-streaming SLO. A completed request violates if its time-to-first-token exceeded `ttft_p95_s` or its mean time-between-tokens exceeded `tbt_p95_s`. An unfinished request violates if it has produced no first token `ttft_hard_s` after arrival.

| SLO target | value (s) |
|---|---|
| `tbt_p95_s` | 0.05 |
| `ttft_hard_s` | 5 |
| `ttft_p95_s` | 2 |

SLO targets come from the config field `slo`, `calibrated = false`. The **weights are a design choice, not a measurement**, and are not covered by the config's claims gate; they are recorded in `unverified_gym.md`. Every term is computable from router and orchestrator telemetry a real deployment has -- the reward never reads a request's ground-truth work (`reads_ground_truth_demand = False`).

### Seeds and splits

`reset(seed=s)` selects `(episode, work_seed)` deterministically as `pairs[s % len(pairs)]` within the env's split, so a rollout is reproducible from the Gym API alone. Splits are disjoint in work seed.

| split | work seeds | (episode, seed) pairs |
|---|---|---|
| `dev` | 16 seeds: 1000..1015 | 32 |
| `test` | 16 seeds: 2000..2015 | 32 |
| `train` | 64 seeds: 0..63 | 128 |

### Calibration and inherited biases

`llm_serving_l40s.json`: UNCALIBRATED OVERALL. Cold start, KV budget and the decode step-time model on batch<=8 are MEASURED on the GPU testbed (vLLM 0.25.1 / Qwen3-1.7B / single L40S 46GB). The per-class OUTPUT-LENGTH DISTRIBUTION is still a labelled placeholder from length_model.py, prefill throughput was never measured, and decode step time above batch 8 is an unmeasured affine extrapolation. No number produced by this simulator may be reported as an empirical result.

Fields flagged as biasing results favourably: `decode_batch_extrapolation_above_8`.

Inherited from INTERFACE.md section 8 (transcribed, not paraphrased):

| choice | direction | note |
|---|---|---|
| decode step time extrapolated affinely above batch 8 | **optimistic** | real engines saturate; makes under-provisioning look cheap. flagged_optimistic |
| prefill_tokens_per_s = 6000, never measured | **unknown sign** | every TTFT number is placeholder-derived |
| chunked prefill not modelled | **mixed** | overstates the inter-token spike on admission; understates long-prompt TTFT under load |
| teardown not billed (0.54 s, weak evidence n=2) | **optimistic** | < 1 % of a scale-down cycle |
| metric staleness modelled as zero | **optimistic about the controller's information** | real scrape interval is 15 s |
| output-length distribution is a placeholder lognormal | **sets the difficulty of the whole problem** | dominant uncalibrated input |

Known infidelities declared by the service model itself:

- prefill_tokens_per_s is a PLACEHOLDER (never measured); all TTFT numbers are placeholder-derived
- decode step time above batch 8 is an unmeasured affine extrapolation; real engines saturate, so this is OPTIMISTIC about large batches
- chunked prefill not modelled: OVERSTATES the inter-token spike on admission, UNDERSTATES long-prompt TTFT under load
- preemption victim order (most-recently-admitted) was not read from vLLM source

Circularity warnings any paper using this core must repeat:

- A controller whose belief is over the SAME distribution the simulator samples from is handed a perfectly specified prior, so a belief-vs-point gap is close to guaranteed by construction and its magnitude is uninformative.
- In the shipped traces thinking_flip_prob = 0.0, so the class signal is a NOISELESS proxy for the length class; set class_predictor_accuracy < 1.0 to inject predictor error.

### Task caveats

- WORKLOAD IS A TEST FIXTURE, NOT THE BENCHMARK CORPUS. The 300 s reference traces in kubegym/tests/data/ are the core's test fixtures (their README says so explicitly). At a 15 s control interval an episode is 20 control steps, which is short for RL: credit for a 31.9 s cold start spans two steps, so roughly a tenth of the episode is spent paying for any single scale-up. Treat these tasks as conformance and smoke-test targets; the workload corpus track supplies the episodes a learning-curve claim should use.
- CALIBRATION: the config is calibrated=false, so no return, reward or cost from this task is an empirical result. The SLO targets the reward is evaluated against are themselves a labelled placeholder (config field `slo`, required_for_claims=true).
- THE SLO TERM IS NEARLY BINARY ON THIS WORKLOAD. Measured over three work seeds at fixed replica counts: 254 violations in total at k=1, and exactly 0 at k=2, k=3 and k=4. At k=1 the KV budget fills, recompute-preemption fires and admission is delayed past ttft_p95_s=2.0 s; from k=2 up, the placeholder prefill throughput (6000 tok/s) makes TTFT negligible and the fitted decode step time (about 17 ms at these batch sizes) is far below tbt_p95_s=0.05 s. So the SLO term behaves as an indicator on 'did the policy sit at one replica under load' rather than as a graded penalty, and the interesting part of the objective on this task is the cost/churn trade-off between k=2 and k=4. Both the targets and the prefill constant are placeholders, so this cliff is a property of uncalibrated inputs and would move under calibration.
- Two arrival traces is not workload diversity. It exercises episode indexing; it does not make a generalisation claim.

## `KubeGym/LLMServing-S1-Abs-v0`

**Calibration status: `calibrated = false`.** Blocked by `decode_batch_extrapolation_above_8`, `output_length_distribution`, `prefill_tokens_per_s`, `slo`. No return, reward or cost from this task is an empirical result.

| field | value |
|---|---|
| service model | `llm_serving` |
| config | `llm_serving_l40s.json` |
| workload | `S1_dev` |
| episode length | 20 control steps x 15.0 s = 300.0 s |
| distinct episodes | 1 |
| initial replicas `k0` | 1 |
| replica range | [1, 4]  (ceiling is a MEASURED physical limit of the testbed) |
| action space | `Discrete(4)`  (absolute mode) |
| observation space | `Box(0.0, 10.0, (37,), float32)` |
| post-horizon drain cap | 3000.0 s |
| drain charged to reward | True |

### Workload

Reference trace S1_dev.jsonl: 225 arrivals over 300 s (0.7 rps nominal, p_long 0.5), sha256 2d8c19b0...c59f7. Output lengths are NOT in the trace; they are drawn per request at episode build time from the PLACEHOLDER length model, which is what the work seed varies.

| manifest key | value |
|---|---|
| `class` | `LLMTraceSource` |
| `class_predictor_accuracy` | `1.0` |
| `length_model_calibrated` | `False` |
| `length_model_label` | `PLACEHOLDER (not measured; per-class output-length distribution pending live-model measurement)` |
| `n_episodes` | `1` |
| `name` | `llm_trace_S1_dev` |
| `output_length_drawn_at` | `episode build time (paired across controllers)` |
| `paths` | `['S1_dev.jsonl']` |

### Action space

`Discrete(4)`, mode `absolute`.

| action | meaning |
|---|---|
| 0 | target = 1 |
| 1 | target = 2 |
| 2 | target = 3 |
| 3 | target = 4 |

Targets are clamped to `[1, 4]` by the core. Exceeding the ceiling requires `allow_hypothetical_replicas=True`, which sets `hypothetical=True` on the pool and taints every result row; no registered task enables it.

### Observation vector

37 elements, `float32`, each scaled then clipped into `[0, 10.0]`. History is stacked oldest-first; `[t-0]` is the most recent scrape. Every element but the last is `kubegym.core.state.FEATURES[...](ClusterState)`, so the policy sees no more than a hand-written controller reading the same scrape.

Normalisation divisors are a-priori constants -- config fields or documented design constants of the task. None is derived from the episode in progress, because a statistic of the episode is information from the future and would break the fairness guarantee silently.

| idx | element | scale kind | divisor | source |
|---|---|---|---|---|
| 0 | `n_ready[t-3]` | replica | 4 | ClusterState |
| 1 | `n_booting[t-3]` | replica | 4 | ClusterState |
| 2 | `n_draining[t-3]` | replica | 4 | ClusterState |
| 3 | `target[t-3]` | replica | 4 | ClusterState |
| 4 | `n_running[t-3]` | concurrency | 32 | ClusterState |
| 5 | `n_waiting[t-3]` | concurrency | 32 | ClusterState |
| 6 | `sat_mean[t-3]` | unit | 1 | ClusterState |
| 7 | `sat_max[t-3]` | unit | 1 | ClusterState |
| 8 | `arrivals[t-3]` | arrivals | 16 | ClusterState |
| 9 | `n_ready[t-2]` | replica | 4 | ClusterState |
| 10 | `n_booting[t-2]` | replica | 4 | ClusterState |
| 11 | `n_draining[t-2]` | replica | 4 | ClusterState |
| 12 | `target[t-2]` | replica | 4 | ClusterState |
| 13 | `n_running[t-2]` | concurrency | 32 | ClusterState |
| 14 | `n_waiting[t-2]` | concurrency | 32 | ClusterState |
| 15 | `sat_mean[t-2]` | unit | 1 | ClusterState |
| 16 | `sat_max[t-2]` | unit | 1 | ClusterState |
| 17 | `arrivals[t-2]` | arrivals | 16 | ClusterState |
| 18 | `n_ready[t-1]` | replica | 4 | ClusterState |
| 19 | `n_booting[t-1]` | replica | 4 | ClusterState |
| 20 | `n_draining[t-1]` | replica | 4 | ClusterState |
| 21 | `target[t-1]` | replica | 4 | ClusterState |
| 22 | `n_running[t-1]` | concurrency | 32 | ClusterState |
| 23 | `n_waiting[t-1]` | concurrency | 32 | ClusterState |
| 24 | `sat_mean[t-1]` | unit | 1 | ClusterState |
| 25 | `sat_max[t-1]` | unit | 1 | ClusterState |
| 26 | `arrivals[t-1]` | arrivals | 16 | ClusterState |
| 27 | `n_ready[t-0]` | replica | 4 | ClusterState |
| 28 | `n_booting[t-0]` | replica | 4 | ClusterState |
| 29 | `n_draining[t-0]` | replica | 4 | ClusterState |
| 30 | `target[t-0]` | replica | 4 | ClusterState |
| 31 | `n_running[t-0]` | concurrency | 32 | ClusterState |
| 32 | `n_waiting[t-0]` | concurrency | 32 | ClusterState |
| 33 | `sat_mean[t-0]` | unit | 1 | ClusterState |
| 34 | `sat_max[t-0]` | unit | 1 | ClusterState |
| 35 | `arrivals[t-0]` | arrivals | 16 | ClusterState |
| 36 | `episode_progress` | unit | 20 | control steps taken / n_control_steps; the core's own controller interface decide(hist, t) is handed t, so this is not privileged |

| scale | value |
|---|---|
| `arrivals_scale` | 16 |
| `clip` | 10 |
| `concurrency_scale` | 32 |
| `per_replica_concurrency` | 8 |
| `preemptions_scale` | 100 |
| `replica_scale` | 4 |

### Reward

`reward = -cost; cost >= 0; higher reward is better`. Cost terms, all non-negative:

| term | unit | canonical weight |
|---|---|---|
| `slo` | reward per violating request | 0.1 |
| `resource` | reward per replica-second | 0.0166667 |
| `churn` | reward per replica added or removed | 0.05 |

Charged once at the final step, in addition: `slo_drain` (violations completed during the post-horizon drain), `resource_drain` (replica-seconds billed during the drain), `slo_unfinished` (requests that never completed). Same weights; `include_drain_in_reward = True`.

**SLO profile `llm`.** LLM token-streaming SLO. A completed request violates if its time-to-first-token exceeded `ttft_p95_s` or its mean time-between-tokens exceeded `tbt_p95_s`. An unfinished request violates if it has produced no first token `ttft_hard_s` after arrival.

| SLO target | value (s) |
|---|---|
| `tbt_p95_s` | 0.05 |
| `ttft_hard_s` | 5 |
| `ttft_p95_s` | 2 |

SLO targets come from the config field `slo`, `calibrated = false`. The **weights are a design choice, not a measurement**, and are not covered by the config's claims gate; they are recorded in `unverified_gym.md`. Every term is computable from router and orchestrator telemetry a real deployment has -- the reward never reads a request's ground-truth work (`reads_ground_truth_demand = False`).

### Seeds and splits

`reset(seed=s)` selects `(episode, work_seed)` deterministically as `pairs[s % len(pairs)]` within the env's split, so a rollout is reproducible from the Gym API alone. Splits are disjoint in work seed.

| split | work seeds | (episode, seed) pairs |
|---|---|---|
| `dev` | 16 seeds: 1000..1015 | 16 |
| `test` | 16 seeds: 2000..2015 | 16 |
| `train` | 64 seeds: 0..63 | 64 |

### Calibration and inherited biases

`llm_serving_l40s.json`: UNCALIBRATED OVERALL. Cold start, KV budget and the decode step-time model on batch<=8 are MEASURED on the GPU testbed (vLLM 0.25.1 / Qwen3-1.7B / single L40S 46GB). The per-class OUTPUT-LENGTH DISTRIBUTION is still a labelled placeholder from length_model.py, prefill throughput was never measured, and decode step time above batch 8 is an unmeasured affine extrapolation. No number produced by this simulator may be reported as an empirical result.

Fields flagged as biasing results favourably: `decode_batch_extrapolation_above_8`.

Inherited from INTERFACE.md section 8 (transcribed, not paraphrased):

| choice | direction | note |
|---|---|---|
| decode step time extrapolated affinely above batch 8 | **optimistic** | real engines saturate; makes under-provisioning look cheap. flagged_optimistic |
| prefill_tokens_per_s = 6000, never measured | **unknown sign** | every TTFT number is placeholder-derived |
| chunked prefill not modelled | **mixed** | overstates the inter-token spike on admission; understates long-prompt TTFT under load |
| teardown not billed (0.54 s, weak evidence n=2) | **optimistic** | < 1 % of a scale-down cycle |
| metric staleness modelled as zero | **optimistic about the controller's information** | real scrape interval is 15 s |
| output-length distribution is a placeholder lognormal | **sets the difficulty of the whole problem** | dominant uncalibrated input |

Known infidelities declared by the service model itself:

- prefill_tokens_per_s is a PLACEHOLDER (never measured); all TTFT numbers are placeholder-derived
- decode step time above batch 8 is an unmeasured affine extrapolation; real engines saturate, so this is OPTIMISTIC about large batches
- chunked prefill not modelled: OVERSTATES the inter-token spike on admission, UNDERSTATES long-prompt TTFT under load
- preemption victim order (most-recently-admitted) was not read from vLLM source

Circularity warnings any paper using this core must repeat:

- A controller whose belief is over the SAME distribution the simulator samples from is handed a perfectly specified prior, so a belief-vs-point gap is close to guaranteed by construction and its magnitude is uninformative.
- In the shipped traces thinking_flip_prob = 0.0, so the class signal is a NOISELESS proxy for the length class; set class_predictor_accuracy < 1.0 to inject predictor error.

### Task caveats

- WORKLOAD IS A TEST FIXTURE, NOT THE BENCHMARK CORPUS. The 300 s reference traces in kubegym/tests/data/ are the core's test fixtures (their README says so explicitly). At a 15 s control interval an episode is 20 control steps, which is short for RL: credit for a 31.9 s cold start spans two steps, so roughly a tenth of the episode is spent paying for any single scale-up. Treat these tasks as conformance and smoke-test targets; the workload corpus track supplies the episodes a learning-curve claim should use.
- CALIBRATION: the config is calibrated=false, so no return, reward or cost from this task is an empirical result. The SLO targets the reward is evaluated against are themselves a labelled placeholder (config field `slo`, required_for_claims=true).
- THE SLO TERM IS NEARLY BINARY ON THIS WORKLOAD. Measured over three work seeds at fixed replica counts: 254 violations in total at k=1, and exactly 0 at k=2, k=3 and k=4. At k=1 the KV budget fills, recompute-preemption fires and admission is delayed past ttft_p95_s=2.0 s; from k=2 up, the placeholder prefill throughput (6000 tok/s) makes TTFT negligible and the fitted decode step time (about 17 ms at these batch sizes) is far below tbt_p95_s=0.05 s. So the SLO term behaves as an indicator on 'did the policy sit at one replica under load' rather than as a graded penalty, and the interesting part of the objective on this task is the cost/churn trade-off between k=2 and k=4. Both the targets and the prefill constant are placeholders, so this cliff is a property of uncalibrated inputs and would move under calibration.
- thinking_flip_prob = 0.0 in the generator that produced this trace, so the observed class signal is a NOISELESS proxy for the length class. Nothing in the observation vector exposes the class signal, so this task does not benefit from it; a task that adds a class feature must repeat this caveat.

## `KubeGym/LLMServing-S1-Delta-v0`

**Calibration status: `calibrated = false`.** Blocked by `decode_batch_extrapolation_above_8`, `output_length_distribution`, `prefill_tokens_per_s`, `slo`. No return, reward or cost from this task is an empirical result.

| field | value |
|---|---|
| service model | `llm_serving` |
| config | `llm_serving_l40s.json` |
| workload | `S1_dev` |
| episode length | 20 control steps x 15.0 s = 300.0 s |
| distinct episodes | 1 |
| initial replicas `k0` | 1 |
| replica range | [1, 4]  (ceiling is a MEASURED physical limit of the testbed) |
| action space | `Discrete(5)`  (delta mode) |
| observation space | `Box(0.0, 10.0, (37,), float32)` |
| post-horizon drain cap | 3000.0 s |
| drain charged to reward | True |

### Workload

Reference trace S1_dev.jsonl: 225 arrivals over 300 s (0.7 rps nominal, p_long 0.5), sha256 2d8c19b0...c59f7. Output lengths are NOT in the trace; they are drawn per request at episode build time from the PLACEHOLDER length model, which is what the work seed varies.

| manifest key | value |
|---|---|
| `class` | `LLMTraceSource` |
| `class_predictor_accuracy` | `1.0` |
| `length_model_calibrated` | `False` |
| `length_model_label` | `PLACEHOLDER (not measured; per-class output-length distribution pending live-model measurement)` |
| `n_episodes` | `1` |
| `name` | `llm_trace_S1_dev` |
| `output_length_drawn_at` | `episode build time (paired across controllers)` |
| `paths` | `['S1_dev.jsonl']` |

### Action space

`Discrete(5)`, mode `delta`.

| action | meaning |
|---|---|
| 0 | target -2 |
| 1 | target -1 |
| 2 | target +0 |
| 3 | target +1 |
| 4 | target +2 |

Targets are clamped to `[1, 4]` by the core. Exceeding the ceiling requires `allow_hypothetical_replicas=True`, which sets `hypothetical=True` on the pool and taints every result row; no registered task enables it.

### Observation vector

37 elements, `float32`, each scaled then clipped into `[0, 10.0]`. History is stacked oldest-first; `[t-0]` is the most recent scrape. Every element but the last is `kubegym.core.state.FEATURES[...](ClusterState)`, so the policy sees no more than a hand-written controller reading the same scrape.

Normalisation divisors are a-priori constants -- config fields or documented design constants of the task. None is derived from the episode in progress, because a statistic of the episode is information from the future and would break the fairness guarantee silently.

| idx | element | scale kind | divisor | source |
|---|---|---|---|---|
| 0 | `n_ready[t-3]` | replica | 4 | ClusterState |
| 1 | `n_booting[t-3]` | replica | 4 | ClusterState |
| 2 | `n_draining[t-3]` | replica | 4 | ClusterState |
| 3 | `target[t-3]` | replica | 4 | ClusterState |
| 4 | `n_running[t-3]` | concurrency | 32 | ClusterState |
| 5 | `n_waiting[t-3]` | concurrency | 32 | ClusterState |
| 6 | `sat_mean[t-3]` | unit | 1 | ClusterState |
| 7 | `sat_max[t-3]` | unit | 1 | ClusterState |
| 8 | `arrivals[t-3]` | arrivals | 16 | ClusterState |
| 9 | `n_ready[t-2]` | replica | 4 | ClusterState |
| 10 | `n_booting[t-2]` | replica | 4 | ClusterState |
| 11 | `n_draining[t-2]` | replica | 4 | ClusterState |
| 12 | `target[t-2]` | replica | 4 | ClusterState |
| 13 | `n_running[t-2]` | concurrency | 32 | ClusterState |
| 14 | `n_waiting[t-2]` | concurrency | 32 | ClusterState |
| 15 | `sat_mean[t-2]` | unit | 1 | ClusterState |
| 16 | `sat_max[t-2]` | unit | 1 | ClusterState |
| 17 | `arrivals[t-2]` | arrivals | 16 | ClusterState |
| 18 | `n_ready[t-1]` | replica | 4 | ClusterState |
| 19 | `n_booting[t-1]` | replica | 4 | ClusterState |
| 20 | `n_draining[t-1]` | replica | 4 | ClusterState |
| 21 | `target[t-1]` | replica | 4 | ClusterState |
| 22 | `n_running[t-1]` | concurrency | 32 | ClusterState |
| 23 | `n_waiting[t-1]` | concurrency | 32 | ClusterState |
| 24 | `sat_mean[t-1]` | unit | 1 | ClusterState |
| 25 | `sat_max[t-1]` | unit | 1 | ClusterState |
| 26 | `arrivals[t-1]` | arrivals | 16 | ClusterState |
| 27 | `n_ready[t-0]` | replica | 4 | ClusterState |
| 28 | `n_booting[t-0]` | replica | 4 | ClusterState |
| 29 | `n_draining[t-0]` | replica | 4 | ClusterState |
| 30 | `target[t-0]` | replica | 4 | ClusterState |
| 31 | `n_running[t-0]` | concurrency | 32 | ClusterState |
| 32 | `n_waiting[t-0]` | concurrency | 32 | ClusterState |
| 33 | `sat_mean[t-0]` | unit | 1 | ClusterState |
| 34 | `sat_max[t-0]` | unit | 1 | ClusterState |
| 35 | `arrivals[t-0]` | arrivals | 16 | ClusterState |
| 36 | `episode_progress` | unit | 20 | control steps taken / n_control_steps; the core's own controller interface decide(hist, t) is handed t, so this is not privileged |

| scale | value |
|---|---|
| `arrivals_scale` | 16 |
| `clip` | 10 |
| `concurrency_scale` | 32 |
| `per_replica_concurrency` | 8 |
| `preemptions_scale` | 100 |
| `replica_scale` | 4 |

### Reward

`reward = -cost; cost >= 0; higher reward is better`. Cost terms, all non-negative:

| term | unit | canonical weight |
|---|---|---|
| `slo` | reward per violating request | 0.1 |
| `resource` | reward per replica-second | 0.0166667 |
| `churn` | reward per replica added or removed | 0.05 |

Charged once at the final step, in addition: `slo_drain` (violations completed during the post-horizon drain), `resource_drain` (replica-seconds billed during the drain), `slo_unfinished` (requests that never completed). Same weights; `include_drain_in_reward = True`.

**SLO profile `llm`.** LLM token-streaming SLO. A completed request violates if its time-to-first-token exceeded `ttft_p95_s` or its mean time-between-tokens exceeded `tbt_p95_s`. An unfinished request violates if it has produced no first token `ttft_hard_s` after arrival.

| SLO target | value (s) |
|---|---|
| `tbt_p95_s` | 0.05 |
| `ttft_hard_s` | 5 |
| `ttft_p95_s` | 2 |

SLO targets come from the config field `slo`, `calibrated = false`. The **weights are a design choice, not a measurement**, and are not covered by the config's claims gate; they are recorded in `unverified_gym.md`. Every term is computable from router and orchestrator telemetry a real deployment has -- the reward never reads a request's ground-truth work (`reads_ground_truth_demand = False`).

### Seeds and splits

`reset(seed=s)` selects `(episode, work_seed)` deterministically as `pairs[s % len(pairs)]` within the env's split, so a rollout is reproducible from the Gym API alone. Splits are disjoint in work seed.

| split | work seeds | (episode, seed) pairs |
|---|---|---|
| `dev` | 16 seeds: 1000..1015 | 16 |
| `test` | 16 seeds: 2000..2015 | 16 |
| `train` | 64 seeds: 0..63 | 64 |

### Calibration and inherited biases

`llm_serving_l40s.json`: UNCALIBRATED OVERALL. Cold start, KV budget and the decode step-time model on batch<=8 are MEASURED on the GPU testbed (vLLM 0.25.1 / Qwen3-1.7B / single L40S 46GB). The per-class OUTPUT-LENGTH DISTRIBUTION is still a labelled placeholder from length_model.py, prefill throughput was never measured, and decode step time above batch 8 is an unmeasured affine extrapolation. No number produced by this simulator may be reported as an empirical result.

Fields flagged as biasing results favourably: `decode_batch_extrapolation_above_8`.

Inherited from INTERFACE.md section 8 (transcribed, not paraphrased):

| choice | direction | note |
|---|---|---|
| decode step time extrapolated affinely above batch 8 | **optimistic** | real engines saturate; makes under-provisioning look cheap. flagged_optimistic |
| prefill_tokens_per_s = 6000, never measured | **unknown sign** | every TTFT number is placeholder-derived |
| chunked prefill not modelled | **mixed** | overstates the inter-token spike on admission; understates long-prompt TTFT under load |
| teardown not billed (0.54 s, weak evidence n=2) | **optimistic** | < 1 % of a scale-down cycle |
| metric staleness modelled as zero | **optimistic about the controller's information** | real scrape interval is 15 s |
| output-length distribution is a placeholder lognormal | **sets the difficulty of the whole problem** | dominant uncalibrated input |

Known infidelities declared by the service model itself:

- prefill_tokens_per_s is a PLACEHOLDER (never measured); all TTFT numbers are placeholder-derived
- decode step time above batch 8 is an unmeasured affine extrapolation; real engines saturate, so this is OPTIMISTIC about large batches
- chunked prefill not modelled: OVERSTATES the inter-token spike on admission, UNDERSTATES long-prompt TTFT under load
- preemption victim order (most-recently-admitted) was not read from vLLM source

Circularity warnings any paper using this core must repeat:

- A controller whose belief is over the SAME distribution the simulator samples from is handed a perfectly specified prior, so a belief-vs-point gap is close to guaranteed by construction and its magnitude is uninformative.
- In the shipped traces thinking_flip_prob = 0.0, so the class signal is a NOISELESS proxy for the length class; set class_predictor_accuracy < 1.0 to inject predictor error.

### Task caveats

- WORKLOAD IS A TEST FIXTURE, NOT THE BENCHMARK CORPUS. The 300 s reference traces in kubegym/tests/data/ are the core's test fixtures (their README says so explicitly). At a 15 s control interval an episode is 20 control steps, which is short for RL: credit for a 31.9 s cold start spans two steps, so roughly a tenth of the episode is spent paying for any single scale-up. Treat these tasks as conformance and smoke-test targets; the workload corpus track supplies the episodes a learning-curve claim should use.
- CALIBRATION: the config is calibrated=false, so no return, reward or cost from this task is an empirical result. The SLO targets the reward is evaluated against are themselves a labelled placeholder (config field `slo`, required_for_claims=true).
- THE SLO TERM IS NEARLY BINARY ON THIS WORKLOAD. Measured over three work seeds at fixed replica counts: 254 violations in total at k=1, and exactly 0 at k=2, k=3 and k=4. At k=1 the KV budget fills, recompute-preemption fires and admission is delayed past ttft_p95_s=2.0 s; from k=2 up, the placeholder prefill throughput (6000 tok/s) makes TTFT negligible and the fitted decode step time (about 17 ms at these batch sizes) is far below tbt_p95_s=0.05 s. So the SLO term behaves as an indicator on 'did the policy sit at one replica under load' rather than as a graded penalty, and the interesting part of the objective on this task is the cost/churn trade-off between k=2 and k=4. Both the targets and the prefill constant are placeholders, so this cliff is a property of uncalibrated inputs and would move under calibration.
- thinking_flip_prob = 0.0 in the generator that produced this trace, so the observed class signal is a NOISELESS proxy for the length class. Nothing in the observation vector exposes the class signal, so this task does not benefit from it; a task that adds a class feature must repeat this caveat.

## `KubeGym/LLMServing-S3-Abs-v0`

**Calibration status: `calibrated = false`.** Blocked by `decode_batch_extrapolation_above_8`, `output_length_distribution`, `prefill_tokens_per_s`, `slo`. No return, reward or cost from this task is an empirical result.

| field | value |
|---|---|
| service model | `llm_serving` |
| config | `llm_serving_l40s.json` |
| workload | `S3_dev` |
| episode length | 20 control steps x 15.0 s = 300.0 s |
| distinct episodes | 1 |
| initial replicas `k0` | 1 |
| replica range | [1, 4]  (ceiling is a MEASURED physical limit of the testbed) |
| action space | `Discrete(4)`  (absolute mode) |
| observation space | `Box(0.0, 10.0, (37,), float32)` |
| post-horizon drain cap | 3000.0 s |
| drain charged to reward | True |

### Workload

Reference trace S3_dev.jsonl: 221 arrivals over 300 s, sha256 4c366a3e...79f007. Output lengths drawn at build time from the PLACEHOLDER length model.

| manifest key | value |
|---|---|
| `class` | `LLMTraceSource` |
| `class_predictor_accuracy` | `1.0` |
| `length_model_calibrated` | `False` |
| `length_model_label` | `PLACEHOLDER (not measured; per-class output-length distribution pending live-model measurement)` |
| `n_episodes` | `1` |
| `name` | `llm_trace_S3_dev` |
| `output_length_drawn_at` | `episode build time (paired across controllers)` |
| `paths` | `['S3_dev.jsonl']` |

### Action space

`Discrete(4)`, mode `absolute`.

| action | meaning |
|---|---|
| 0 | target = 1 |
| 1 | target = 2 |
| 2 | target = 3 |
| 3 | target = 4 |

Targets are clamped to `[1, 4]` by the core. Exceeding the ceiling requires `allow_hypothetical_replicas=True`, which sets `hypothetical=True` on the pool and taints every result row; no registered task enables it.

### Observation vector

37 elements, `float32`, each scaled then clipped into `[0, 10.0]`. History is stacked oldest-first; `[t-0]` is the most recent scrape. Every element but the last is `kubegym.core.state.FEATURES[...](ClusterState)`, so the policy sees no more than a hand-written controller reading the same scrape.

Normalisation divisors are a-priori constants -- config fields or documented design constants of the task. None is derived from the episode in progress, because a statistic of the episode is information from the future and would break the fairness guarantee silently.

| idx | element | scale kind | divisor | source |
|---|---|---|---|---|
| 0 | `n_ready[t-3]` | replica | 4 | ClusterState |
| 1 | `n_booting[t-3]` | replica | 4 | ClusterState |
| 2 | `n_draining[t-3]` | replica | 4 | ClusterState |
| 3 | `target[t-3]` | replica | 4 | ClusterState |
| 4 | `n_running[t-3]` | concurrency | 32 | ClusterState |
| 5 | `n_waiting[t-3]` | concurrency | 32 | ClusterState |
| 6 | `sat_mean[t-3]` | unit | 1 | ClusterState |
| 7 | `sat_max[t-3]` | unit | 1 | ClusterState |
| 8 | `arrivals[t-3]` | arrivals | 16 | ClusterState |
| 9 | `n_ready[t-2]` | replica | 4 | ClusterState |
| 10 | `n_booting[t-2]` | replica | 4 | ClusterState |
| 11 | `n_draining[t-2]` | replica | 4 | ClusterState |
| 12 | `target[t-2]` | replica | 4 | ClusterState |
| 13 | `n_running[t-2]` | concurrency | 32 | ClusterState |
| 14 | `n_waiting[t-2]` | concurrency | 32 | ClusterState |
| 15 | `sat_mean[t-2]` | unit | 1 | ClusterState |
| 16 | `sat_max[t-2]` | unit | 1 | ClusterState |
| 17 | `arrivals[t-2]` | arrivals | 16 | ClusterState |
| 18 | `n_ready[t-1]` | replica | 4 | ClusterState |
| 19 | `n_booting[t-1]` | replica | 4 | ClusterState |
| 20 | `n_draining[t-1]` | replica | 4 | ClusterState |
| 21 | `target[t-1]` | replica | 4 | ClusterState |
| 22 | `n_running[t-1]` | concurrency | 32 | ClusterState |
| 23 | `n_waiting[t-1]` | concurrency | 32 | ClusterState |
| 24 | `sat_mean[t-1]` | unit | 1 | ClusterState |
| 25 | `sat_max[t-1]` | unit | 1 | ClusterState |
| 26 | `arrivals[t-1]` | arrivals | 16 | ClusterState |
| 27 | `n_ready[t-0]` | replica | 4 | ClusterState |
| 28 | `n_booting[t-0]` | replica | 4 | ClusterState |
| 29 | `n_draining[t-0]` | replica | 4 | ClusterState |
| 30 | `target[t-0]` | replica | 4 | ClusterState |
| 31 | `n_running[t-0]` | concurrency | 32 | ClusterState |
| 32 | `n_waiting[t-0]` | concurrency | 32 | ClusterState |
| 33 | `sat_mean[t-0]` | unit | 1 | ClusterState |
| 34 | `sat_max[t-0]` | unit | 1 | ClusterState |
| 35 | `arrivals[t-0]` | arrivals | 16 | ClusterState |
| 36 | `episode_progress` | unit | 20 | control steps taken / n_control_steps; the core's own controller interface decide(hist, t) is handed t, so this is not privileged |

| scale | value |
|---|---|
| `arrivals_scale` | 16 |
| `clip` | 10 |
| `concurrency_scale` | 32 |
| `per_replica_concurrency` | 8 |
| `preemptions_scale` | 100 |
| `replica_scale` | 4 |

### Reward

`reward = -cost; cost >= 0; higher reward is better`. Cost terms, all non-negative:

| term | unit | canonical weight |
|---|---|---|
| `slo` | reward per violating request | 0.1 |
| `resource` | reward per replica-second | 0.0166667 |
| `churn` | reward per replica added or removed | 0.05 |

Charged once at the final step, in addition: `slo_drain` (violations completed during the post-horizon drain), `resource_drain` (replica-seconds billed during the drain), `slo_unfinished` (requests that never completed). Same weights; `include_drain_in_reward = True`.

**SLO profile `llm`.** LLM token-streaming SLO. A completed request violates if its time-to-first-token exceeded `ttft_p95_s` or its mean time-between-tokens exceeded `tbt_p95_s`. An unfinished request violates if it has produced no first token `ttft_hard_s` after arrival.

| SLO target | value (s) |
|---|---|
| `tbt_p95_s` | 0.05 |
| `ttft_hard_s` | 5 |
| `ttft_p95_s` | 2 |

SLO targets come from the config field `slo`, `calibrated = false`. The **weights are a design choice, not a measurement**, and are not covered by the config's claims gate; they are recorded in `unverified_gym.md`. Every term is computable from router and orchestrator telemetry a real deployment has -- the reward never reads a request's ground-truth work (`reads_ground_truth_demand = False`).

### Seeds and splits

`reset(seed=s)` selects `(episode, work_seed)` deterministically as `pairs[s % len(pairs)]` within the env's split, so a rollout is reproducible from the Gym API alone. Splits are disjoint in work seed.

| split | work seeds | (episode, seed) pairs |
|---|---|---|
| `dev` | 16 seeds: 1000..1015 | 16 |
| `test` | 16 seeds: 2000..2015 | 16 |
| `train` | 64 seeds: 0..63 | 64 |

### Calibration and inherited biases

`llm_serving_l40s.json`: UNCALIBRATED OVERALL. Cold start, KV budget and the decode step-time model on batch<=8 are MEASURED on the GPU testbed (vLLM 0.25.1 / Qwen3-1.7B / single L40S 46GB). The per-class OUTPUT-LENGTH DISTRIBUTION is still a labelled placeholder from length_model.py, prefill throughput was never measured, and decode step time above batch 8 is an unmeasured affine extrapolation. No number produced by this simulator may be reported as an empirical result.

Fields flagged as biasing results favourably: `decode_batch_extrapolation_above_8`.

Inherited from INTERFACE.md section 8 (transcribed, not paraphrased):

| choice | direction | note |
|---|---|---|
| decode step time extrapolated affinely above batch 8 | **optimistic** | real engines saturate; makes under-provisioning look cheap. flagged_optimistic |
| prefill_tokens_per_s = 6000, never measured | **unknown sign** | every TTFT number is placeholder-derived |
| chunked prefill not modelled | **mixed** | overstates the inter-token spike on admission; understates long-prompt TTFT under load |
| teardown not billed (0.54 s, weak evidence n=2) | **optimistic** | < 1 % of a scale-down cycle |
| metric staleness modelled as zero | **optimistic about the controller's information** | real scrape interval is 15 s |
| output-length distribution is a placeholder lognormal | **sets the difficulty of the whole problem** | dominant uncalibrated input |

Known infidelities declared by the service model itself:

- prefill_tokens_per_s is a PLACEHOLDER (never measured); all TTFT numbers are placeholder-derived
- decode step time above batch 8 is an unmeasured affine extrapolation; real engines saturate, so this is OPTIMISTIC about large batches
- chunked prefill not modelled: OVERSTATES the inter-token spike on admission, UNDERSTATES long-prompt TTFT under load
- preemption victim order (most-recently-admitted) was not read from vLLM source

Circularity warnings any paper using this core must repeat:

- A controller whose belief is over the SAME distribution the simulator samples from is handed a perfectly specified prior, so a belief-vs-point gap is close to guaranteed by construction and its magnitude is uninformative.
- In the shipped traces thinking_flip_prob = 0.0, so the class signal is a NOISELESS proxy for the length class; set class_predictor_accuracy < 1.0 to inject predictor error.

### Task caveats

- WORKLOAD IS A TEST FIXTURE, NOT THE BENCHMARK CORPUS. The 300 s reference traces in kubegym/tests/data/ are the core's test fixtures (their README says so explicitly). At a 15 s control interval an episode is 20 control steps, which is short for RL: credit for a 31.9 s cold start spans two steps, so roughly a tenth of the episode is spent paying for any single scale-up. Treat these tasks as conformance and smoke-test targets; the workload corpus track supplies the episodes a learning-curve claim should use.
- CALIBRATION: the config is calibrated=false, so no return, reward or cost from this task is an empirical result. The SLO targets the reward is evaluated against are themselves a labelled placeholder (config field `slo`, required_for_claims=true).
- THE SLO TERM IS NEARLY BINARY ON THIS WORKLOAD. Measured over three work seeds at fixed replica counts: 254 violations in total at k=1, and exactly 0 at k=2, k=3 and k=4. At k=1 the KV budget fills, recompute-preemption fires and admission is delayed past ttft_p95_s=2.0 s; from k=2 up, the placeholder prefill throughput (6000 tok/s) makes TTFT negligible and the fitted decode step time (about 17 ms at these batch sizes) is far below tbt_p95_s=0.05 s. So the SLO term behaves as an indicator on 'did the policy sit at one replica under load' rather than as a graded penalty, and the interesting part of the objective on this task is the cost/churn trade-off between k=2 and k=4. Both the targets and the prefill constant are placeholders, so this cliff is a property of uncalibrated inputs and would move under calibration.

## `KubeGym/LLMServing-S3-Delta-v0`

**Calibration status: `calibrated = false`.** Blocked by `decode_batch_extrapolation_above_8`, `output_length_distribution`, `prefill_tokens_per_s`, `slo`. No return, reward or cost from this task is an empirical result.

| field | value |
|---|---|
| service model | `llm_serving` |
| config | `llm_serving_l40s.json` |
| workload | `S3_dev` |
| episode length | 20 control steps x 15.0 s = 300.0 s |
| distinct episodes | 1 |
| initial replicas `k0` | 1 |
| replica range | [1, 4]  (ceiling is a MEASURED physical limit of the testbed) |
| action space | `Discrete(5)`  (delta mode) |
| observation space | `Box(0.0, 10.0, (37,), float32)` |
| post-horizon drain cap | 3000.0 s |
| drain charged to reward | True |

### Workload

Reference trace S3_dev.jsonl: 221 arrivals over 300 s, sha256 4c366a3e...79f007. Output lengths drawn at build time from the PLACEHOLDER length model.

| manifest key | value |
|---|---|
| `class` | `LLMTraceSource` |
| `class_predictor_accuracy` | `1.0` |
| `length_model_calibrated` | `False` |
| `length_model_label` | `PLACEHOLDER (not measured; per-class output-length distribution pending live-model measurement)` |
| `n_episodes` | `1` |
| `name` | `llm_trace_S3_dev` |
| `output_length_drawn_at` | `episode build time (paired across controllers)` |
| `paths` | `['S3_dev.jsonl']` |

### Action space

`Discrete(5)`, mode `delta`.

| action | meaning |
|---|---|
| 0 | target -2 |
| 1 | target -1 |
| 2 | target +0 |
| 3 | target +1 |
| 4 | target +2 |

Targets are clamped to `[1, 4]` by the core. Exceeding the ceiling requires `allow_hypothetical_replicas=True`, which sets `hypothetical=True` on the pool and taints every result row; no registered task enables it.

### Observation vector

37 elements, `float32`, each scaled then clipped into `[0, 10.0]`. History is stacked oldest-first; `[t-0]` is the most recent scrape. Every element but the last is `kubegym.core.state.FEATURES[...](ClusterState)`, so the policy sees no more than a hand-written controller reading the same scrape.

Normalisation divisors are a-priori constants -- config fields or documented design constants of the task. None is derived from the episode in progress, because a statistic of the episode is information from the future and would break the fairness guarantee silently.

| idx | element | scale kind | divisor | source |
|---|---|---|---|---|
| 0 | `n_ready[t-3]` | replica | 4 | ClusterState |
| 1 | `n_booting[t-3]` | replica | 4 | ClusterState |
| 2 | `n_draining[t-3]` | replica | 4 | ClusterState |
| 3 | `target[t-3]` | replica | 4 | ClusterState |
| 4 | `n_running[t-3]` | concurrency | 32 | ClusterState |
| 5 | `n_waiting[t-3]` | concurrency | 32 | ClusterState |
| 6 | `sat_mean[t-3]` | unit | 1 | ClusterState |
| 7 | `sat_max[t-3]` | unit | 1 | ClusterState |
| 8 | `arrivals[t-3]` | arrivals | 16 | ClusterState |
| 9 | `n_ready[t-2]` | replica | 4 | ClusterState |
| 10 | `n_booting[t-2]` | replica | 4 | ClusterState |
| 11 | `n_draining[t-2]` | replica | 4 | ClusterState |
| 12 | `target[t-2]` | replica | 4 | ClusterState |
| 13 | `n_running[t-2]` | concurrency | 32 | ClusterState |
| 14 | `n_waiting[t-2]` | concurrency | 32 | ClusterState |
| 15 | `sat_mean[t-2]` | unit | 1 | ClusterState |
| 16 | `sat_max[t-2]` | unit | 1 | ClusterState |
| 17 | `arrivals[t-2]` | arrivals | 16 | ClusterState |
| 18 | `n_ready[t-1]` | replica | 4 | ClusterState |
| 19 | `n_booting[t-1]` | replica | 4 | ClusterState |
| 20 | `n_draining[t-1]` | replica | 4 | ClusterState |
| 21 | `target[t-1]` | replica | 4 | ClusterState |
| 22 | `n_running[t-1]` | concurrency | 32 | ClusterState |
| 23 | `n_waiting[t-1]` | concurrency | 32 | ClusterState |
| 24 | `sat_mean[t-1]` | unit | 1 | ClusterState |
| 25 | `sat_max[t-1]` | unit | 1 | ClusterState |
| 26 | `arrivals[t-1]` | arrivals | 16 | ClusterState |
| 27 | `n_ready[t-0]` | replica | 4 | ClusterState |
| 28 | `n_booting[t-0]` | replica | 4 | ClusterState |
| 29 | `n_draining[t-0]` | replica | 4 | ClusterState |
| 30 | `target[t-0]` | replica | 4 | ClusterState |
| 31 | `n_running[t-0]` | concurrency | 32 | ClusterState |
| 32 | `n_waiting[t-0]` | concurrency | 32 | ClusterState |
| 33 | `sat_mean[t-0]` | unit | 1 | ClusterState |
| 34 | `sat_max[t-0]` | unit | 1 | ClusterState |
| 35 | `arrivals[t-0]` | arrivals | 16 | ClusterState |
| 36 | `episode_progress` | unit | 20 | control steps taken / n_control_steps; the core's own controller interface decide(hist, t) is handed t, so this is not privileged |

| scale | value |
|---|---|
| `arrivals_scale` | 16 |
| `clip` | 10 |
| `concurrency_scale` | 32 |
| `per_replica_concurrency` | 8 |
| `preemptions_scale` | 100 |
| `replica_scale` | 4 |

### Reward

`reward = -cost; cost >= 0; higher reward is better`. Cost terms, all non-negative:

| term | unit | canonical weight |
|---|---|---|
| `slo` | reward per violating request | 0.1 |
| `resource` | reward per replica-second | 0.0166667 |
| `churn` | reward per replica added or removed | 0.05 |

Charged once at the final step, in addition: `slo_drain` (violations completed during the post-horizon drain), `resource_drain` (replica-seconds billed during the drain), `slo_unfinished` (requests that never completed). Same weights; `include_drain_in_reward = True`.

**SLO profile `llm`.** LLM token-streaming SLO. A completed request violates if its time-to-first-token exceeded `ttft_p95_s` or its mean time-between-tokens exceeded `tbt_p95_s`. An unfinished request violates if it has produced no first token `ttft_hard_s` after arrival.

| SLO target | value (s) |
|---|---|
| `tbt_p95_s` | 0.05 |
| `ttft_hard_s` | 5 |
| `ttft_p95_s` | 2 |

SLO targets come from the config field `slo`, `calibrated = false`. The **weights are a design choice, not a measurement**, and are not covered by the config's claims gate; they are recorded in `unverified_gym.md`. Every term is computable from router and orchestrator telemetry a real deployment has -- the reward never reads a request's ground-truth work (`reads_ground_truth_demand = False`).

### Seeds and splits

`reset(seed=s)` selects `(episode, work_seed)` deterministically as `pairs[s % len(pairs)]` within the env's split, so a rollout is reproducible from the Gym API alone. Splits are disjoint in work seed.

| split | work seeds | (episode, seed) pairs |
|---|---|---|
| `dev` | 16 seeds: 1000..1015 | 16 |
| `test` | 16 seeds: 2000..2015 | 16 |
| `train` | 64 seeds: 0..63 | 64 |

### Calibration and inherited biases

`llm_serving_l40s.json`: UNCALIBRATED OVERALL. Cold start, KV budget and the decode step-time model on batch<=8 are MEASURED on the GPU testbed (vLLM 0.25.1 / Qwen3-1.7B / single L40S 46GB). The per-class OUTPUT-LENGTH DISTRIBUTION is still a labelled placeholder from length_model.py, prefill throughput was never measured, and decode step time above batch 8 is an unmeasured affine extrapolation. No number produced by this simulator may be reported as an empirical result.

Fields flagged as biasing results favourably: `decode_batch_extrapolation_above_8`.

Inherited from INTERFACE.md section 8 (transcribed, not paraphrased):

| choice | direction | note |
|---|---|---|
| decode step time extrapolated affinely above batch 8 | **optimistic** | real engines saturate; makes under-provisioning look cheap. flagged_optimistic |
| prefill_tokens_per_s = 6000, never measured | **unknown sign** | every TTFT number is placeholder-derived |
| chunked prefill not modelled | **mixed** | overstates the inter-token spike on admission; understates long-prompt TTFT under load |
| teardown not billed (0.54 s, weak evidence n=2) | **optimistic** | < 1 % of a scale-down cycle |
| metric staleness modelled as zero | **optimistic about the controller's information** | real scrape interval is 15 s |
| output-length distribution is a placeholder lognormal | **sets the difficulty of the whole problem** | dominant uncalibrated input |

Known infidelities declared by the service model itself:

- prefill_tokens_per_s is a PLACEHOLDER (never measured); all TTFT numbers are placeholder-derived
- decode step time above batch 8 is an unmeasured affine extrapolation; real engines saturate, so this is OPTIMISTIC about large batches
- chunked prefill not modelled: OVERSTATES the inter-token spike on admission, UNDERSTATES long-prompt TTFT under load
- preemption victim order (most-recently-admitted) was not read from vLLM source

Circularity warnings any paper using this core must repeat:

- A controller whose belief is over the SAME distribution the simulator samples from is handed a perfectly specified prior, so a belief-vs-point gap is close to guaranteed by construction and its magnitude is uninformative.
- In the shipped traces thinking_flip_prob = 0.0, so the class signal is a NOISELESS proxy for the length class; set class_predictor_accuracy < 1.0 to inject predictor error.

### Task caveats

- WORKLOAD IS A TEST FIXTURE, NOT THE BENCHMARK CORPUS. The 300 s reference traces in kubegym/tests/data/ are the core's test fixtures (their README says so explicitly). At a 15 s control interval an episode is 20 control steps, which is short for RL: credit for a 31.9 s cold start spans two steps, so roughly a tenth of the episode is spent paying for any single scale-up. Treat these tasks as conformance and smoke-test targets; the workload corpus track supplies the episodes a learning-curve claim should use.
- CALIBRATION: the config is calibrated=false, so no return, reward or cost from this task is an empirical result. The SLO targets the reward is evaluated against are themselves a labelled placeholder (config field `slo`, required_for_claims=true).
- THE SLO TERM IS NEARLY BINARY ON THIS WORKLOAD. Measured over three work seeds at fixed replica counts: 254 violations in total at k=1, and exactly 0 at k=2, k=3 and k=4. At k=1 the KV budget fills, recompute-preemption fires and admission is delayed past ttft_p95_s=2.0 s; from k=2 up, the placeholder prefill throughput (6000 tok/s) makes TTFT negligible and the fitted decode step time (about 17 ms at these batch sizes) is far below tbt_p95_s=0.05 s. So the SLO term behaves as an indicator on 'did the policy sit at one replica under load' rather than as a graded penalty, and the interesting part of the objective on this task is the cost/churn trade-off between k=2 and k=4. Both the targets and the prefill constant are placeholders, so this cliff is a property of uncalibrated inputs and would move under calibration.

## `KubeGym/RequestService-Poisson-Abs-v0`

**Calibration status: `calibrated = false`.** Blocked by `replica_cold_start_s`, `rs_max_concurrency`, `rs_service_time_distribution`, `slo`. No return, reward or cost from this task is an empirical result.

| field | value |
|---|---|
| service model | `request_service` |
| config | `request_service_default.json` |
| workload | `poisson_4rps_300s` |
| episode length | 20 control steps x 15.0 s = 300.0 s |
| distinct episodes | 1 |
| initial replicas `k0` | 1 |
| replica range | [1, 20]  (ceiling is a placeholder, not a measured limit) |
| action space | `Discrete(20)`  (absolute mode) |
| observation space | `Box(0.0, 10.0, (37,), float32)` |
| post-horizon drain cap | 3000.0 s |
| drain charged to reward | True |

### Workload

Homogeneous Poisson arrivals at 4 rps over 300 s (generator seed 7, ~1200 arrivals), service times drawn per request at build time from the PLACEHOLDER lognormal (mu=0, sigma=1). A deliberately uninteresting reference process, as the core's own docstring for `poisson_arrivals` says.

| manifest key | value |
|---|---|
| `arrival_generator_seed` | `7` |
| `arrival_process` | `homogeneous Poisson` |
| `calibrated` | `False` |
| `class` | `RequestServiceSource` |
| `n_records` | `1231` |
| `name` | `rs_poisson_4rps` |
| `note` | `Arrival times are FIXED by arrival_generator_seed, so this task has one episode; per-episode variation comes from the work realisation (service times drawn at build time from the placeholder lognormal).` |
| `rate_rps` | `4.0` |
| `service_time_distribution` | `{'family': 'lognormal', 'mu': 0.0, 'sigma': 1.0, 'min_s': 0.01, 'max_s': 600.0}` |

### Action space

`Discrete(20)`, mode `absolute`.

| action | meaning |
|---|---|
| 0 | target = 1 |
| 1 | target = 2 |
| 2 | target = 3 |
| 3 | target = 4 |
| 4 | target = 5 |
| 5 | target = 6 |
| 6 | target = 7 |
| 7 | target = 8 |
| 8 | target = 9 |
| 9 | target = 10 |
| 10 | target = 11 |
| 11 | target = 12 |
| 12 | target = 13 |
| 13 | target = 14 |
| 14 | target = 15 |
| 15 | target = 16 |
| 16 | target = 17 |
| 17 | target = 18 |
| 18 | target = 19 |
| 19 | target = 20 |

Targets are clamped to `[1, 20]` by the core. The ceiling here is a placeholder rather than a measured limit, so a target near it is not automatically hypothetical.

### Observation vector

37 elements, `float32`, each scaled then clipped into `[0, 10.0]`. History is stacked oldest-first; `[t-0]` is the most recent scrape. Every element but the last is `kubegym.core.state.FEATURES[...](ClusterState)`, so the policy sees no more than a hand-written controller reading the same scrape.

Normalisation divisors are a-priori constants -- config fields or documented design constants of the task. None is derived from the episode in progress, because a statistic of the episode is information from the future and would break the fairness guarantee silently.

| idx | element | scale kind | divisor | source |
|---|---|---|---|---|
| 0 | `n_ready[t-3]` | replica | 20 | ClusterState |
| 1 | `n_booting[t-3]` | replica | 20 | ClusterState **identically zero on this task, see caveats** |
| 2 | `n_draining[t-3]` | replica | 20 | ClusterState |
| 3 | `target[t-3]` | replica | 20 | ClusterState |
| 4 | `n_running[t-3]` | concurrency | 80 | ClusterState |
| 5 | `n_waiting[t-3]` | concurrency | 80 | ClusterState |
| 6 | `sat_mean[t-3]` | unit | 1 | ClusterState |
| 7 | `sat_max[t-3]` | unit | 1 | ClusterState |
| 8 | `arrivals[t-3]` | arrivals | 64 | ClusterState |
| 9 | `n_ready[t-2]` | replica | 20 | ClusterState |
| 10 | `n_booting[t-2]` | replica | 20 | ClusterState **identically zero on this task, see caveats** |
| 11 | `n_draining[t-2]` | replica | 20 | ClusterState |
| 12 | `target[t-2]` | replica | 20 | ClusterState |
| 13 | `n_running[t-2]` | concurrency | 80 | ClusterState |
| 14 | `n_waiting[t-2]` | concurrency | 80 | ClusterState |
| 15 | `sat_mean[t-2]` | unit | 1 | ClusterState |
| 16 | `sat_max[t-2]` | unit | 1 | ClusterState |
| 17 | `arrivals[t-2]` | arrivals | 64 | ClusterState |
| 18 | `n_ready[t-1]` | replica | 20 | ClusterState |
| 19 | `n_booting[t-1]` | replica | 20 | ClusterState **identically zero on this task, see caveats** |
| 20 | `n_draining[t-1]` | replica | 20 | ClusterState |
| 21 | `target[t-1]` | replica | 20 | ClusterState |
| 22 | `n_running[t-1]` | concurrency | 80 | ClusterState |
| 23 | `n_waiting[t-1]` | concurrency | 80 | ClusterState |
| 24 | `sat_mean[t-1]` | unit | 1 | ClusterState |
| 25 | `sat_max[t-1]` | unit | 1 | ClusterState |
| 26 | `arrivals[t-1]` | arrivals | 64 | ClusterState |
| 27 | `n_ready[t-0]` | replica | 20 | ClusterState |
| 28 | `n_booting[t-0]` | replica | 20 | ClusterState **identically zero on this task, see caveats** |
| 29 | `n_draining[t-0]` | replica | 20 | ClusterState |
| 30 | `target[t-0]` | replica | 20 | ClusterState |
| 31 | `n_running[t-0]` | concurrency | 80 | ClusterState |
| 32 | `n_waiting[t-0]` | concurrency | 80 | ClusterState |
| 33 | `sat_mean[t-0]` | unit | 1 | ClusterState |
| 34 | `sat_max[t-0]` | unit | 1 | ClusterState |
| 35 | `arrivals[t-0]` | arrivals | 64 | ClusterState |
| 36 | `episode_progress` | unit | 20 | control steps taken / n_control_steps; the core's own controller interface decide(hist, t) is handed t, so this is not privileged |

| scale | value |
|---|---|
| `arrivals_scale` | 64 |
| `clip` | 10 |
| `concurrency_scale` | 80 |
| `per_replica_concurrency` | 4 |
| `preemptions_scale` | 100 |
| `replica_scale` | 20 |

### Reward

`reward = -cost; cost >= 0; higher reward is better`. Cost terms, all non-negative:

| term | unit | canonical weight |
|---|---|---|
| `slo` | reward per violating request | 0.05 |
| `resource` | reward per replica-second | 0.00333333 |
| `churn` | reward per replica added or removed | 0.05 |

Charged once at the final step, in addition: `slo_drain` (violations completed during the post-horizon drain), `resource_drain` (replica-seconds billed during the drain), `slo_unfinished` (requests that never completed). Same weights; `include_drain_in_reward = True`.

**SLO profile `request_service`.** Request/response SLO. A completed request violates if its end-to-end latency exceeded `latency_p95_s` or its queueing delay exceeded `queue_delay_p95_s`. An unfinished request violates if it is still in the system `latency_hard_s` after arrival.

| SLO target | value (s) |
|---|---|
| `latency_hard_s` | 5 |
| `latency_p95_s` | 1 |
| `queue_delay_p95_s` | 0.5 |

SLO targets come from the config field `slo`, `calibrated = false`. The **weights are a design choice, not a measurement**, and are not covered by the config's claims gate; they are recorded in `unverified_gym.md`. Every term is computable from router and orchestrator telemetry a real deployment has -- the reward never reads a request's ground-truth work (`reads_ground_truth_demand = False`).

### Seeds and splits

`reset(seed=s)` selects `(episode, work_seed)` deterministically as `pairs[s % len(pairs)]` within the env's split, so a rollout is reproducible from the Gym API alone. Splits are disjoint in work seed.

| split | work seeds | (episode, seed) pairs |
|---|---|---|
| `dev` | 16 seeds: 1000..1015 | 16 |
| `test` | 16 seeds: 2000..2015 | 16 |
| `train` | 64 seeds: 0..63 | 64 |

### Calibration and inherited biases

`request_service_default.json`: UNCALIBRATED, ENTIRELY. No constant in this config was measured. The request-service model exists so that a serverless/microservice workload corpus has a physics layer to land on and so that the ServiceModel seam is exercised by a model structurally unlike LLM decoding (no token stream, no growing memory footprint, no preemption). No number produced with this config may be reported as an empirical result. Cold start, per-replica concurrency and the service-time distribution are the three constants that would need real measurement (or a documented citation to a measured public dataset) before any claim.

Fields flagged as biasing results favourably: `metric_scrape_interval_s`, `rs_contention_slowdown`.

Inherited from INTERFACE.md section 8 (transcribed, not paraphrased):

| choice | direction | note |
|---|---|---|
| teardown not billed (0.54 s, weak evidence n=2) | **optimistic** | < 1 % of a scale-down cycle |
| parallel bring-up (source simulator; not the default here) | **optimistic** | scale-up looks faster than the verified sequential path |
| metric staleness modelled as zero | **optimistic about the controller's information** | real scrape interval is 15 s |
| rs_contention_slowdown = 0.0 | **optimistic** | a loaded replica is as fast as an idle one |
| no load shedding / drops in request_service | **pessimistic on latency, optimistic on success rate** | queueing delay is the only symptom of overload |

Known infidelities declared by the service model itself:

- rs_contention_slowdown default 0.0 makes a loaded replica as fast as an idle one: OPTIMISTIC, under-provisioning looks cheaper than on real hardware
- no per-request cold start (only replica-level); a warm-container platform would differ
- unbounded queue, no admission control, no drops or timeouts
- service time is independent of replica state and of the request payload

### Task caveats

- EVERY CONSTANT IN THIS TASK'S CONFIG IS A PLACEHOLDER. request_service exists to exercise the ServiceModel seam with physics structurally unlike LLM decoding; nothing it produces is an empirical result about any real platform.
- The arrival times are fixed by the generator seed, so this task has ONE episode. Per-episode variation comes only from the service-time realisation, which the work seed selects. A task whose arrivals also vary needs a workload source that indexes arrivals by episode -- that is the corpus track's job.
- n_replicas_max = 20 here is a placeholder ceiling, NOT a measured physical limit (unlike the LLM config's 4), so a target near it is not hypothetical.
- THE ACTUATION LAG IS INVISIBLE ON THIS TASK. The placeholder cold start is 2.0 s and the control interval is 15.0 s, so a scale-up completes entirely inside one control step and `n_booting` is zero at every scrape -- measured over 1400 steps across constant, ramp, alternating and hold-then-drop policies in both action modes. The four `n_booting[t-k]` elements of the observation are therefore constant zero, and the policy cannot see a scale-up in flight because there is nothing in flight to see. This makes the task a no-lag control problem, structurally easier than the LLM tasks, whose measured 31.9 s cold start spans 2.13 control intervals. It is an artefact of a placeholder constant, not a property of any real platform.
- THE SLO TARGET IS UNREACHABLE FOR ROUGHLY HALF THE REQUESTS. The placeholder service-time lognormal has median 1.0 s and the placeholder `latency_p95_s` is also 1.0 s, so about half of all requests miss the target on service time alone, before any queueing. Measured at the ceiling k=20 (80 concurrency slots against an offered load of about 6.6 requests in service): 582/580/600 violations out of 1231 requests on seeds 0/1/2, i.e. 47-49%. No policy can drive the SLO term below that floor, so on this task the term is largely a constant offset and the reward is dominated by the resource term.
- CALIBRATION: the config is calibrated=false, so no return, reward or cost from this task is an empirical result. The SLO targets the reward is evaluated against are themselves a labelled placeholder (config field `slo`, required_for_claims=true).

## `KubeGym/RequestService-Poisson-Delta-v0`

**Calibration status: `calibrated = false`.** Blocked by `replica_cold_start_s`, `rs_max_concurrency`, `rs_service_time_distribution`, `slo`. No return, reward or cost from this task is an empirical result.

| field | value |
|---|---|
| service model | `request_service` |
| config | `request_service_default.json` |
| workload | `poisson_4rps_300s` |
| episode length | 20 control steps x 15.0 s = 300.0 s |
| distinct episodes | 1 |
| initial replicas `k0` | 1 |
| replica range | [1, 20]  (ceiling is a placeholder, not a measured limit) |
| action space | `Discrete(5)`  (delta mode) |
| observation space | `Box(0.0, 10.0, (37,), float32)` |
| post-horizon drain cap | 3000.0 s |
| drain charged to reward | True |

### Workload

Homogeneous Poisson arrivals at 4 rps over 300 s (generator seed 7, ~1200 arrivals), service times drawn per request at build time from the PLACEHOLDER lognormal (mu=0, sigma=1). A deliberately uninteresting reference process, as the core's own docstring for `poisson_arrivals` says.

| manifest key | value |
|---|---|
| `arrival_generator_seed` | `7` |
| `arrival_process` | `homogeneous Poisson` |
| `calibrated` | `False` |
| `class` | `RequestServiceSource` |
| `n_records` | `1231` |
| `name` | `rs_poisson_4rps` |
| `note` | `Arrival times are FIXED by arrival_generator_seed, so this task has one episode; per-episode variation comes from the work realisation (service times drawn at build time from the placeholder lognormal).` |
| `rate_rps` | `4.0` |
| `service_time_distribution` | `{'family': 'lognormal', 'mu': 0.0, 'sigma': 1.0, 'min_s': 0.01, 'max_s': 600.0}` |

### Action space

`Discrete(5)`, mode `delta`.

| action | meaning |
|---|---|
| 0 | target -2 |
| 1 | target -1 |
| 2 | target +0 |
| 3 | target +1 |
| 4 | target +2 |

Targets are clamped to `[1, 20]` by the core. The ceiling here is a placeholder rather than a measured limit, so a target near it is not automatically hypothetical.

### Observation vector

37 elements, `float32`, each scaled then clipped into `[0, 10.0]`. History is stacked oldest-first; `[t-0]` is the most recent scrape. Every element but the last is `kubegym.core.state.FEATURES[...](ClusterState)`, so the policy sees no more than a hand-written controller reading the same scrape.

Normalisation divisors are a-priori constants -- config fields or documented design constants of the task. None is derived from the episode in progress, because a statistic of the episode is information from the future and would break the fairness guarantee silently.

| idx | element | scale kind | divisor | source |
|---|---|---|---|---|
| 0 | `n_ready[t-3]` | replica | 20 | ClusterState |
| 1 | `n_booting[t-3]` | replica | 20 | ClusterState **identically zero on this task, see caveats** |
| 2 | `n_draining[t-3]` | replica | 20 | ClusterState |
| 3 | `target[t-3]` | replica | 20 | ClusterState |
| 4 | `n_running[t-3]` | concurrency | 80 | ClusterState |
| 5 | `n_waiting[t-3]` | concurrency | 80 | ClusterState |
| 6 | `sat_mean[t-3]` | unit | 1 | ClusterState |
| 7 | `sat_max[t-3]` | unit | 1 | ClusterState |
| 8 | `arrivals[t-3]` | arrivals | 64 | ClusterState |
| 9 | `n_ready[t-2]` | replica | 20 | ClusterState |
| 10 | `n_booting[t-2]` | replica | 20 | ClusterState **identically zero on this task, see caveats** |
| 11 | `n_draining[t-2]` | replica | 20 | ClusterState |
| 12 | `target[t-2]` | replica | 20 | ClusterState |
| 13 | `n_running[t-2]` | concurrency | 80 | ClusterState |
| 14 | `n_waiting[t-2]` | concurrency | 80 | ClusterState |
| 15 | `sat_mean[t-2]` | unit | 1 | ClusterState |
| 16 | `sat_max[t-2]` | unit | 1 | ClusterState |
| 17 | `arrivals[t-2]` | arrivals | 64 | ClusterState |
| 18 | `n_ready[t-1]` | replica | 20 | ClusterState |
| 19 | `n_booting[t-1]` | replica | 20 | ClusterState **identically zero on this task, see caveats** |
| 20 | `n_draining[t-1]` | replica | 20 | ClusterState |
| 21 | `target[t-1]` | replica | 20 | ClusterState |
| 22 | `n_running[t-1]` | concurrency | 80 | ClusterState |
| 23 | `n_waiting[t-1]` | concurrency | 80 | ClusterState |
| 24 | `sat_mean[t-1]` | unit | 1 | ClusterState |
| 25 | `sat_max[t-1]` | unit | 1 | ClusterState |
| 26 | `arrivals[t-1]` | arrivals | 64 | ClusterState |
| 27 | `n_ready[t-0]` | replica | 20 | ClusterState |
| 28 | `n_booting[t-0]` | replica | 20 | ClusterState **identically zero on this task, see caveats** |
| 29 | `n_draining[t-0]` | replica | 20 | ClusterState |
| 30 | `target[t-0]` | replica | 20 | ClusterState |
| 31 | `n_running[t-0]` | concurrency | 80 | ClusterState |
| 32 | `n_waiting[t-0]` | concurrency | 80 | ClusterState |
| 33 | `sat_mean[t-0]` | unit | 1 | ClusterState |
| 34 | `sat_max[t-0]` | unit | 1 | ClusterState |
| 35 | `arrivals[t-0]` | arrivals | 64 | ClusterState |
| 36 | `episode_progress` | unit | 20 | control steps taken / n_control_steps; the core's own controller interface decide(hist, t) is handed t, so this is not privileged |

| scale | value |
|---|---|
| `arrivals_scale` | 64 |
| `clip` | 10 |
| `concurrency_scale` | 80 |
| `per_replica_concurrency` | 4 |
| `preemptions_scale` | 100 |
| `replica_scale` | 20 |

### Reward

`reward = -cost; cost >= 0; higher reward is better`. Cost terms, all non-negative:

| term | unit | canonical weight |
|---|---|---|
| `slo` | reward per violating request | 0.05 |
| `resource` | reward per replica-second | 0.00333333 |
| `churn` | reward per replica added or removed | 0.05 |

Charged once at the final step, in addition: `slo_drain` (violations completed during the post-horizon drain), `resource_drain` (replica-seconds billed during the drain), `slo_unfinished` (requests that never completed). Same weights; `include_drain_in_reward = True`.

**SLO profile `request_service`.** Request/response SLO. A completed request violates if its end-to-end latency exceeded `latency_p95_s` or its queueing delay exceeded `queue_delay_p95_s`. An unfinished request violates if it is still in the system `latency_hard_s` after arrival.

| SLO target | value (s) |
|---|---|
| `latency_hard_s` | 5 |
| `latency_p95_s` | 1 |
| `queue_delay_p95_s` | 0.5 |

SLO targets come from the config field `slo`, `calibrated = false`. The **weights are a design choice, not a measurement**, and are not covered by the config's claims gate; they are recorded in `unverified_gym.md`. Every term is computable from router and orchestrator telemetry a real deployment has -- the reward never reads a request's ground-truth work (`reads_ground_truth_demand = False`).

### Seeds and splits

`reset(seed=s)` selects `(episode, work_seed)` deterministically as `pairs[s % len(pairs)]` within the env's split, so a rollout is reproducible from the Gym API alone. Splits are disjoint in work seed.

| split | work seeds | (episode, seed) pairs |
|---|---|---|
| `dev` | 16 seeds: 1000..1015 | 16 |
| `test` | 16 seeds: 2000..2015 | 16 |
| `train` | 64 seeds: 0..63 | 64 |

### Calibration and inherited biases

`request_service_default.json`: UNCALIBRATED, ENTIRELY. No constant in this config was measured. The request-service model exists so that a serverless/microservice workload corpus has a physics layer to land on and so that the ServiceModel seam is exercised by a model structurally unlike LLM decoding (no token stream, no growing memory footprint, no preemption). No number produced with this config may be reported as an empirical result. Cold start, per-replica concurrency and the service-time distribution are the three constants that would need real measurement (or a documented citation to a measured public dataset) before any claim.

Fields flagged as biasing results favourably: `metric_scrape_interval_s`, `rs_contention_slowdown`.

Inherited from INTERFACE.md section 8 (transcribed, not paraphrased):

| choice | direction | note |
|---|---|---|
| teardown not billed (0.54 s, weak evidence n=2) | **optimistic** | < 1 % of a scale-down cycle |
| parallel bring-up (source simulator; not the default here) | **optimistic** | scale-up looks faster than the verified sequential path |
| metric staleness modelled as zero | **optimistic about the controller's information** | real scrape interval is 15 s |
| rs_contention_slowdown = 0.0 | **optimistic** | a loaded replica is as fast as an idle one |
| no load shedding / drops in request_service | **pessimistic on latency, optimistic on success rate** | queueing delay is the only symptom of overload |

Known infidelities declared by the service model itself:

- rs_contention_slowdown default 0.0 makes a loaded replica as fast as an idle one: OPTIMISTIC, under-provisioning looks cheaper than on real hardware
- no per-request cold start (only replica-level); a warm-container platform would differ
- unbounded queue, no admission control, no drops or timeouts
- service time is independent of replica state and of the request payload

### Task caveats

- EVERY CONSTANT IN THIS TASK'S CONFIG IS A PLACEHOLDER. request_service exists to exercise the ServiceModel seam with physics structurally unlike LLM decoding; nothing it produces is an empirical result about any real platform.
- The arrival times are fixed by the generator seed, so this task has ONE episode. Per-episode variation comes only from the service-time realisation, which the work seed selects. A task whose arrivals also vary needs a workload source that indexes arrivals by episode -- that is the corpus track's job.
- n_replicas_max = 20 here is a placeholder ceiling, NOT a measured physical limit (unlike the LLM config's 4), so a target near it is not hypothetical.
- THE ACTUATION LAG IS INVISIBLE ON THIS TASK. The placeholder cold start is 2.0 s and the control interval is 15.0 s, so a scale-up completes entirely inside one control step and `n_booting` is zero at every scrape -- measured over 1400 steps across constant, ramp, alternating and hold-then-drop policies in both action modes. The four `n_booting[t-k]` elements of the observation are therefore constant zero, and the policy cannot see a scale-up in flight because there is nothing in flight to see. This makes the task a no-lag control problem, structurally easier than the LLM tasks, whose measured 31.9 s cold start spans 2.13 control intervals. It is an artefact of a placeholder constant, not a property of any real platform.
- THE SLO TARGET IS UNREACHABLE FOR ROUGHLY HALF THE REQUESTS. The placeholder service-time lognormal has median 1.0 s and the placeholder `latency_p95_s` is also 1.0 s, so about half of all requests miss the target on service time alone, before any queueing. Measured at the ceiling k=20 (80 concurrency slots against an offered load of about 6.6 requests in service): 582/580/600 violations out of 1231 requests on seeds 0/1/2, i.e. 47-49%. No policy can drive the SLO term below that floor, so on this task the term is largely a constant offset and the reward is dominated by the resource term.
- CALIBRATION: the config is calibrated=false, so no return, reward or cost from this task is an empirical result. The SLO targets the reward is evaluated against are themselves a labelled placeholder (config field `slo`, required_for_claims=true).
