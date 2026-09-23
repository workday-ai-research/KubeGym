# `kubegym.gym` — RL-facing API reference (Phase 1)

The Gymnasium layer over the KubeGym core. Depends on `gymnasium` (tested against 1.3.0),
`numpy` and the core; nothing else was installed for this phase.

**Calibration status: `calibrated = false` for every registered task.** Both shipped configs
have `required_for_claims` fields that are labelled placeholders, so no reward, return or cost
this layer produces is an empirical result. The flag is on every `info` dict, every
`list_tasks()` row, and every stamped episode row. See `unverified_gym.md` for the register of
what is unverified *in this layer* on top of `unverified_core.md`.

---

## 1. Quick start

```python
import gymnasium
import kubegym.gym                      # importing this registers the builtin tasks

for row in kubegym.gym.list_tasks():
    print(row["id"], row["calibrated"], row["n_control_steps"])

env = gymnasium.make("KubeGym/LLMServing-S1-Delta-v0", split="train")
obs, info = env.reset(seed=0)
while True:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break
print(info["episode_cost"])            # provenance-stamped episode row
```

Or without Gymnasium's registry, for a task you built but did not register:

```python
from kubegym.gym import KubeGymEnv, get_task
env = KubeGymEnv(get_task("LLMServing-S1"), action_mode="absolute", split="dev")
```

`KubeGymEnv` passes `gymnasium.utils.env_checker.check_env` for every registered task in both
action modes, with no warnings (see `gym_tests_report.md`).

### Construction

| argument | default | meaning |
|---|---|---|
| `task` | — | a `TaskSpec`, a task name (`"LLMServing-S1"`), or a full id |
| `action_mode` | `"delta"` | `"delta"` or `"absolute"`; see §3 |
| `split` | `"train"` | `"train"` / `"dev"` / `"test"`; selects which work seeds `reset` may draw |
| `allow_hypothetical_replicas` | `False` | raise the action ceiling above the config's `n_replicas_max`. Warns at construction and taints every row; no registered task enables it |
| `hypothetical_replicas_max` | `n_replicas_max + 4` | the raised ceiling, only read when the above is `True` |
| `include_drain_in_reward` | task's canonical value (`True`) | charge post-horizon drain cost to the final step; see §4.3 |
| `render_mode` | `None` | must stay `None`; an autoscaling episode has no frame to draw and passing anything raises |

One `KubeGymEnv` owns one `Simulator`, one service model and **its own** `WorkloadSource`
instance — two envs never share a source, which is asserted in
`test_gym_determinism.py::test_two_envs_do_not_share_workload_state`.

---

## 2. Episode structure

```
reset(seed=s)                       -> t = 0 scrape, 1 entry in sim.hist
step() x n_control_steps            -> one advance_control_interval() each
    at the last step                -> Simulator.drain_all(cap), then the episode closes
```

`n_control_steps = horizon_s / control_interval_s`, and the division must be exact — a
`TaskSpec` whose horizon does not divide is refused at registration rather than silently
rounding the episode length. Every builtin task is 300 s / 15 s = **20 control steps**.

One `env.step()` is exactly one `Simulator.advance_control_interval()`. The action is applied
via `set_target_replicas` **before** the interval is advanced, so the actuation and its effect
are in the same step. The 15 s control interval is the config's `control_interval_s`, which
matches the Kubernetes HPA default sync period, so every controller in a comparison — analytic
or learned — decides at the same cadence.

### `terminated` vs `truncated`

* **`terminated = True`** at the horizon. The workload is exhausted and the post-horizon drain
  has closed the episode out; there is no continuation to bootstrap into. This is a
  finite-horizon episodic task, not an ongoing process cut short.
* **`truncated = True`** only when `Simulator.drain_all` hit its cap with work still in flight
  (the core's `truncated_drain`). Then the episode genuinely did not finish. With the default
  cap of `10 x horizon` this does not occur on the shipped tasks.

No `TimeLimit` wrapper is applied by `gymnasium.make`: a second, wrapper-level horizon would
let the episode length disagree with the task spec.

### Seeds, episodes and splits

The core takes two integers: `episode` selects the trace, `seed` the work realisation (for the
LLM tasks, the per-request output lengths drawn at build time). A Gymnasium seed is one
integer, so a task declares a list of `(episode, work_seed)` pairs per split and

```
reset(seed=s)  ->  pairs[s % len(pairs)]
```

which is total and deterministic, so a rollout replays byte-identically from the Gym API alone.
`reset(seed=None)` draws a pair with `self.np_random`. To pin an exact episode:

```python
env.reset(options={"episode": 1, "work_seed": 1000})
```

Both keys are required together, and the work seed must belong to the env's split — passing a
`test` seed to a `split="train"` env raises, because a test episode inside a training run under
another name is exactly the failure that split separation exists to prevent.

Default work seeds, **disjoint by construction**: `train` = 0–63, `dev` = 1000–1015,
`test` = 2000–2015. Pair count is `len(work_seeds) x n_episodes`.

---

## 3. Action modes

Both are supported for every task; each registered task exists under both ids.

### `delta` — `Discrete(5)`

Increments applied to the **current target**, then clamped:

| action | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| increment | −2 | −1 | 0 | +1 | +2 |

The set is symmetric and always contains 0, so a hold action exists (a `TaskSpec` without one
is refused). Because the action is relative, `target` is part of the observation — otherwise
the policy would be acting on a quantity it cannot see.

### `absolute` — `Discrete(n_replicas_max - n_replicas_min + 1)`

Action `a` requests `n_replicas_min + a`. For `llm_serving_l40s.json` that is `Discrete(4)`
over targets 1–4; for `request_service_default.json`, `Discrete(20)` over 1–20.

### The replica ceiling

Targets are clamped to `[n_replicas_min, n_replicas_max]` inside the core. The env reads the
post-clamp value back from `sim.target` rather than tracking its own, so it can never disagree
with the pool about what was actuated; `info["action_clamped"]` records when a request was
clipped.

For `llm_serving_l40s.json`, **`n_replicas_max = 4` is a measured physical limit** — 4
co-resident vLLM replicas on one L40S 46 GB. Exceeding it requires the core's
`allow_hypothetical_replicas`, which sets `hypothetical=True` on the pool and taints every
result row. This layer never enables it implicitly: turning it on emits a `RuntimeWarning`
naming the taint, and `info["hypothetical_replica_count_used"]` reports the pool's flag on
every step. For `request_service_default.json` the ceiling of 20 is itself a placeholder, not a
measured limit, so a target near it is not hypothetical.

---

## 4. The reward

### 4.1 Sign convention and terms

`cost >= 0` always, and **`reward = -cost`**. Higher is better; 0 is an unattainable upper
bound, since holding `n_replicas_min` replicas already accrues resource cost. Nothing is
rescaled to a "return of 1.0 = solved" convention: there is no known optimum for these tasks
and inventing a normaliser would imply one.

| term | raw quantity | unit of the weight |
|---|---|---|
| `slo` | requests whose SLO outcome was decided in this interval, and violated | reward / violating request |
| `resource` | replica-seconds billed during this interval | reward / replica-second |
| `churn` | `abs(target_after - target_before)` | reward / replica added or removed |

Charged once, at the final step, in addition:

| term | raw quantity |
|---|---|
| `slo_drain` | violating requests that completed during the post-horizon drain |
| `resource_drain` | replica-seconds billed during the post-horizon drain |
| `slo_unfinished` | requests that never completed (drain hit its cap) |

`total = slo + resource + churn + slo_drain + resource_drain + slo_unfinished`, and
`reward = -total`. `info["cost"]` reports every term, every raw count (`raw_*`), the weights and
the units, so an episode can be re-weighted after the fact without re-running it. The identity
`sum(terms) == total == -reward` is asserted on every step of every task in
`test_gym_cost.py`.

Replica-seconds accrue for **every** replica that exists — booting, serving and draining — which
is what makes thrash expensive; that is the core's billing rule, not this layer's.

### 4.2 SLO profiles

The targets come from the config's `slo` field (a `required_for_claims` placeholder in both
shipped configs), so every controller in a comparison is scored against identical targets.

**`llm`** (`llm_serving`) — targets `ttft_p95_s`, `tbt_p95_s`, `ttft_hard_s`. A completed
request violates if its time-to-first-token exceeded `ttft_p95_s` or its mean
time-between-tokens exceeded `tbt_p95_s`. An unfinished request violates if it has produced no
first token `ttft_hard_s` after arrival.

**`request_service`** — targets `latency_p95_s`, `latency_hard_s`, `queue_delay_p95_s`. A
completed request violates if its end-to-end latency exceeded `latency_p95_s` or its queueing
delay exceeded `queue_delay_p95_s`. An unfinished request violates if it is still in the system
`latency_hard_s` after arrival.

Each request is scored **once**, at the first control boundary at which its outcome is decided:
either it completed inside the window, or it has already been waiting past the hard threshold,
which is observable at that moment without knowing when it will finish. Double counting is
pinned by `test_gym_cost.py::test_pending_and_completed_scoring_do_not_double_count` and by an
end-of-episode accounting identity (`len(scored) + unfinished == n_requests`).

**No lookahead, and no privileged information either.** Every quantity the reward reads —
`ttft`, `mean_tbt`, `latency`, `queue_delay`, billed replica-seconds, the controller's own
actions — is one a real deployment computes from router and orchestrator telemetry. The cost
model never reads `Request.demand` (ground-truth total work) or its aliases, which is checked
mechanically by grepping `kubegym/gym/cost.py` for those attribute names
(`test_cost_model_source_reads_no_ground_truth`). So the benchmark's objective is implementable
on the cluster the physics were measured on, rather than being a simulator artefact.

### 4.3 Why the post-horizon drain is charged

Work still in the system at the horizon keeps costing replica-seconds and keeps missing its SLO
during the drain. If that were dropped, the optimal end-of-episode policy would be to scale to
`n_replicas_min` a few steps early and push the backlog into an unpriced window — a boundary
artefact an RL agent finds quickly and an analytic controller never looks for, i.e. one that
would flatter RL specifically. So the drain is charged to the final step, as three separately
reported terms. This is delayed reward, not lookahead: the agent observes nothing about the
drain before acting.

`include_drain_in_reward=False` restores the unpriced tail for anyone who wants to measure the
size of the artefact. On `LLMServing-S1` at a fixed `k=1`, dropping it improves the return by
exactly `resource_drain + slo_drain + slo_unfinished`
(`test_unpriced_tail_is_measurably_cheaper_when_the_drain_is_not_charged`).

### 4.4 Canonical weights

One weighting per task, so two papers reporting the same id report the same number.

| task | `slo` | `resource` | `churn` |
|---|---|---|---|
| `LLMServing-S1` / `-S3` / `-Mixed` | 0.10 | 1/60 = 0.0166667 | 0.05 |
| `RequestService-Poisson` | 0.05 | 1/300 = 0.00333333 | 0.05 |

`resource` is `1 / (control_interval_s x n_replicas_max)`, so running at the ceiling for a whole
control interval costs exactly 1.0 — that is the only reason to prefer it to a round number: it
gives the resource term the same meaning across tasks with different ceilings. `slo` and `churn`
are round numbers chosen so a handful of violations or a one-replica move is commensurate with a
fraction of the ceiling cost.

**The weights are a design choice, not a measurement**, and they are not physical constants, so
the config's claims gate does not cover them — a config could in principle be fully calibrated
while the weights remain arbitrary. They are recorded in `unverified_gym.md` and surfaced in
`info["cost"]["weights"]` and every env card. The `churn` term in particular is *shaping*, not a
physical cost: boot and drain time are already billed as replica-seconds, so `churn` charges a
scaling move a second time. Its consequence for baseline comparisons is in §7.

Non-negativity of every weight is enforced at construction — that is exactly what makes the cost
monotone in each term, which is the property the tests pin.

---

## 5. The observation vector

37 `float32` elements, each scaled by an a-priori constant then clipped into
`[0, clip]` with `clip = 10.0`; the space is `Box(0.0, 10.0, (37,), float32)`.

Layout: 9 features x 4 history lags, **oldest first**, features as the inner loop, then one
appended element. Lag `[t-0]` is the most recent scrape; the window is zero-padded at episode
start by the core's `ObservationSpec`.

| idx | element | raw source | divisor (LLM tasks) | divisor (`RequestService`) |
|---|---|---|---|---|
| 0, 9, 18, 27 | `n_ready[t-3..t-0]` | ready replicas | 4 | 20 |
| 1, 10, 19, 28 | `n_booting[t-3..t-0]` | replicas paying cold start | 4 | 20 |
| 2, 11, 20, 29 | `n_draining[t-3..t-0]` | replicas finishing work, still billed | 4 | 20 |
| 3, 12, 21, 30 | `target[t-3..t-0]` | last requested replica count | 4 | 20 |
| 4, 13, 22, 31 | `n_running[t-3..t-0]` | requests in service, cluster-wide | 32 | 80 |
| 5, 14, 23, 32 | `n_waiting[t-3..t-0]` | requests queued, cluster-wide | 32 | 80 |
| 6, 15, 24, 33 | `sat_mean[t-3..t-0]` | mean saturation of the binding per-replica resource, already in [0,1] | 1 | 1 |
| 7, 16, 25, 34 | `sat_max[t-3..t-0]` | max of the same | 1 | 1 |
| 8, 17, 26, 35 | `arrivals[t-3..t-0]` | arrivals observed over the control interval | 16 | 64 |
| 36 | `episode_progress` | control steps taken / `n_control_steps` | `n_control_steps` | `n_control_steps` |

"Saturation of the binding per-replica resource" is KV-cache occupancy for `llm_serving` and
concurrency-slot occupancy for `request_service`; the core's `ClusterState` exposes it under
each deployed system's own metric name and the aggregate picks whichever is present.

### The fairness guarantee, and the one element that is not a core feature

Elements 0–35 are keys of `kubegym.core.state.FEATURES`, each derived only from a
`ClusterState`. That is the core's guarantee that an RL policy can never observe more than a
hand-written controller reading the same scrape, and this layer inherits it by construction
rather than re-arguing it — `test_gym_observation.py` asserts the feature set is a subset of
`FEATURES`.

Element 36, `episode_progress`, is **not** a `FEATURES` key. It is fair anyway: the core's
controller interface is `decide(hist, t)`, so an analytic baseline is handed the wall time and
knows the horizon too. Without it the task would be non-Markov in a way that penalises the
learner specifically — the correct end-of-episode action depends on how much episode is left,
and the drain is charged at the end. It carries no workload information.

### Normalisation rules

A divisor may only be a constant knowable **before the episode starts**: a config field
(`n_replicas_max`), a documented design constant of the task, or a measured range boundary.
Never a statistic of the episode being run — no running mean, no per-episode max. Two reasons,
the second load-bearing: a quantity computed from the whole episode is information from the
future, and adaptive normalisation would break the fairness guarantee *silently*, because the
resulting element still looks like a scaled cluster metric.
`test_normalisation_uses_no_episode_statistics` re-derives each step's observation from only the
scrapes available at that step and asserts it matches what was emitted.

`concurrency_scale = n_replicas_max x 8` on the LLM tasks because batch 8 is the top of the
**measured** decode-throughput range (`decode_step_time_s` was fitted at batch 1/4/8), so the
element reads 1.0 at the edge of the calibrated region. That is a convenience, not a claim that
8 is a capacity limit.

The scales are round design constants. They change the conditioning of the learning problem,
not its physics, and a study reporting learning curves is reporting them under these scales.

### Clipping

Clipping is lossy at the top: under extreme overload `n_waiting / concurrency_scale` saturates
and the policy stops seeing the queue grow. Measured across every constant action in both modes
on three work seeds per task, the largest scaled element ever observed is:

| task | max scaled element | element | headroom to `clip = 10.0` |
|---|---|---|---|
| `LLMServing-S1` | 1.938 | `n_waiting[t-0]` | 5.2x |
| `LLMServing-S3` | 1.906 | `n_waiting[t-0]` | 5.2x |
| `LLMServing-Mixed` | 1.938 | `n_waiting[t-0]` | 5.2x |
| `RequestService-Poisson` | 5.825 | `n_waiting[t-0]` | 1.7x |

So clipping never bites on the shipped fixtures
(`test_no_clipping_saturation_on_the_shipped_fixtures`), but the request-service task has only
1.7x of headroom, and it is the queue-length element that is closest — a heavier arrival rate or
a longer horizon on that task would saturate it. Raise `concurrency_scale` or `clip` when
registering such a task, and say so in its card.

---

## 6. `info`

Enough to reconstruct and audit the episode. Present on every step:

**Identity** (sufficient to re-run): `task`, `task_id`, `action_mode`, `split`, `episode`,
`work_seed`, `config`.

**Claims gate**: `calibrated`, `uncalibrated_required_fields`.

**Actuation**: `t`, `step`, `n_control_steps`, `action`, `target_before`, `target_requested`,
`target`, `action_clamped`, `n_replicas_min`, `n_replicas_max`, `n_replicas_max_physical`,
`hypothetical_replicas_allowed`, `hypothetical_replica_count_used`.

**Cluster state as scraped**: `n_ready`, `n_booting`, `n_draining`, `n_replicas_effective`
(= ready + booting + draining, what the Kubernetes HPA calls `currentReplicas`),
`arrivals_since_last`, `replica_seconds_step`, `replica_seconds_total`.

**Objective**: `cost` (every term, every `raw_*` count, `weights`, `units`,
`sign_convention`), `cost_cumulative`, `slo` (`violations_step`, `reasons_step` with a
per-reason breakdown and `n_scored`, `violations_cumulative`, `targets`, `profile`).

At the final step, additionally:

* `post_horizon_drain` — `t_end`, `duration_s`, `replica_seconds`, `slo_violations`,
  `slo_reasons`, `unfinished_requests`, `truncated_drain`, `cap_s`, `charged_to_reward`.
* `episode_summary` — the core's own provenance-stamped conservation quantities.
* `episode_cost` — the episode cost row, passed through `ProvenancedConfig.stamp`, so a return
  cannot travel without its calibration label.

---

## 7. Where this layer makes the problem easier or harder than for analytic baselines

Stated explicitly because a benchmark that hides this is not auditable. No baselines ship in
this phase, so these are properties of the environment, not measured gaps.

1. **The churn term is shaping, not physics.** Boot and drain time are already billed as
   replica-seconds; `churn` charges a scaling move a second time. An RL agent optimises the
   benchmark objective including it. A hand-written controller tuned for cost-plus-SLO only —
   which is how autoscaling heuristics are normally tuned — is not optimising the same
   objective and will look worse for a reason that is not about control quality. Whoever tunes
   the baselines must tune them against this exact reward, weights included.
2. **`episode_progress` is in the observation.** It is available to `decide(hist, t)`
   controllers too, so it is not privileged — but a typical HPA-style baseline ignores time,
   while a learner will exploit it, particularly around the charged drain at the horizon. That
   is a real asymmetry in what the two classes of controller *use*, not in what they *may see*.
3. **The observation is a fixed 4-lag window; a hand-written controller gets all of `sim.hist`.**
   `decide(hist, t)` receives the entire history, so a controller with a longer memory than 4
   intervals is *advantaged* relative to a policy on this observation. This is the one axis
   where the RL layer is handicapped, and it is a choice of the task spec, changeable by
   registering a task with a larger `obs_history`.
4. **Episodes are 20 control steps.** The measured 31.9 s cold start spans 2.13 intervals, so
   roughly a tenth of an episode is spent paying for any single scale-up, and end effects are a
   large fraction of the return. This penalises learning more than it penalises a stateless
   heuristic. The 300 s traces are the core's *test fixtures*; longer episodes are the workload
   corpus track's job.
5. **The SLO term is nearly binary on the LLM tasks and has an unreachable floor on the
   request-service task.** Measured: 254 violations over three seeds at `k=1` and exactly 0 at
   `k=2/3/4` on `LLMServing-S1`; 47–49 % of requests violate on `RequestService-Poisson` even at
   the ceiling `k=20`, because the placeholder service-time lognormal's median (1.0 s) equals
   the placeholder `latency_p95_s`. In both cases the shape of the objective is set by
   placeholder constants and would move under calibration. Details in `unverified_gym.md` §3.
6. **The actuation lag is invisible on `RequestService-Poisson`.** Its placeholder 2 s cold
   start completes inside one 15 s control interval, so `n_booting` is zero at every scrape and
   four of the 37 observation elements are constant zero. That task is a no-lag control problem
   — structurally easier than the LLM tasks — as an artefact of a placeholder constant.

---

## 8. Registering a new task

`register_task` is the whole contract. A downstream paper or the workload-corpus track adds
tasks without this package importing theirs, and needs nothing from `kubegym.gym` beyond
`TaskSpec`, `ObsNorm`, `CostWeights` and this function.

```python
from kubegym.gym import TaskSpec, ObsNorm, CostWeights, register_task

def my_workload():                      # () -> WorkloadSource, fresh per env
    from kubegym.models.request_service import RequestServiceSource
    return RequestServiceSource(records, service_time_distribution=dist,
                                name="azure_2019_replay", horizon_s=1800.0)

def my_service_model(cfg):              # cfg -> ServiceModel
    from kubegym.models.request_service import RequestServiceModel
    return RequestServiceModel(cfg)

ids = register_task(TaskSpec(
    name="Serverless-Azure2019",                       # id becomes KubeGym/<name>-<Mode>-v0
    service_model_name="request_service",
    config_file="request_service_default.json",         # or an absolute path
    workload_name="azure_functions_2019",
    workload_description="... provenance, checksum, extraction procedure ...",
    workload_factory=my_workload,
    service_model_factory=my_service_model,
    n_episodes=32, horizon_s=1800.0, k0=1,
    obs_features=("n_ready", "n_booting", "n_draining", "target",
                  "n_running", "n_waiting", "sat_mean", "sat_max", "arrivals"),
    obs_history=4,
    obs_norm=ObsNorm(replica_scale=20.0, concurrency_scale=80.0,
                     per_replica_concurrency=4.0, arrivals_scale=64.0),
    weights=CostWeights(slo=0.05, resource=1.0 / (15.0 * 20), churn=0.05),
    slo_profile="request_service",
    notes=("... caveats, copied verbatim into the env card ...",),
))
```

If the new task differs only in workload — same service model, observation, weights and SLO
profile as a builtin — `register_workload_task` inherits those conventions from the reference
spec for that service model and records in the card that it did so, which keeps an inherited
choice visible rather than implicit:

```python
kubegym.gym.register_workload_task(
    name="Serverless-Azure2019", workload_factory=my_workload,
    n_episodes=32, horizon_s=1800.0, service_model="request_service",
    workload_description="...", notes=("...",))
```

Rules the registry enforces:

* **Never mutate a registered `-v0` in place.** Re-registering a name raises unless
  `override=True`; bump the version instead. A reader comparing their `-v0` against a published
  `-v0` must be comparing the same task.
* `horizon_s` must be an exact multiple of `control_interval_s`.
* `delta_set` must contain 0.
* Split work seeds must be disjoint, or a test episode is a training episode under another
  name. Enforced in `TaskSpec.__post_init__` and asserted per task in `test_gym_registry.py`.
* A task carries **no physical constants of its own** — they all come from the config, so the
  claims gate covers them. Observation scales and cost weights are the exception, which is why
  both are recorded in the env card and in `unverified_gym.md`.

The id scheme is `KubeGym/<Task>-<Delta|Abs>-v0`. The split is a constructor keyword, not part
of the id: it selects seeds rather than redefining the task.

### Regenerating the cards

```
python -m kubegym.gym.cards > env_cards.md
```

Cards are rendered from live envs, so every number in `env_cards.md` was read off the object
`gymnasium.make(<id>)` returns. A hand-typed card drifts the first time a weight changes.

### The smoke run

```
python -m kubegym.gym.smoke --task KubeGym/LLMServing-S1-Delta-v0 --json-out smoke.json
```

Runs a random and a fixed-target policy, prints the per-episode cost decomposition, and times
1000 env steps against the same episodes driven straight on `Simulator` — which isolates what
the wrapper adds. Numbers in `gym_tests_report.md`.
