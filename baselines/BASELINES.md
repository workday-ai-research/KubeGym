# KubeGym reference baselines — methods, budget, protocol, and honest interpretation

**Every number in this document is an in-simulator number produced under a
configuration whose `calibrated` flag is `False`.** All 8,496 result rows carry
`calibrated=false` and the field list
`replica_cold_start_s;rs_max_concurrency;rs_service_time_distribution;slo`,
stamped by `cfg.stamp()`. Those four constants are claims-relevant and unmeasured,
so nothing below is an empirical claim about how any autoscaler performs on real
infrastructure. What is claimed is narrower and checkable: *under this
simulator, with this protocol, these methods separate in this way*.

---

## 1. Methods

### 1.1 Analytic controllers, ported onto the Gym interface

All six read the **same 37-element observation vector** an RL policy reads,
decoded by `baselines/obsmap.py`, which is the documented inverse of the
environment's observation construction. `ObsView.assert_lossless` re-checks the
decode against the live `ClusterState` on every step of every verification run,
so "the controller sees what the policy sees" is enforced, not asserted.

| controller | provenance | notes |
|---|---|---|
| `static` | source `StaticK` | run at **every feasible k**, 1–20, as the trivial reference front |
| `hpa` | source `HPA`, semantics from the Kubernetes primary source | ratio rule, tolerance band (0.1), scale-down stabilization window, default policies (up: `max(cur+4, 2·cur)` per 15 s, `selectPolicy: Max`; down: 100 %/15 s), 15 s sync period |
| `queue` | source `QueueConcurrency` | per-replica in-flight concurrency target, optional hysteresis and cooldown |
| `leading` † | source `LeadingIndicator`, **adapted** | source drives on differenced `vllm:generation_tokens_total`; `request_service` has no token stream and no token counter appears in the observation, so the signal is the arrival rate plus its derivative. Structural idea only; **not** a measurement of the cited method |
| `forecast` | source `ForecastThreshold` | EWMA / Holt / ridge-AR forecast to `t + cold_start + H`, then threshold. The `signal="tokens"` variant has no counterpart and was dropped. Ridge coefficients fitted on **dev only**; `ridge_fallback` records any unfitted fallback |
| `oracle` * | **new** for `request_service` | reads true per-request `demand`, `t_arrival` and `progress`, including for requests that have not arrived. `uses_privileged_info=True` on every row; excluded from the deployable comparison and from the front |

\* privileged  † adaptation, not a faithful port

The Kubernetes HPA attribution string is carried over verbatim in
`gym_controllers.HPA_SEMANTICS_SOURCE` and cross-checked against the
primary-source extract shipped with the project.

**One grid was deliberately re-ranged.** The source's `conc`/`running` HPA target
grid spans 2–30, sized for an LLM replica with a large `max_num_seqs`. Under
`request_service_default.json` a replica holds `rs_max_concurrency = 4`, so a
per-replica concurrency target of 30 is unreachable and the ratio rule would
never fire — HPA would silently become `static-k0`. That is the same class of
degenerate configuration the prior project's tuning protocol documents, so the
grid is re-ranged per service model and the re-ranging is recorded in the class
docstring.

### 1.2 Learned controllers

`stable-baselines3` 2.9.0 PPO and DQN, `MlpPolicy`, `net_arch=[64, 64]`, CPU,
`DummyVecEnv` with 4 envs (a control step is 0.7–2.2 ms, so subprocess IPC would
cost more than the step it parallelises). No reward or observation normalisation
wrapper: the observation is already scaled by a-priori constants and the per-step
reward is O(1), and a normaliser would put a moving statistic derived from
episode aggregates between the policy and the benchmark's objective — a signal
the analytic controllers are not allowed. That is a fairness choice and is
recorded as one, not an oversight.

Fixed: `gamma=0.99`; PPO `n_steps=256`, `batch_size=256`, `n_epochs=10`,
`gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0`, `vf_coef=0.5`,
`max_grad_norm=0.5`; DQN `buffer_size=100k`, `learning_starts=2k`,
`batch_size=64`, `train_freq=4`, `target_update_interval=1000`,
`exploration_fraction=0.2`, `exploration_final_eps=0.05`.

Searched, identically shaped for both: {two learning rates} × {`absolute`} plus
{`delta` at the lower rate} — three configurations. The **action mode is inside
the grid on purpose**: `absolute` can request any target in one step, matching
what the analytic controllers can do, while `delta` is rate-limited to ±2
replicas per interval but has a 5-way rather than 20-way action space. Which is
easier to learn is an empirical question, answered on dev.

Selected on dev:

| algo | family | action mode | learning rate |
|---|---|---|---|
| ppo | constant | delta | 0.0003 |
| ppo | diurnal | delta | 0.0003 |
| ppo | variable | delta | 0.0003 |
| ppo | burst | delta | 0.0003 |
| ppo | azure_replay | delta | 0.0003 |
| dqn | constant | absolute | 0.0005 |
| dqn | diurnal | absolute | 0.0005 |
| dqn | variable | absolute | 0.0005 |
| dqn | burst | delta | 0.0001 |
| dqn | azure_replay | delta | 0.0001 |

---

## 2. The enforced tuning budget

**The budget is one integer for every method and every family**: 1,200,000
environment steps taken on the **train and dev** splits, charged through
`runner.BudgetLedger`, which raises `BudgetExceeded` past the cap. Test-split
steps are not charged because they are not search — they are the single
measurement at the end. `runner.assert_not_test` raises if any tuning or training
code path opens the test split; it warns about nothing.

| method | cap (env steps) | spent (min–max over families) | % of cap | configs evaluated | wall (h) |
|---|---|---|---|---|---|
| dqn | 1,200,000.0 | 1,129,200.0–1,133,040.0 | 94.1–94.4 | 3.0 of 3.0 | 2.33 |
| forecast | 1,200,000.0 | 290,400.0–348,480.0 | 24.2–29.0 | 120.0 of 1728.0 | 0.56 |
| hpa | 1,200,000.0 | 115,200.0–138,240.0 | 9.6–11.5 | 48.0 of 48.0 | 0.19 |
| leading | 1,200,000.0 | 288,000.0–345,600.0 | 24.0–28.8 | 120.0 of 144.0 | 0.52 |
| oracle | 1,200,000.0 | 57,600.0–69,120.0 | 4.8–5.8 | 24.0 of 24.0 | 0.14 |
| ppo | 1,200,000.0 | 1,129,200.0–1,133,040.0 | 94.1–94.4 | 3.0 of 3.0 | 2.58 |
| queue | 1,200,000.0 | 115,200.0–138,240.0 | 9.6–11.5 | 48.0 of 48.0 | 0.19 |
| static | 1,200,000.0 | 48,000.0–57,600.0 | 4.0–4.8 | 20.0 of 20.0 | 0.10 |

**Read that table before reading any result.** The two learners spent 94 % of the
shared cap; the strongest analytic controllers spent 10–29 % and `static` spent
4 %. The parity being claimed is *equal cap*, and the realised spend favours the
learners by a factor of 3 to 24. If the analytic controllers beat the learners
here, it is not because they were given more search.

Selection is on dev only, by the **environment's own mean episode cost** —
identical for every method, including the `churn` shaping term, because a
controller tuned against a different objective than the benchmark scores looks
worse for a reason that is not about control quality. Ties break deterministically
(lower replica-seconds, then fewer scale events, then lexicographic parameters).
Grids larger than 120 configurations are sampled **blind at a fixed seed**
(`leading` 120 of 144, `forecast` 120 of 1728), never hand-shortlisted. Every
trial is logged in `tuned_<family>.json`, and each family's selection is
additionally recorded at SLO-weight multipliers 0.25 and 4.0 at no extra cost.

Dev tuning episodes: 12 per family (10 for `constant`, whose dev split has 10
traces), **stratified by regime** rather than truncated — the `azure_replay` dev
split spans five regimes whose difficulty differs by two orders of magnitude and
its first 12 trace ids are not a sample of that.

Dev-selected parameters:

| family | static | hpa | queue | leading | forecast | oracle |
|---|---|---|---|---|---|---|
| constant | `{"k":3}` | `{"metric":"running","target":3.5,"tolerance":0.1,"down_stabilization_s":300.0}` | `{"cooldown_s":0.0,"hyst":0.15,"target_concurrency":2.0}` | `{"backlog_target":2.0,"kappa":5.0,"per_replica_rps":4.0,"window_s":45.0}` | `{"H_s":0.0,"alpha":0.4,"backlog_target":4.0,"beta":0.1,"forecaster":"holt","per_replica_rps":4.0,"safety":1.0}` | `{"H_s":30.0,"mode":"rate","safety":0.5}` |
| diurnal | `{"k":3}` | `{"metric":"running","target":2.5,"tolerance":0.1,"down_stabilization_s":60.0}` | `{"cooldown_s":0.0,"hyst":0.15,"target_concurrency":2.0}` | `{"backlog_target":4.0,"kappa":5.0,"per_replica_rps":4.0,"window_s":45.0}` | `{"H_s":30.0,"alpha":0.7,"backlog_target":4.0,"beta":0.1,"forecaster":"ewma","per_replica_rps":4.0,"safety":1.2}` | `{"H_s":30.0,"mode":"rate","safety":0.5}` |
| variable | `{"k":4}` | `{"metric":"running","target":2.5,"tolerance":0.1,"down_stabilization_s":60.0}` | `{"cooldown_s":30.0,"hyst":0.15,"target_concurrency":2.0}` | `{"backlog_target":2.0,"kappa":5.0,"per_replica_rps":4.0,"window_s":30.0}` | `{"H_s":0.0,"alpha":0.4,"backlog_target":2.0,"beta":0.3,"forecaster":"ewma","per_replica_rps":4.0,"safety":1.0}` | `{"H_s":30.0,"mode":"rate","safety":0.5}` |
| burst | `{"k":4}` | `{"metric":"running","target":2.5,"tolerance":0.1,"down_stabilization_s":60.0}` | `{"cooldown_s":0.0,"hyst":0.0,"target_concurrency":2.0}` | `{"backlog_target":2.0,"kappa":5.0,"per_replica_rps":4.0,"window_s":45.0}` | `{"H_s":60.0,"alpha":0.2,"backlog_target":2.0,"beta":0.3,"forecaster":"ridge","per_replica_rps":4.0,"safety":1.0}` | `{"H_s":30.0,"mode":"rate","safety":0.5}` |
| azure_replay | `{"k":1}` | `{"metric":"running","target":4.0,"tolerance":0.1,"down_stabilization_s":300.0}` | `{"cooldown_s":0.0,"hyst":0.0,"target_concurrency":8.0}` | `{"backlog_target":4.0,"kappa":0.0,"per_replica_rps":8.0,"window_s":45.0}` | `{"H_s":0.0,"alpha":0.2,"backlog_target":4.0,"beta":0.1,"forecaster":"ewma","per_replica_rps":8.0,"safety":1.0}` | `{"H_s":30.0,"mode":"rate","safety":1.0}` |

---

## 3. Port verification

The core's parity harness compares two simulators step for step. A controller
port cannot be checked that way, because the moment two controllers disagree
their trajectories diverge and later comparisons are meaningless. So decisions
are compared **along one trajectory**: the ported controller drives the episode
while the source controller is asked, at the same step, for its decision given
the raw `ClusterState`. Agreement at every step implies identical trajectories by
induction.

| service model | controller | kind | runs | shadowed steps | disagreements |
|---|---|---|---|---|---|
| llm_serving | hpa | port | 8 | 160 | 0 |
| request_service | forecast | port | 48 | 11,520 | 0 |
| request_service | hpa | port | 72 | 17,280 | 0 |
| request_service | leading | adaptation | 36 | 8,640 | 0 |
| request_service | queue | port | 72 | 17,280 | 0 |
| request_service | static | port | 24 | 5,760 | 0 |

**0 disagreements over 60,640 shadowed control steps and 260 runs.** Source file
sha256 `4185faabc833ba2759dd9f621c91e85638cb9314a2e4e5830295df8f814fb612`.

`hpa` with `metric="kv"` is checked on the builtin `LLMServing-S1` task, because
the source reads the raw metric name `vllm:kv_cache_usage_perc`, which does not
exist under `request_service` (the metric guard raises rather than returning
zero). `leading` is compared against a reference implementation of the *same
adapted algorithm* on the unrestricted scrape history, which isolates the cost of
the 4-lag observation window; it is not a check against the published method.

### 3.1 The first parity run did not pass, and the reason is worth recording

It reported 123 disagreements for `hpa`, 106 for `queue` and 7 for `leading`. The
cause was not clipping (that hypothesis was tested and rejected): the
`episode_progress` observation element is `float32`, so `progress × horizon_s`
recovers the episode time only to ~2 × 10⁻⁵ s. Several controllers compare an
elapsed time against a window boundary (`t − t_change < cooldown_s`,
`t − t' ≤ 300`), and at exactly the boundary a 2 × 10⁻⁵ s error flips the
comparison. Recovering the step index first and multiplying by the control
interval makes the time exact, after which every disagreement vanished. The
arrival-rate estimator was also rewritten to reproduce the source's window
convention exactly, including its treatment of which scrapes fall inside a
window, and to use the recoverable count of non-padded lags so the port matches
the source at the start of an episode instead of reading padded zeros as real
intervals. The swept `window_s` grid was narrowed to 15/30/45 s, the range a
4-lag observation can represent.

### 3.2 The privileged oracle: property checks, not a port check

The source `Oracle` is LLM-specific, so there is nothing to shadow. Two
properties are asserted instead, both along a fixed replayed action sequence:

| family | P1a demand read (mode `rate`) | P1a same, configured mode | P1b future arrivals read | P2 anticipatory scale-ups |
|---|---|---|---|---|
| burst | 321 | 0 | 702 | 28 |
| azure_replay | 11 | 0 | 685 | 30 |
| variable | 380 | 0 | 720 | 34 |

P1a and P1b both pass, so `uses_privileged_info` is an accurate label. The middle
column is a **finding, not a failure**: doubling every request's true service
demand moves 11–380 decisions in mode `rate` and **zero** in the configured
`rate+residency` mode. On this corpus the residency bound
`ceil(n_concurrent / rs_max_concurrency)` dominates the work-rate bound, so the
oracle's privilege is mostly *knowing future arrival times*, not knowing their
durations.

---

## 4. Evaluation protocol

Held-out **test** split, opened once per method. Every method on every test
episode of every family: 236 episodes (constant 10, diurnal 24, variable 40,
burst 48, azure_replay 114) × 28 method-configurations (6 analytic, 20 `static-k`
front points, 2 learners × 5 seeds) = **8,496 rows** in
`reference_results.csv`.

Comparisons are **paired**: for a given (family, episode) every method faces a
byte-identical trace, because the corpus samples per-request work at build time.
Dispersion is reported across **episodes** and, for the learners, across
**training seeds**. There is deliberately **no work-seed dispersion** on the
corpus tasks: `seed` does not redraw the offered load there, so a work-seed
replicate is a byte-identical rerun. The other kind of variance is measured
separately (§7) and never pooled with these numbers.

---

## 5. Results

Lower cost is better. `*` privileged, `†` adaptation. `sd(ep)` is across test
episodes; `sd(seed)` is across the 5 training seeds and exists only for the
learners.

| family | method | mean cost | median cost | sd(ep) | sd(seed) | SLO attain. | replica-s | scale events | p95 latency (s) | worst-episode cost share |
|---|---|---|---|---|---|---|---|---|---|---|
| azure_replay | oracle* | 49.97 | 12.33 | 119.33 | -- | 0.8223 | 5338 | 15.4 | 7.41 | 0.165 |
| azure_replay | leading† | 64.16 | 12.35 | 159.55 | -- | 0.8040 | 5633 | 20.0 | 8.48 | 0.153 |
| azure_replay | forecast | 64.19 | 12.35 | 159.90 | -- | 0.8040 | 5638 | 20.1 | 8.46 | 0.154 |
| azure_replay | queue | 76.40 | 12.33 | 201.07 | -- | 0.7829 | 5249 | 20.5 | 9.32 | 0.154 |
| azure_replay | ppo | 105.59 | 12.33 | 295.75 | 19.79 | 0.7315 | 5877 | 4.1 | 395.67 | 0.160 |
| azure_replay | hpa (constant target k=1) | 112.01 | 12.33 | 313.70 | -- | 0.7150 | 4368 | 0.0 | 763.01 | 0.150 |
| azure_replay | static (constant target k=1) | 112.01 | 12.33 | 313.70 | -- | 0.7150 | 4368 | 0.0 | 763.01 | 0.150 |
| azure_replay | dqn | 112.79 | 12.33 | 316.23 | 1.74 | 0.7151 | 4607 | 0.0 | 761.54 | 0.149 |
| burst | oracle* | 45.98 | 40.74 | 24.58 | -- | 0.9767 | 8792 | 44.7 | 0.46 | 0.063 |
| burst | hpa | 53.89 | 41.20 | 36.12 | -- | 0.9633 | 8478 | 29.1 | 1.42 | 0.076 |
| burst | ppo | 58.95 | 47.76 | 41.79 | 3.50 | 0.9658 | 9460 | 56.5 | 1.41 | 0.081 |
| burst | forecast | 59.21 | 46.18 | 40.57 | -- | 0.9656 | 9344 | 75.6 | 1.22 | 0.077 |
| burst | leading† | 60.56 | 47.50 | 43.93 | -- | 0.9629 | 9196 | 56.1 | 1.35 | 0.079 |
| burst | queue | 63.57 | 52.67 | 43.07 | -- | 0.9585 | 9074 | 57.6 | 1.46 | 0.072 |
| burst | dqn | 83.98 | 67.60 | 68.01 | 8.01 | 0.9194 | 7768 | 58.5 | 10.89 | 0.077 |
| burst | static (constant target k=4) | 146.63 | 60.51 | 289.31 | -- | 0.9243 | 14980 | 1.0 | 25.09 | 0.226 |
| constant | ppo | 42.95 | 43.34 | 18.58 | 1.88 | 0.9779 | 8546 | 9.3 | 0.44 | 0.164 |
| constant | hpa | 44.54 | 44.18 | 19.24 | -- | 0.9711 | 7676 | 7.2 | 0.53 | 0.155 |
| constant | leading† | 44.72 | 45.57 | 21.25 | -- | 0.9790 | 8539 | 54.4 | 0.43 | 0.169 |
| constant | forecast | 45.29 | 43.13 | 20.94 | -- | 0.9710 | 7530 | 13.0 | 0.54 | 0.154 |
| constant | oracle* | 47.14 | 47.87 | 23.25 | -- | 0.9768 | 8748 | 46.8 | 0.46 | 0.187 |
| constant | dqn | 51.95 | 45.67 | 27.32 | 8.49 | 0.9724 | 9440 | 20.3 | 0.61 | 0.180 |
| constant | queue | 52.30 | 52.37 | 28.44 | -- | 0.9731 | 8325 | 75.2 | 0.50 | 0.181 |
| constant | static (constant target k=3) | 53.49 | 49.65 | 15.28 | -- | 0.9773 | 11153 | 1.0 | 0.45 | 0.168 |
| diurnal | forecast | 43.79 | 44.14 | 14.06 | -- | 0.9782 | 8505 | 15.0 | 0.44 | 0.073 |
| diurnal | hpa | 44.88 | 44.54 | 14.95 | -- | 0.9790 | 8749 | 26.8 | 0.43 | 0.070 |
| diurnal | ppo | 48.18 | 51.30 | 16.44 | 6.31 | 0.9757 | 8561 | 64.3 | 0.47 | 0.073 |
| diurnal | oracle* | 49.50 | 48.19 | 18.62 | -- | 0.9762 | 9039 | 52.7 | 0.47 | 0.077 |
| diurnal | leading† | 49.96 | 47.60 | 13.82 | -- | 0.9677 | 7792 | 34.6 | 0.58 | 0.067 |
| diurnal | dqn | 52.51 | 53.50 | 14.74 | 2.17 | 0.9750 | 10276 | 20.2 | 0.51 | 0.069 |
| diurnal | static (constant target k=3) | 54.82 | 51.49 | 18.52 | -- | 0.9767 | 11184 | 1.0 | 0.95 | 0.103 |
| diurnal | queue | 55.21 | 53.60 | 22.27 | -- | 0.9729 | 8981 | 76.2 | 0.51 | 0.078 |
| variable | hpa | 41.82 | 42.23 | 16.08 | -- | 0.9748 | 7714 | 24.9 | 0.51 | 0.048 |
| variable | leading† | 42.29 | 42.79 | 15.86 | -- | 0.9789 | 8112 | 54.5 | 0.43 | 0.051 |
| variable | forecast | 42.55 | 41.61 | 16.98 | -- | 0.9781 | 8052 | 48.5 | 0.44 | 0.053 |
| variable | oracle* | 42.98 | 40.74 | 15.98 | -- | 0.9764 | 8020 | 43.6 | 0.46 | 0.047 |
| variable | dqn | 45.43 | 43.27 | 16.92 | 1.86 | 0.9746 | 8532 | 28.4 | 0.50 | 0.049 |
| variable | ppo | 45.46 | 44.99 | 19.53 | 3.84 | 0.9719 | 7576 | 65.5 | 0.55 | 0.056 |
| variable | queue | 50.00 | 47.63 | 21.71 | -- | 0.9690 | 8014 | 53.7 | 0.60 | 0.055 |
| variable | static (constant target k=4) | 72.05 | 61.52 | 56.77 | -- | 0.9705 | 14780 | 1.0 | 1.96 | 0.142 |

### 5.1 The headline, stated as narrowly as the data supports

A calibrated analytic controller is the best deployable method on **four of five
families** (`hpa` on burst and variable, `forecast` on diurnal, `leading`/`forecast`
on azure_replay). On `constant`, PPO has the lowest mean cost (42.95 against
44.54 for `hpa`), but the paired difference is +1.59 with a 95 % bootstrap CI of
[−0.27, +3.61] and PPO's across-seed SD is 1.88 — **inside noise, so the two are
not ranked**.

Neither learner beats the best analytic controller on any family by a margin
that survives its own seed dispersion. DQN is never the best method on any
family. This is consistent with the prior benchmark study's conclusion
(arXiv:2605.26418) that baseline calibration rather than
algorithm choice is the binding constraint — but it is a *replication under one
simulator with uncalibrated constants*, not an independent confirmation.

### 5.2 Where methods are not distinguishable

47 of 140 ordered method pairs are flagged
`within_noise` in `pairwise_comparisons.csv` — either the 95 % paired bootstrap CI
of the mean difference covers zero, or the point difference is smaller than the
relevant across-seed SD. **Flagged pairs are not ranked anywhere in this
document.** Restricted to deployable methods:

| family | pair | mean diff | 95 % CI | why flagged |
|---|---|---|---|---|
| azure_replay | static vs hpa | +0.00 | [+0.00, +0.00] | CI covers 0 |
| azure_replay | static vs ppo | +6.42 | [+2.01, +12.27] | below seed SD |
| azure_replay | static vs dqn | -0.78 | [-1.78, -0.13] | below seed SD |
| azure_replay | hpa vs ppo | +6.42 | [+2.01, +12.27] | below seed SD |
| azure_replay | hpa vs dqn | -0.78 | [-1.78, -0.13] | below seed SD |
| azure_replay | leading vs forecast | -0.03 | [-0.31, +0.28] | CI covers 0 |
| azure_replay | ppo vs dqn | -7.19 | [-13.18, -2.53] | below seed SD |
| burst | leading vs forecast | +1.36 | [-0.12, +3.51] | CI covers 0 |
| burst | leading vs ppo | +1.61 | [-0.32, +3.72] | CI covers 0 & below seed SD |
| burst | forecast vs ppo | +0.26 | [-1.98, +2.56] | CI covers 0 & below seed SD |
| constant | static vs queue | +1.18 | [-9.08, +11.19] | CI covers 0 |
| constant | static vs dqn | +1.54 | [-7.83, +10.30] | CI covers 0 & below seed SD |
| constant | hpa vs leading | -0.19 | [-2.79, +2.20] | CI covers 0 |
| constant | hpa vs forecast | -0.76 | [-2.97, +1.07] | CI covers 0 |
| constant | hpa vs ppo | +1.59 | [-0.27, +3.61] | CI covers 0 & below seed SD |
| constant | hpa vs dqn | -7.41 | [-13.27, -2.25] | below seed SD |
| constant | queue vs dqn | +0.36 | [-1.98, +2.93] | CI covers 0 & below seed SD |
| constant | leading vs forecast | -0.57 | [-4.68, +3.26] | CI covers 0 |
| constant | leading vs ppo | +1.78 | [+0.11, +3.54] | below seed SD |
| constant | leading vs dqn | -7.22 | [-11.68, -3.03] | below seed SD |
| constant | forecast vs ppo | +2.35 | [-1.13, +6.41] | CI covers 0 |
| constant | forecast vs dqn | -6.65 | [-12.83, -1.75] | below seed SD |
| diurnal | static vs queue | -0.39 | [-6.43, +5.61] | CI covers 0 |
| diurnal | static vs dqn | +2.31 | [-1.72, +7.54] | CI covers 0 |
| diurnal | hpa vs ppo | -3.30 | [-4.37, -2.32] | below seed SD |
| diurnal | queue vs dqn | +2.70 | [-0.83, +6.47] | CI covers 0 |
| diurnal | leading vs ppo | +1.78 | [-0.69, +4.30] | CI covers 0 & below seed SD |
| diurnal | forecast vs ppo | -4.39 | [-5.63, -3.11] | below seed SD |
| diurnal | ppo vs dqn | -4.33 | [-5.31, -3.37] | below seed SD |
| variable | hpa vs leading | -0.47 | [-1.66, +0.93] | CI covers 0 |
| variable | hpa vs forecast | -0.74 | [-1.97, +0.50] | CI covers 0 |
| variable | hpa vs ppo | -3.64 | [-5.34, -2.14] | below seed SD |
| variable | leading vs forecast | -0.26 | [-1.40, +0.56] | CI covers 0 |
| variable | leading vs ppo | -3.17 | [-5.00, -1.72] | below seed SD |
| variable | forecast vs ppo | -2.90 | [-3.94, -1.89] | below seed SD |
| variable | ppo vs dqn | +0.02 | [-1.37, +1.61] | CI covers 0 & below seed SD |

### 5.3 Two results dominated by a handful of episodes, named

*Azure replay.* Every method's **median** episode cost is 12.325 — identical to
three decimal places across `static`, `hpa`, `queue`, `ppo` and `dqn`. All the
separation lives in the tail: 80.7 % of the 114 test episodes have every method
agreeing to within 10⁻⁹, and the ten heaviest episodes carry 75.2 % of the
family's total cost. Per regime:

| regime | n episodes | static | hpa | dqn | ppo | forecast |
|---|---|---|---|---|---|---|
| bursty | 24 | 245.72 | 245.72 | 248.90 | 233.66 | 135.42 |
| diurnal | 24 | 252.82 | 252.82 | 253.34 | 234.40 | 135.97 |
| intermittent | 24 | 12.23 | 12.23 | 12.23 | 12.23 | 12.23 |
| sparse | 18 | 12.08 | 12.08 | 12.08 | 12.08 | 12.08 |
| steady | 24 | 12.22 | 12.22 | 12.22 | 12.22 | 12.22 |

On `sparse`, `intermittent` and `steady` (66 of 114 episodes) **every method is
identical**, because holding one replica is optimal and nothing any controller
does changes the outcome. The entire azure_replay result is the 48 `bursty` and
`diurnal` episodes, where `forecast` costs about 45 % less than `static`, `hpa`
and `dqn`. A corpus-level azure_replay average is therefore not an
Azure-population estimate, exactly as `WORKLOADS.md` §3 warns; the per-regime
table is the reportable unit.

*Burst under `static`.* Mean cost 146.63 against a median of 60.51, with the
single worst episode contributing 22.6 % of family cost. The `static` mean on
burst is a statement about a few overload episodes, not about typical behaviour.

### 5.4 HPA degenerates to Static-1 on Azure replay

On azure_replay the dev-selected HPA configuration
(`metric=running, target=4.0, tolerance=0.1, down_stabilization_s=300`) produces a
**constant replica target of 1** on every test episode, and its test metrics are
bit-identical to `static-k1` (mean cost 112.011 for both). This is flagged
mechanically by `is_constant_target` in `results_summary.csv` rather than left for
a reader to infer from two equal numbers. It is not a broken grid: per-replica
running concurrency on those traces rarely reaches 4, so the ratio rule stays
inside the tolerance band, and holding one replica is what the stratified dev
sample genuinely preferred. It does mean that on azure_replay "HPA" and "the
trivial baseline" are the same row, and any statement about HPA there is a
statement about static provisioning.

### 5.5 The SLO metric is largely determined by an uncalibrated constant

`slo_attainment_feasible_floor` counts requests whose **service time alone**
exceeds the p95 latency target; those violate under any policy. The floor is
0.9799–0.9808 on the four synthetic families but **0.8279 on azure_replay**. The
best deployable controller there attains 0.8040 and the privileged oracle 0.8223 —
i.e. within 0.6 percentage points of a ceiling set by the service-time
distribution, which is one of the four placeholder constants.

Decomposing the shortfall from perfect attainment on azure_replay: total 0.1960,
of which 0.1721 is imposed by the floor and 0.0239 is controllable. **87.8 % of
the shortfall is therefore set by `rs_service_time_distribution`**, and the best
deployable controller reaches 97.1 % of the attainable ceiling. On azure_replay
the SLO axis measures the placeholder distribution far more than it measures
control.

### 5.6 The oracle is a reference, not an upper bound

The privileged oracle has the lowest cost on azure_replay (49.97) and burst
(45.98), but is **beaten by a deployable controller** on constant (47.14 vs 44.54
for `hpa`), diurnal (49.50 vs 43.79 for `forecast`) and variable (42.98 vs 41.82
for `hpa`). That is expected and is why it is labelled a reference: it provisions
with an analytic capacity rule given exact knowledge of the offered work, and
that rule is not the cost-optimal policy under this reward. The gap it measures is
the value of *knowing the realisation* holding the provisioning rule fixed — not
the distance to optimal control.

### 5.7 Did the learners learn?

Yes on four families, no on one, and worse on one algorithm/family pair.
Dev-evaluation cost from the first to the last curve point (`learning_curves.csv`,
means over 5 seeds):

* PPO improves 25.3–30.3 % on constant, diurnal, variable and burst.
* DQN improves 14.6 % (constant), 19.5 % (variable), 33.4 % (burst), and
  **degrades 5.4 % on diurnal** (36.40 → 38.35).
* On **azure_replay both learners show 0.00 % change with 0.00 across-seed SD.**
  Both selected `delta` mode, whose dev objective is 14.158 — *bit-identical to
  `static-k1`'s* — for all five seeds. Their absolute-mode search configurations
  were far worse (PPO 50.24 and 81.78; DQN 27.60 and 23.98), so dev selection
  preferred the configuration that reproduces the trivial baseline. On the five
  curve episodes every method scores exactly 12.16, so the curve panel is flat by
  construction and is annotated as such in the figure. On test PPO does act
  occasionally (4.13 scale events per episode, and it beats `static` on the
  heaviest episodes: 1479.8 vs 1699.4 on episode 20), while DQN is numerically
  identical to `static-k1` on most heavy episodes. Reported as a failure to learn
  on this family, with the evidence, rather than tuned until it looked better.

### 5.8 Two mechanical caveats the rows carry

*Action clamping.* The learners hit an action-space bound on a mean of 123–125 of
240 steps per episode; the analytic controllers never do (0.0). This is
`delta`-mode rate limiting, and it means a large fraction of learned actions are
requests the environment cannot honour.

*Observation saturation.* 214 rows have at least one control step at the
observation clip ceiling (mean 39–191 saturated steps on those rows, worst on
azure_replay). At those steps the decoded queue length is a lower bound for the
ported controllers **and** for the policies — both are blind in the same way, which
is the point of the shared observation, but a saturated step is not a step where
either method saw the queue.

*Truncated drains.* 29 of 8,496 rows hit the post-horizon drain cap, all on
azure_replay (`ppo` 20, `dqn` 7, `queue` 2). No row finished with unfinished
requests outside the drain accounting.

---

## 6. An environment defect found while running this

`kubegym.models.request_service.RequestServiceEngine` completes a request when
`progress >= demand - 1e-12` — an **absolute** tolerance on an accumulator that
reaches O(10²) — and schedules the next event at
`t + min(demand - progress)/rate`. Measured instance, azure_replay test episode 4
(`azure-009f72f7-d11-m0840-s200004`, 23,286 requests) under `static-k1`:

```
t           = 16391.100897071403
demand      = 0.40464384787069557
progress    = 0.40464384786901064
residual    = 1.6849299733223688e-12   > 1e-12, so NOT completed
next_step_t = 16391.100897071403       == t   (double spacing at 1e4 is ~3.6e-12)
```

`Simulator.step_to` then advances time by exactly zero forever, and
`drain_all`'s own no-progress guard cannot fire because control never returns to
it. The first azure_replay evaluation run spun for 37 minutes with no output and
no traceback before it was killed; this is how it was found.

The trigger needs a long post-horizon drain (so `t` reaches 10⁴ and the time-axis
spacing exceeds 1e-12), many service events per slot (so `progress` accumulates
rounding), and a large backlog. Under-provisioned policies on heavy traces hit all
three — so the defect selectively makes the **cheapest end of the `Static-k`
reference front unevaluable**, which is the end a benchmark most needs.

`baselines/engine_guard.py` works around it in one line
(`next_step_t = t + max(rem/rate, 1e-9)`), installed once globally at
`runner._ensure_registered` so every method runs under identical physics.
`engine_guard.verify_inert()` recomputes all 30 tuned dev objectives — measured
*before* the guard existed — and finds them **bit-identical (max abs diff 0.0,
30/30)**, which is the evidence that the floor never binds on an episode that
terminated. The proper fix belongs in the core: make the completion tolerance
relative to `demand`.

A second, smaller interoperability defect: `KubeGymEnv` publishes the corpus
episode index under `info["episode"]`, the key Gymnasium's `Monitor` reserves for
the end-of-episode statistics dict. SB3's logger then raises
`TypeError: object of type 'int' has no len()` on its first log dump, so **PPO and
DQN cannot be trained through the standard `stable_baselines3` helper without a
wrapper**. `train_rl.InfoEpisodeKeyGuard` is that wrapper; it preserves the index
under `kubegym_episode`.

---

## 7. The other variance component, measured separately

On builtin `llm_serving` tasks `seed` **does** redraw the work realisation. On
`LLMServing-S1` test (1 episode, 4 work seeds, analytic controllers at shipped or
default parameters, **not** re-tuned), the within-episode SD across work seeds is
4.29 (`static`), 4.85 (`forecast`), 6.71 (`leading`), 6.91 (`queue`), 9.13 (`hpa`)
on mean costs of 20.2–34.6 — a 21–26 % coefficient of variation. On the corpus
tasks that component is **exactly zero** by construction. Pooling the two would
manufacture degrees of freedom, so they are never pooled, and no controller
comparison is drawn from this side study.

PPO and DQN were **not** trained on `llm_serving`, for wall-clock reasons only: a
control step there costs ~13 ms against the 0.7–2.2 ms quoted for
`request_service`, so the same 1.2 M-step cap is about 4.3 h of environment time
per (algorithm, family) on this machine against 0.23–0.73 h for a corpus family.
Running a smaller budget would break the parity the whole protocol exists to
enforce. No conclusion about learned control on `llm_serving` is drawn or implied.

---

## 8. Cost of the run

Ledgered wall-clock across tuning and RL training: **6.59 h** of job time
(`tuning_budget.csv`). Test evaluation added 2,144 s (0.60 h). Total ≈ **7.2 h**
of job time; elapsed wall-clock was about 2 h because up to nine jobs ran
concurrently on 12 cores. No GPU.

---

## 9. Regression fixture

`regression_fixture.json` carries 40 per-(family, method) result entries and 40
budget entries with three tolerance classes: `exact` for protocol integers
(budget spends, configuration counts, parity disagreement counts, the oracle
property booleans), `deterministic` with `rtol = 1e-9` for analytic metrics (a
corpus rollout is a pure function of parameters, episode and work seed), and
`stochastic` for learned metrics with `rtol = max(3·sd_seeds/|value|, 0.05)`
derived from *measured* seed dispersion and recorded alongside each entry.
`fixture.check()` re-checks a fresh run; the self-check passes 480/480.

---

## 10. What would change these conclusions

1. **Calibrating the four flagged constants.** `rs_service_time_distribution`
   alone sets the SLO ceiling that dominates azure_replay (§5.5).
2. **A larger corpus of non-degenerate traces.** 66 of 114 azure_replay test
   episodes cannot distinguish any method (§5.3); the effective sample size for a
   controller comparison there is 48, not 114.
3. **A larger RL budget.** 1.2 M steps is small by RL standards. The claim here is
   that under a *matched and enforced* cap the learners do not win, not that they
   could not win with more.
4. **Fixing the drain defect in the core** (§6), so the cheap end of the
   `Static-k` front is evaluable without a workaround.
