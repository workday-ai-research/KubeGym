# KubeGym workload corpus

**Status: UNCALIBRATED, and one half of it is a reconstruction rather than a recording.** Read section 2 before quoting any number computed over this corpus.

The corpus has two halves. The **replay** half takes the real per-minute invocation-count series of 45 Azure Functions applications and turns each hour of it into a request-level trace. The **synthetic** half generates four parametrized families (`constant`, `variable`, `burst`, `diurnal`) over a documented 61-point grid. Both halves are pure functions of (spec, seed): 767 traces, 5,497,539 requests, and every trace regenerates from its manifest entry to a byte-identical checksum.

| | traces | train | dev | test |
|---|---|---|---|---|
| `azure_replay` | 401 | 202 | 85 | 114 |
| `constant` | 30 | 10 | 10 | 10 |
| `variable` | 120 | 40 | 40 | 40 |
| `burst` | 144 | 48 | 48 | 48 |
| `diurnal` | 72 | 24 | 24 | 24 |
| **total** | **767** | 324 | 207 | 236 |

## 1. Dataset provenance and attribution

Source: **Azure Functions Trace 2019 (revision 2, 20200618)**, licence **CC-BY Attribution License (per dataset LICENSE)**. The dataset was downloaded, checksummed and reduced to four parquet tables before this track began; `azure2019_staging_manifest.json` records the source URL, per-file checksums and the exact reductions applied, and its file checksums are embedded in `corpus_manifest.json` under `azure_extract.staging_manifest_checksums`. Staged tables: `azure2019_app_invocations_per_minute.parquet`, `azure2019_app_memory.parquet`, `azure2019_app_trigger_mix.parquet`, `azure2019_function_durations.parquet`. Nothing in this package re-downloads them.

**Required attribution.** Any use of this corpus must cite:

> Shahrad, Fonseca, Goiri, Chaudhry, Batum, Cooke, Laureano, Tresness, Russinovich, Bianchini. 'Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider', USENIX ATC 2020.

### What the source contains, verified against its own documentation

These five facts were read out of `AzureFunctionsDataset2019.md` rather than recalled, because the reconstruction turns on them:

1. Execution time is in milliseconds and does NOT include cold start time (note 4)
2. The percentile_Average_* columns are weighted percentiles of 30-second-interval AVERAGES, not of individual invocation durations (note 7)
3. Minimum/Maximum are the true extremes but were not recorded for a few functions owing to a field naming issue in some Azure Functions runtime versions (note 6)
4. The durations-table Count comes from a different log than the invocation counts and may rarely diverge (note 5)
5. Invocation counts are recorded after the functions execute (note 3)

Fact 2 is the load-bearing one. The dataset has **no per-request duration** and **no arrival timestamps**; it has per-minute counts and seven weighted percentiles of 30-second-interval averages per function-day.

## 2. The reconstruction, and why it is not data

Producing arrival instants and per-request service times from that summary requires sampling assumptions that are **modelling choices of this benchmark**. `kubegym/workloads/reconstruction.py` names seven of them, gives each a default, a set of alternative arms and a stated bias direction, and emits them as a machine-readable block (`reconstruction_provenance` in the manifest, `calibrated: false`, `kind: "modelling_assumption"`, `is_measured_data: false`). Every trace entry carries the same two flags, so an entry read in isolation still cannot be mistaken for measurement.

| id | assumption | default | alternatives | bias direction |
|---|---|---|---|---|
| A1 | arrival placement within a minute | `uniform` | `poisson`, `deterministic`, `bursty_batch` | unknown a priori -- **measured**, section 4 |
| A2 | service time from the percentile summary | linear inversion of the 7 knots | `loglinear` | **OPTIMISTIC**: percentiles are of 30 s averages, so within-interval variance is absent; lower dispersion at equal utilisation means shorter queues |
| A3 | which function an invocation belongs to | `count_weighted` | `uniform_functions` | the staged counts are application-level, so attribution is invented; count-weighting matches the durations table's own weights |
| A4 | functions with missing `Minimum`/`Maximum` | `percentile_bounds` | `true_extremes`, `drop_functions` | **none by construction**: the default never reads the extremes, so a missing one changes no draw and nothing is imputed |
| A5 | service-time floor | 1 ms | any value | prevents zero-length requests; raises the mean slightly |
| A6 | cold start | excluded from service time | n/a | grounded in the source (note 4); cold start enters once via the replica pool, so it is not double counted |
| A7 | request class label | the application's dominant trigger group | none/off | grounded in the trigger-mix table; coarse -- one label per application, not per request |

What is real and what is not, per replay trace:

* **Real**: the per-minute invocation **count** series of a named application on a named source day, and the seven duration percentiles of each of its functions on that day.
* **Reconstructed**: every arrival instant, the function each invocation belongs to, and every service time.

Consequence for the paper: a statement of the form *"controller X beats controller Y on the Azure replay traces"* is a statement about this model of the Azure workload. Section 4 measures how much of that statement depends on A1.

### Duration-table audit

The documented field-naming defect is present but rare. Over the full table (662,927 function-days, 82,375 functions, 24,937 applications): 29 rows have an unusable `Minimum`/`Maximum` pair (0.0044%) -- 1 null, 7 negative, 20 with `Minimum` above p0 and 1 with `Maximum` below p100, across 23 functions in 14 applications. 7 rows have a negative percentile knot and are clamped to zero; **0 rows have non-monotone percentiles**. Among the 45 selected applications (2,248 function-days) the count is 0.

`Maximum == 0` appears on 6,085 full-table rows (12 of them in the selected subset). These are **not** treated as missing: their percentile knots are zero too, so the row is internally consistent and describes a function whose 30 s averages rounded to zero milliseconds. The 1 ms floor (A5) then applies.

## 3. Application selection

`select_applications()` is a deterministic function of the staged tables and a fixed salt (`kubegym-azure2019-selection-v1`) -- no RNG, no manual curation. In order:

1. **Candidate filter.** Present on all 14 days of the invocation table, duration rows on all 14 days, a trigger-mix row, and a 14-day peak at or below 4800 invocations/min. Of 24,274 applications, 13,057 are present on all 14 days and 12,792 survive the filter.
2. **Regime label** from 14-day statistics, evaluated in this order -- the order matters, because a diurnal application also has inflated dispersion and a very sparse series has a large relative Fourier amplitude by construction:

   | regime | rule | candidates |
   |---|---|---|
   | `sparse` | zero-minute fraction >= 0.95 | 4,294 |
   | `intermittent` | 0.60 <= zero-minute fraction < 0.95 | 5,105 |
   | `diurnal` | relative 24 h Fourier amplitude >= 0.5 | 633 |
   | `bursty` | index of dispersion >= 4.0 | 523 |
   | `steady` | otherwise | 2,237 |

3. **Volume tercile** within each regime, by 14-day total invocations.
4. **Trigger spread and split assignment.** Within each of the 15 (regime x tercile) cells, candidates are ordered by a salted hash and picked round-robin over dominant trigger groups; the three picks of a cell go to train, dev and test respectively. That makes the three application sets disjoint **by construction**, gives each split all 5 regimes x 3 terciles, and yields 45 applications, 15 per split.

### The selection bias a reviewer will look for, stated with its size

The peak filter is a **capacity-envelope filter and a real bias**. It removes 194 of the continuously-present applications (1.49%) but **78.7% of total 14-day invocation volume**. peak per-minute rate above the modelled platform's serving capacity (n_replicas_max=20 x rs_max_concurrency=4). A trace above the ceiling measures the ceiling, not the controller. Direction: the corpus under-represents the very-high-volume head of the Azure population, which is where most invocation VOLUME lives even though it is a small share of applications.

The corpus also **deliberately oversamples the rare regimes** -- 20% of traces per regime against a population that is 33% sparse and 39% intermittent -- because a corpus drawn in population proportion would be almost entirely low-rate functions and would not exercise a controller at all. The consequence is that corpus-level averages are **not** estimates of Azure-population averages; per-regime numbers are the reportable unit. `population_comparison` in the manifest gives the selected-vs-population distribution on five axes:

| axis | population p10 / p50 / p90 (n=13,057) | selected p10 / p50 / p90 (n=45) |
|---|---|---|
| mean rate (inv/min) | 0.00139 / 0.233 / 8.8 | 0.0115 / 2.16 / 73.7 |
| peak (inv/min) | 1 / 4 / 260 | 2 / 30 / 938 |
| 14-day total invocations | 28 / 4,701 / 177,418 | 233 / 43,488 / 1,485,943 |
| index of dispersion | 0.449 / 0.999 / 37.2 | 0.358 / 4.76 / 116 |
| relative 24 h amplitude | 0.000299 / 0.0782 / 1.99 | 0.00138 / 0.199 / 1.74 |

Replay traces span 1 to 37,283 requests per hour (quartiles 15, 84, 480; p95 15,908). The low end is intentional: those are the sparse and intermittent applications where cold starts dominate and scale-to-zero is the whole decision. A single-request episode is a legitimate regime, not a degenerate one -- but a p95 latency over 3 requests is noise, which is why the simulated sensitivity study in section 4 excludes traces below 50 arrivals and says so.

## 4. Sensitivity to the arrival-placement assumption (A1)

Every replay trace was rebuilt under all four placement arms. Trace statistics cover all 339 replay traces with at least 10 arrivals; the downstream measure runs a stratified subsample of 42 traces (>=50 arrivals, all 5 regimes) through the `request_service` model at a **fixed static replica count**, computed from the reference arm and held identical across arms so the four arms face the same provisioning decision. Static provisioning is deliberate: a closed-loop controller would absorb part of the difference and confound the assumption with the controller.

Paired median ratio against the `uniform` reference, with the 10th-90th percentile band over traces:

| measure | pool | `poisson` | `deterministic` | `bursty_batch` |
|---|---|---|---|---|
| index of dispersion, 1 s bins | all 339 replay traces | 1.009x [0.97, 1.10] | 0.903x [0.38, 1.00] | 2.794x [1.00, 5.28] |
| index of dispersion, 15 s bins | all 339 replay traces | 1.189x [0.98, 1.48] | 0.827x [0.28, 1.00] | 1.554x [1.00, 3.94] |
| index of dispersion, 60 s bins | all 339 replay traces | 1.529x [0.99, 5.41] | 1.000x [1.00, 1.00] | 1.000x [1.00, 1.00] |
| peak-to-mean, 15 s bins | all 339 replay traces | 1.157x [0.88, 2.11] | 0.750x [0.50, 1.00] | 1.333x [1.00, 2.00] |
| mean request latency | 42-trace simulated subsample | 0.999x [0.93, 1.11] | 1.000x [0.96, 1.00] | 1.034x [1.00, 1.24] |
| p95 request latency | 42-trace simulated subsample | 1.000x [0.97, 1.14] | 1.000x [0.98, 1.00] | 1.000x [1.00, 1.79] |
| queue-delay violation rate | 42-trace simulated subsample | 1.024x [0.82, 1.21] | 0.972x [0.64, 1.01] | 1.096x [1.03, 1.34] |

**The two pools are not interchangeable.** The trace-statistic rows are pooled over all 339 replay traces; the queueing rows are the 42-trace simulated subsample, which skews denser (it excludes traces below 50 arrivals) and therefore has different burstiness magnitudes -- `iod_1s` for `bursty_batch`, for instance, is 2.794x pooled but 3.667x on that subsample. Read each row against its own pool and do not carry a magnitude across the two.

**Result.** Sub-minute burstiness moves substantially between arms -- the batch-arrival arm several-fold, the deterministic arm downward -- and the accompanying trace-statistics table carries the magnitudes at each bin width and their scope. The downstream queueing outcome does not follow: over the 42-trace simulated subsample, across mean latency, p95 latency and the queue-delay violation rate, no arm's paired median differs from the uniform reference by more than 1.10x. The arrival-placement assumption is therefore a modelling detail rather than a driver of difficulty at the median trace, which is a useful negative result: a reviewer need not accept A1 to accept a controller comparison run over this corpus. The tail is not negligible, though: in the upper decile of that subsample the batch-arrival arm inflates the same quantities by up to 1.79x, so a per-trace claim -- as opposed to a corpus-median claim -- does carry A1 as an uncertainty. Reporting a controller comparison as a median over the corpus is robust to A1; reporting the worst-case trace is not.

Three structural facts make the table readable.

1. `uniform`, `deterministic` and `bursty_batch` all preserve the recorded per-minute count exactly, so their 60 s index of dispersion is **identical by construction** -- the `1.000x` entries are an identity, not a measurement.
2. `poisson` re-draws each minute's count from Poisson(c), so it is the only arm that moves the 60 s statistics. The effect is concentrated on the `steady` regime (1.53x corpus-wide but 6.7x within `steady`), where the recorded minute-to-minute variance is near zero and the injected count noise dominates.
3. All of A1's real effect lives **below one minute**, and `bursty_batch` is the only arm that moves it materially. It is also the arm with the least support in the data: nothing in the Azure trace says whether serverless invocations arrive in batches within a minute.

Per-regime paired ratios for the 1 s index of dispersion, within the **all-339-trace pool** (the `sparse` cell's 3.67x is a per-regime figure and is not the pooled 2.794x above, nor the coincidentally similar subsample figure):

| regime | `poisson` | `deterministic` | `bursty_batch` |
|---|---|---|---|
| `sparse` | 1.01x | 0.78x | 3.67x |
| `intermittent` | 1.00x | 1.00x | 1.13x |
| `steady` | 1.03x | 0.97x | 1.92x |
| `diurnal` | 1.01x | 0.84x | 2.74x |
| `bursty` | 1.02x | 0.69x | 4.45x |

**How to use this.** Report controller comparisons as medians (or per-regime medians) over the corpus and A1 is not a threat to the conclusion. Report a single worst-case trace and it is: rerun that trace under `bursty_batch` and quote the range.

## 5. Generator families and parameter grid

61 grid points over four families. Rates are expressed as **fractions of single-replica capacity** so a trace is portable across service-model configs; the conversion constant is `PROVISIONAL_CAPACITY_RPS = 4.4` req/s = `rs_max_concurrency` (4) / mean reconstructed service time (0.900 s). Both inputs are placeholders from `request_service_default.json`, so **the rate unit is a labelled placeholder**; an arithmetic consequence of placeholders is still a placeholder.

| family | points | parameters | what it tests |
|---|---|---|---|
| `constant` | 5 | `load` in {0.25, 0.5, 0.75, 1.0, 1.5} | steady-state provisioning and the cost of over-provisioning; the levels straddle single-replica capacity so both under- and over-provisioned regimes appear |
| `variable` | 20 | `mean_load` in {0.5, 1.0} x `sigma` in {0.3, 0.6, 1.0} x `tau_s` in {60, 300, 900} s, plus 2 non-stationary arms (`tau_s = inf`, `sigma` in {0.3, 0.6}) | tracking a stochastic rate whose correlation time is above, near and below the control interval; the regime where a reactive controller provably cannot keep up is included on purpose |
| `burst` | 24 | `base_load` in {0.3, 0.6} x `magnitude` in {3, 6, 12} x `duration_s` in {60, 300} x {single, repeating every 900 s} | scale-up latency against a step change, with bursts both shorter and longer than a cold start |
| `diurnal` | 12 | `mean_load` in {0.5, 1.0} x `amplitude` in {0.4, 0.8} x `period_s` in {900, 3600, 14400} s | predictable seasonality at periods shorter than, equal to and longer than the 1 h episode |

### The `variable` family, defined

The prior project's generator had constant, diurnal and burst but no `variable`; leaving it implicit is exactly the kind of undefined family a benchmark should not ship, so it is defined explicitly as a **log-Ornstein-Uhlenbeck (mean-reverting) rate process** on a piecewise-constant 60 s grid (matching the granularity at which the real trace is recorded):

```
x_0     ~ N(0, sigma^2)                                  (stationary start)
x_{k+1}  = a x_k + sigma sqrt(1 - a^2) z_k,   a = exp(-grid_dt_s / tau_s)
lam(t)   = mean_load * exp(clip(x_k, +-clip_sigmas*sigma) - sigma^2 / 2)
```

* `mean_load` -- E[lam] in capacity fractions. The `-sigma^2/2` term makes the *unclipped* process have exactly this mean; clipping at the default 3 sigma biases it low by about 0.3%, and each trace records its realised mean in `rate_path.realised_mean_over_target`.
* `sigma` -- standard deviation of log-rate: how far the rate wanders (0.3 is a +-35% band, 1.0 spans an order of magnitude).
* `tau_s` -- mean-reversion time: how *fast* it wanders. This is the knob that interacts with cold-start time and the control interval.
* `grid_dt_s`, `clip_sigmas` -- grid granularity, and the bound that keeps `lam_max` finite so Lewis-Shedler thinning terminates.

Mean reversion rather than a pure random walk makes the family **stationary**, so two seeds of the same spec are exchangeable and a seed-averaged result means something. A non-stationary random-walk arm (`tau_s = inf`) is included, and labelled, for the case where drift is the point.

### Service-time model of the synthetic families

All four families draw service times from the **pooled reconstructed Azure distribution** (inverse CDF of the Count-weighted percentile knots of 238 function-days from the 15 **train** applications on **train days only**: mean 0.900 s, median 0.01736 s, p99 7.43 s -- a heavy tail, and the reason mean and median differ by 50x). Holding the service-time marginal fixed across families is deliberate: it makes every generated-vs-real discrepancy in section 7 attributable to the **arrival process alone**. It also means the synthetic families inherit assumption A2 in full. A `lognormal` alternative is provided so a family can be generated without the Azure extract present.

## 6. Splits: the rule, and how it is checked

A trace belongs to exactly one of `train` / `dev` / `test`, and the three share nothing on four axes:

| axis | rule | why |
|---|---|---|
| **applications** | the 3 picks of each (regime x tercile) cell go to train/dev/test; 15 applications per split | disjoint by construction, and every split gets all 5 regimes x 3 terciles |
| **time segments** | the 14 source days are partitioned: train {1,2,3,6,8,9,13}, dev {4,10,14}, test {5,7,11,12}. Within a day, segments are non-overlapping whole-hour windows | no minute of source time appears in two splits |
| **seeds** | disjoint bands: train [0, 100000), dev [100000, 200000), test [200000, 300000) | no realisation is shared even where parameters are |
| **service-time fit** | the pooled synthetic service-time model is fitted on train applications and train days only | no dev/test duration information reaches any trace |

The day partition is deliberately **non-contiguous** so that each split contains at least one of days 6, 7, 13, 14, which carry markedly fewer distinct functions than the rest. Grouping them into one split would make that split systematically different.

**Synthetic parameters are shared across splits on purpose.** The synthetic families exist to measure generalisation over *realisations* of a stated rate process, not over hyperparameters; hiding grid points from train would make the dev/test numbers measure extrapolation to unseen parameters, a different question. The disjointness that matters for the synthetic half is the seed band, and it is enforced.

`verify_splits()` re-derives every claim above from the manifest -- pairwise application, day and seed-set intersections, per-(app, day, segment) uniqueness, seed-band membership, and absence of duplicate seeds anywhere in the corpus. It is exercised by `kubegym/tests/test_workload_corpus.py` and by `cli verify`. Current status: all checks pass, 0 overlaps on every axis.

## 7. Distributional validation

Full numbers and reading guidance are in `validation_report.md`; `validation_results.json` has the raw battery. Summary of the honest picture, split `train`:

The Azure replay traces span 0.00056 to 8.74 req/s (median 0.0333), while the synthetic families cover roughly 1.1 to 6.6 req/s. **The generator does not span the Azure rate range**, and the per-60s-count KS statistics of ~0.85 against all Azure traces are that statement, not a subtle distributional mismatch. Against the rate-matched subset (7 of 185 traces) the same statistic falls to 0.18-0.29.

| family | interarrival (normalised) KS D, all / rate-matched | per-60s count KS D, all / rate-matched | IoD@1s gen/real | IoD@60s gen/real | ACF60 lag1 gen-real |
|---|---|---|---|---|---|
| `constant` | 0.022 / 0.006 | 0.880 / 0.253 | 0.98x | 0.59x | +0.034 |
| `variable` | 0.040 / 0.056 | 0.850 / 0.180 | 1.44x | 17.32x | +0.656 |
| `burst` | 0.121 / 0.137 | 0.883 / 0.294 | 2.97x | 75.14x | +0.399 |
| `diurnal` | 0.012 / 0.027 | 0.862 / 0.276 | 1.24x | 10.87x | +0.901 |

### Where the generator does not match, stated plainly

* **Rate coverage.** The largest mismatch on every family. Not fixable by tuning the arrival process: the families are specified in capacity fractions and the Azure population is dominated by functions three orders of magnitude below single-replica capacity. A downstream paper that needs the low-rate regime should use the replay half, which covers it.
* **Minute-scale burstiness of `burst` and `variable`.** IoD@60s runs 75x (burst) and 17x (variable) the median replay trace. These families are *designed* to be more variable than a typical application, so this is intended, but it means they are not a substitute for real traces when the question is 'how bursty is real serverless traffic'.
* **`constant` is less bursty than the real median** (IoD@60s 0.59x): a homogeneous Poisson process is smoother than any real application. That is the point of having it as a floor case, and it is a mismatch.
* **Rate memory.** `diurnal` carries far more one-minute autocorrelation than the real median trace (+0.90) and `constant` almost none (+0.03); the replay traces sit near zero at the median because half of them are sparse. `variable` at +0.66 is the closest of the four on this axis.
* **Inter-arrival shape is the one axis where agreement is good** (normalised KS D of 0.01 to 0.14). That is expected and is *not* strong evidence: below one minute the 'real' inter-arrival distribution **is** assumption A1, so this comparison is between two models, not against ground truth. `validation_report.md` marks every row with which side of that line it falls on.

A caveat on the trace-level axes (`iod_*`, `acf60_*`): their sample size is the number of *traces*, and the rate-matched Azure side is only 7 (train), 7 (dev), 13 (test) traces. A large KS statistic in those columns partly reflects the small held-in sample; read the ratio columns instead.

## 8. Determinism and regenerability

Every trace is a pure function of (spec, seed). The mechanism is the prior project's substream scheme, extended to accept multiple named parts so that hundreds of traces each get independent streams:

```python
substream(seed, "arrival", app_hash, day, segment)   # random.Random
np_substream(seed, "service", trace_tag)             # numpy Generator(PCG64)
```

Streams in use, and the closed set is recorded in the manifest under `rng_scheme.streams`: `arrival`, `rate_process`, `func_attrib`, `service`, `mix`, `segment`. Because each component has its own stream, changing the service-time model cannot move the arrival times and changing a burst magnitude cannot move the service draws -- both properties are unit-tested.

**Version stability.** The numpy substreams call **only** `Generator.random(n)`; normals, geometrics, categoricals and Poissons are derived from those uniforms by explicit inverse transform in `kubegym/workloads/seeding.py`. That is deliberate: it keeps byte-level output independent of which numpy version implements `standard_normal` or `poisson`, so a checksum recorded today still verifies on a future numpy. Do not call any other `Generator` method on a substream.

**Per-request work is sampled at build time, never inside the episode.** Two controllers running the same (trace, seed) see identical arrival times *and* identical service times, so every comparison in the benchmark is paired (`INTERFACE.md` section 4). Drawing in the episode would silently break that; the test suite asserts equality of the full demand vector across two different static replica counts, for both trace kinds.

**Checksums.** A trace's digest is the SHA256 of its canonical JSONL bytes, which is also the SHA256 of the materialised file, so a shipped file and its manifest entry are checkable with `sha256sum`. `cli verify` regenerates every trace from its manifest entry alone and compares. Current status: **767/767 traces regenerate to identical digests, 30/30 materialised files match**, 0 mismatches.

## 9. What is shipped, and how to regenerate the rest

The full corpus is 5,497,539 requests, which is too large to ship as flat files. Shipped instead:

* `corpus_manifest.json` -- all 767 trace specs, parameters, split, horizon, checksum, per-trace `tests` string, and the full provenance blocks.
* `traces/<split>/<trace_id>.jsonl` -- 30 materialised traces, 2 per (family, split), as a representative subset and a checksum cross-check.
* `azure2019_selection.npz` / `.json` -- the small checksummed extract (per-minute counts and duration knots for the 45 selected applications). **The replay half is regenerable from this alone**; the 2 GB source is not needed.
* `validation_results.json`, `sensitivity_arrival_placement.json` -- the analysis outputs.

Regenerate one trace, or all of them:

```bash
python -m kubegym.workloads.cli regenerate --corpus corpus \
    --trace-id azure-2e862038-d05-m0000-s200013 --out trace.jsonl
python -m kubegym.workloads.cli verify --corpus corpus     # all traces + split checks
```

Rebuild the whole corpus from the extract (about 7 s):

```bash
python -m kubegym.workloads.cli build       --corpus corpus
python -m kubegym.workloads.cli sensitivity --corpus corpus
python -m kubegym.workloads.cli validate    --corpus corpus --out .
python -m kubegym.workloads.cli figures     --corpus corpus --out figures
```

Rebuild the extract itself from the staged parquet tables (needs them present):

```bash
python -m kubegym.workloads.cli build-extract \
    --invocations azure2019_app_invocations_per_minute.parquet \
    --durations   azure2019_function_durations.parquet \
    --triggers    azure2019_app_trigger_mix.parquet \
    --staging-manifest azure2019_staging_manifest.json --out corpus
```

## 10. How a downstream paper adds a family

Adding a synthetic family is four steps and touches two files.

1. **Define the rate schedule.** Subclass `RateSchedule` in `kubegym/workloads/generator.py` with a frozen dataclass, a `family` string, `realise(horizon_s, seed, tag) -> RatePath` and a `spec()` dict. Draw any randomness from `np_substream(seed, "rate_process", tag)` and only via `.random(n)` -- use the inverse-transform helpers in `seeding.py` for anything non-uniform, or add a new helper there. Rates are **capacity fractions**, not req/s. Put `lam_max` under a finite bound so Lewis-Shedler thinning terminates.
2. **Register it** in `SCHEDULE_CLASSES` and add the family name to `FAMILIES`, so `schedule_from_spec` can rebuild it and regeneration works from the manifest alone.
3. **Add grid points** to `parameter_grid()` in `kubegym/workloads/corpus.py`, each with a `tests` string saying what the point is for, and an entry in `FAMILY_INTENT`. If the new family needs a new stochastic component, add its stream name to `STREAM_NAMES` -- appending never shifts an existing stream.
4. **Rebuild and re-validate.** `cli build` then `cli validate`; the new family appears in `validation_report.md` automatically, with its own mismatch line. Add a test in `kubegym/tests/test_workload_generator.py` pinning whatever property the family claims (a stated mean, a stated correlation structure), because that claim is what a reviewer will check.

Adding a *replay* source is different: extend `select_applications()`'s criteria, keep the selection deterministic and salted, rebuild the extract, and report the new subset against the population as section 3 does. Do not hand-pick applications -- the selection rule being mechanical is what makes the corpus auditable.

## 11. Files

| file | what |
|---|---|
| `kubegym/workloads/seeding.py` | substream scheme and version-stable inverse-transform helpers |
| `kubegym/workloads/reconstruction.py` | assumptions A1-A7, the arrival-placement arms, the service-time inverse CDF, the duration-table audit |
| `kubegym/workloads/azure2019.py` | application selection, the extract, `AzureReplaySource` |
| `kubegym/workloads/generator.py` | the four families including `VariableRate`, thinning, service-time model |
| `kubegym/workloads/corpus.py` | parameter grid, planning, canonical serialization, splits, manifest, verification |
| `kubegym/workloads/validation.py` | KS / Wasserstein / index-of-dispersion / ACF battery |
| `kubegym/workloads/sensitivity.py` | the A1 sensitivity study, trace-level and downstream |
| `kubegym/workloads/reports.py` | validation results and the generated markdown report |
| `kubegym/workloads/diagnostics.py` | the four figures |
| `kubegym/workloads/cli.py` | `build-extract`, `build`, `verify`, `sensitivity`, `validate`, `figures`, `regenerate` |
| `kubegym/tests/conftest_workloads.py` | synthetic extract fixture so tests need no Azure data |
| `kubegym/tests/test_workload_reconstruction.py` | 23 tests pinning A1-A7 |
| `kubegym/tests/test_workload_generator.py` | 20 tests on the families |
| `kubegym/tests/test_workload_corpus.py` | 21 tests: determinism, regeneration, splits, manifest, build-time sampling |

The package must be importable: `pip install -e .` from the package root, or set `PYTHONPATH` to it.

## 12. Known limitations

1. **The replay half is a reconstruction, not a recording** (section 2). This is the limitation, not a caveat on one.
2. **Service-time dispersion is understated** (A2), because the source gives percentiles of 30 s averages. The bias is optimistic and its magnitude is not bounded by this dataset.
3. **The corpus under-represents high-volume applications** by 79% of invocation volume (section 3), and oversamples rare regimes, so corpus averages are not population averages.
4. **The rate unit is a placeholder** derived from placeholder constants, so absolute utilisation levels in the synthetic half are not calibrated.
5. **The generator does not span the Azure rate range** (section 7).
6. **The calendar alignment of days 1-14 is unknown** -- the source documents no collection-day mapping, so no weekday/weekend interpretation is asserted anywhere.
7. **A1's tail matters even though its median does not** (section 4): per-trace worst-case claims carry the assumption.

