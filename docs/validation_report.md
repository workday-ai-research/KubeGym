# KubeGym workload corpus -- distributional validation report

**Status: UNCALIBRATED.** Every number below is computed over a corpus whose Azure half is a *reconstruction* (per-minute counts and duration percentiles turned into arrival instants and service times under stated assumptions) and whose rate unit derives from placeholder constants. Nothing here is a measurement of the Azure Functions workload. See `WORKLOADS.md` and the `reconstruction_provenance` block of `corpus_manifest.json`.

## 1. What is compared, and against what

each synthetic family's pooled arrival statistics against the RECONSTRUCTED Azure replay traces of the same split.

> The Azure side is itself a reconstruction. Below one minute the 'real' inter-arrival distribution IS arrival-placement assumption A1, so agreement there compares two models. At and above one minute the count series is the recorded data and the comparison is against data.

Two comparisons are reported for every family:

1. **vs all Azure traces** -- the honest coverage answer. The synthetic families are specified in fractions of single-replica capacity (0.25x to 1.5x, about 1.10 to 6.60 req/s at the provisional capacity), while the Azure replay traces span four and a half orders of magnitude of rate. This comparison therefore *reports the rate mismatch*, and it is large.
2. **vs rate-matched Azure traces** -- Azure traces whose mean rate falls inside the synthetic band. This isolates the *shape* of the arrival process from the rate.

Two-sample Kolmogorov-Smirnov statistic `D` (maximum CDF gap, an effect size) and the 1-Wasserstein / earth-mover distance are reported. **The KS p-values are not the point:** pooled samples run to 10^5-10^6 gaps, where every difference is significant. `D` and the Wasserstein distance are the numbers to read.

## 2.1 Split `train`

Azure side: 185 replay traces, mean rate 5.56e-04 to 8.7 req/s (median 0.033). Rate-matched subset: 7 traces.

### vs ALL Azure traces

| family | n gen | interarrival (raw) D | W (s) | interarrival (normalised) D | W | per-60s-bin count D | W | IoD@1s gen/real | IoD@60s gen/real | ACF60 lag1 gen-real |
|---|---|---|---|---|---|---|---|---|---|---|
| `burst` | 48 | 0.097 | 0.927 | 0.121 | 0.245 | 0.883 | 160.72 | 2.97x | 75.14x | 0.399 |
| `constant` | 10 | 0.157 | 0.952 | 0.022 | 0.053 | 0.880 | 178.71 | 0.98x | 0.59x | 0.034 |
| `diurnal` | 24 | 0.143 | 0.949 | 0.012 | 0.038 | 0.862 | 185.39 | 1.24x | 10.87x | 0.901 |
| `variable` | 40 | 0.143 | 0.939 | 0.040 | 0.094 | 0.850 | 152.40 | 1.44x | 17.32x | 0.656 |

### vs RATE-MATCHED Azure traces

| family | n gen | interarrival (raw) D | W (s) | interarrival (normalised) D | W | per-60s-bin count D | W | IoD@1s gen/real | IoD@60s gen/real | ACF60 lag1 gen-real |
|---|---|---|---|---|---|---|---|---|---|---|
| `burst` | 48 | 0.226 | 0.084 | 0.137 | 0.279 | 0.294 | 98.60 | 2.84x | 42.37x | -0.121 |
| `constant` | 10 | 0.112 | 0.083 | 0.006 | 0.012 | 0.253 | 49.15 | 0.94x | 0.34x | -0.486 |
| `diurnal` | 24 | 0.127 | 0.099 | 0.027 | 0.070 | 0.276 | 60.58 | 1.19x | 6.13x | 0.381 |
| `variable` | 40 | 0.119 | 0.076 | 0.056 | 0.129 | 0.180 | 48.79 | 1.38x | 9.76x | 0.136 |

## 2.2 Split `dev`

Azure side: 82 replay traces, mean rate 5.56e-04 to 8.3 req/s (median 0.047). Rate-matched subset: 7 traces.

### vs ALL Azure traces

| family | n gen | interarrival (raw) D | W (s) | interarrival (normalised) D | W | per-60s-bin count D | W | IoD@1s gen/real | IoD@60s gen/real | ACF60 lag1 gen-real |
|---|---|---|---|---|---|---|---|---|---|---|
| `burst` | 48 | 0.163 | 1.391 | 0.100 | 0.249 | 0.887 | 179.90 | 3.01x | 45.08x | 0.353 |
| `constant` | 10 | 0.120 | 1.395 | 0.045 | 0.134 | 0.879 | 187.32 | 1.00x | 0.35x | 0.048 |
| `diurnal` | 24 | 0.131 | 1.411 | 0.017 | 0.082 | 0.845 | 199.04 | 1.27x | 5.82x | 0.908 |
| `variable` | 40 | 0.108 | 1.379 | 0.031 | 0.123 | 0.823 | 172.27 | 1.62x | 13.99x | 0.700 |

### vs RATE-MATCHED Azure traces

| family | n gen | interarrival (raw) D | W (s) | interarrival (normalised) D | W | per-60s-bin count D | W | IoD@1s gen/real | IoD@60s gen/real | ACF60 lag1 gen-real |
|---|---|---|---|---|---|---|---|---|---|---|
| `burst` | 48 | 0.207 | 0.079 | 0.136 | 0.277 | 0.576 | 126.31 | 2.84x | 22.38x | 0.462 |
| `constant` | 10 | 0.079 | 0.040 | 0.009 | 0.015 | 0.337 | 57.45 | 0.94x | 0.17x | 0.157 |
| `diurnal` | 24 | 0.094 | 0.053 | 0.026 | 0.065 | 0.327 | 57.79 | 1.20x | 2.89x | 1.017 |
| `variable` | 40 | 0.095 | 0.051 | 0.069 | 0.156 | 0.288 | 72.28 | 1.53x | 6.95x | 0.808 |

## 2.3 Split `test`

Azure side: 102 replay traces, mean rate 5.56e-04 to 10.4 req/s (median 0.023). Rate-matched subset: 13 traces.

### vs ALL Azure traces

| family | n gen | interarrival (raw) D | W (s) | interarrival (normalised) D | W | per-60s-bin count D | W | IoD@1s gen/real | IoD@60s gen/real | ACF60 lag1 gen-real |
|---|---|---|---|---|---|---|---|---|---|---|
| `burst` | 48 | 0.052 | 0.934 | 0.039 | 0.223 | 0.854 | 163.48 | 3.02x | 105.25x | 0.516 |
| `constant` | 10 | 0.108 | 0.971 | 0.111 | 0.330 | 0.819 | 181.02 | 1.02x | 0.87x | 0.088 |
| `diurnal` | 24 | 0.095 | 0.977 | 0.083 | 0.262 | 0.810 | 190.60 | 1.29x | 14.09x | 1.051 |
| `variable` | 40 | 0.089 | 0.945 | 0.052 | 0.220 | 0.792 | 148.77 | 1.66x | 33.59x | 0.823 |

### vs RATE-MATCHED Azure traces

| family | n gen | interarrival (raw) D | W (s) | interarrival (normalised) D | W | per-60s-bin count D | W | IoD@1s gen/real | IoD@60s gen/real | ACF60 lag1 gen-real |
|---|---|---|---|---|---|---|---|---|---|---|
| `burst` | 48 | 0.092 | 0.085 | 0.055 | 0.236 | 0.351 | 99.78 | 1.28x | 1.52x | 0.476 |
| `constant` | 10 | 0.077 | 0.065 | 0.096 | 0.286 | 0.207 | 47.72 | 0.43x | 0.01x | 0.048 |
| `diurnal` | 24 | 0.063 | 0.058 | 0.068 | 0.217 | 0.238 | 50.81 | 0.55x | 0.20x | 1.010 |
| `variable` | 40 | 0.068 | 0.070 | 0.038 | 0.181 | 0.124 | 38.46 | 0.71x | 0.48x | 0.783 |

## 3. Where the generator does NOT match the trace

Read this section before using the synthetic families as a stand-in for the real one. The `rate_per_*_bin` and raw-inter-arrival axes are dominated by the rate-coverage gap: the generator does not span the Azure rate range, and the all-traces `D` around 0.85 on the per-60s-bin count distribution *is* that statement. The `iod_*` and `acf60_*` axes are KS statistics over per-trace scalars, so their sample size is the number of TRACES; in the rate-matched columns the Azure side is only 7 (train), 7 (dev), 13 (test) traces, and a `D` near 1 there reflects that small sample as much as a real mismatch. Read the ratio columns of section 2 for those axes instead.

* **`burst`** -- worst agreement (by KS D) against all Azure traces: `rate_per_60s_bin` D=0.883, `rate_per_15s_bin` D=0.872, `iod_1s` D=0.717.
  Rate-matched: `iod_1s` D=0.979, `iod_15s` D=0.979, `iod_60s` D=0.979.
* **`constant`** -- worst agreement (by KS D) against all Azure traces: `rate_per_60s_bin` D=0.880, `rate_per_15s_bin` D=0.866, `iod_60s` D=0.519.
  Rate-matched: `iod_15s` D=1.000, `iod_60s` D=1.000, `acf60_lag1` D=0.900.
* **`diurnal`** -- worst agreement (by KS D) against all Azure traces: `acf60_lag1` D=0.898, `rate_per_60s_bin` D=0.862, `rate_per_15s_bin` D=0.854.
  Rate-matched: `iod_1s` D=0.833, `iod_15s` D=0.792, `iod_60s` D=0.750.
* **`variable`** -- worst agreement (by KS D) against all Azure traces: `rate_per_60s_bin` D=0.850, `rate_per_15s_bin` D=0.838, `iod_15s` D=0.670.
  Rate-matched: `iod_1s` D=0.925, `iod_15s` D=0.857, `iod_60s` D=0.857.

## 4. Sensitivity of the corpus to the arrival-placement assumption (A1)

Trace statistics over **all 339** replay traces with at least 10 arrivals; downstream queueing over a stratified subsample of **42** traces run through the `request_service` model at a fixed static replica count (held identical across arms so the comparison is paired).

Paired median ratio against the `uniform` reference arm, with the 10th-90th percentile band over traces:

| measure | scope | poisson | deterministic | bursty_batch |
|---|---|---|---|---|
| `iod_1s` | all 339 traces | 1.009x [0.97, 1.10] | 0.903x [0.38, 1.00] | 2.794x [1.00, 5.28] |
| `iod_15s` | all 339 traces | 1.189x [0.98, 1.48] | 0.827x [0.28, 1.00] | 1.554x [1.00, 3.94] |
| `iod_60s` | all 339 traces | 1.529x [0.99, 5.41] | 1.000x [1.00, 1.00] | 1.000x [1.00, 1.00] |
| `peak_to_mean_15s` | all 339 traces | 1.157x [0.88, 2.11] | 0.750x [0.50, 1.00] | 1.333x [1.00, 2.00] |
| `sim_mean_latency_s` | 42-trace subsample | 0.999x [0.93, 1.11] | 1.000x [0.96, 1.00] | 1.034x [1.00, 1.24] |
| `sim_p95_latency_s` | 42-trace subsample | 1.000x [0.97, 1.14] | 1.000x [0.98, 1.00] | 1.000x [1.00, 1.79] |
| `sim_queue_violation_rate` | 42-trace subsample | 1.024x [0.82, 1.21] | 0.972x [0.64, 1.01] | 1.096x [1.03, 1.34] |

**The two pools are not interchangeable.** The trace-statistic rows are pooled over every replay trace (339); the queueing rows are the 42-trace simulated subsample, which skews denser because it excludes traces below 50 arrivals, and therefore has different burstiness magnitudes. Where both pools report the same statistic they are shown side by side below; never carry a magnitude across the two.

| statistic | all-traces pool | simulated subsample |
|---|---|---|
| `iod_1s` `bursty_batch`/`uniform` | 2.794x (n=339) | 3.667x (n=42) |
| `iod_15s` `bursty_batch`/`uniform` | 1.554x (n=339) | 1.516x (n=42) |

**Result.** Sub-minute burstiness moves substantially between arms -- the batch-arrival arm several-fold, the deterministic arm downward -- and the accompanying trace-statistics table carries the magnitudes at each bin width and their scope. The downstream queueing outcome does not follow: over the 42-trace simulated subsample, across mean latency, p95 latency and the queue-delay violation rate, no arm's paired median differs from the uniform reference by more than 1.10x. The arrival-placement assumption is therefore a modelling detail rather than a driver of difficulty at the median trace, which is a useful negative result: a reviewer need not accept A1 to accept a controller comparison run over this corpus. The tail is not negligible, though: in the upper decile of that subsample the batch-arrival arm inflates the same quantities by up to 1.79x, so a per-trace claim -- as opposed to a corpus-median claim -- does carry A1 as an uncertainty. Reporting a controller comparison as a median over the corpus is robust to A1; reporting the worst-case trace is not.

The `bursty_batch` arm is the intended stress case for this question: it is the only arm that changes sub-minute structure by more than Poisson noise, and it is also the arm with the least support in the data -- nothing in the Azure trace says whether serverless invocations arrive in batches within a minute.

Three structural facts make this table readable. (i) `uniform`, `deterministic` and `bursty_batch` all preserve the recorded per-minute count exactly, so their 60 s index of dispersion is identical by construction -- the 1.000x entries are an identity, not a measurement. (ii) The `poisson` arm re-draws each minute's count from Poisson(c), which is why it is the only arm that moves the 60 s statistics; the effect is largest on the `steady` regime, where the recorded minute-to-minute variance is near zero and the injected count noise dominates. (iii) All of A1's real effect lives below one minute, and the `bursty_batch` arm is the only arm that moves it materially.

## 5. Reading guidance

* A `D` of 0.1 on a pooled sample of 10^5 gaps is a small effect with a p-value of zero. Compare families to each other on `D`, not to a significance threshold.
* The Wasserstein distance on raw inter-arrivals is in seconds and is dominated by the rate gap; the normalised column (each trace's gaps divided by their own mean) is the shape comparison.
* The index-of-dispersion ratios are medians over traces, so a ratio near 1 means the *typical* trace matches, not that every trace does; the per-family p10-p90 bands are in `validation_results.json`.

