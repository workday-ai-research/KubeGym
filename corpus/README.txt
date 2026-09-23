KubeGym workload corpus -- shipped contents
===========================================

UNCALIBRATED. The Azure half of this corpus is a RECONSTRUCTION, not a recording: the source
dataset provides per-minute invocation COUNTS and duration percentiles of 30-second averages, not
arrival timestamps or per-request durations. Arrival instants, function attribution and every
service time are modelling assumptions of this benchmark. See WORKLOADS.md section 2 and the
`reconstruction_provenance` block of corpus_manifest.json (calibrated: false,
kind: "modelling_assumption").

Dataset: Azure Functions Trace 2019 (revision 2, 20200618), CC-BY. Required attribution:
  Shahrad, Fonseca, Goiri, Chaudhry, Batum, Cooke, Laureano, Tresness, Russinovich, Bianchini.
  "Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud
  Provider." USENIX ATC 2020.

CONTENTS
  corpus_manifest.json                    all 767 trace specs: parameters, split, horizon, seed,
                                          sha256, n_requests, per-trace `tests` string, summary
                                          statistics, plus the full provenance blocks (dataset,
                                          licence, attribution, selection report, population
                                          comparison, duration audit, reconstruction assumptions,
                                          split rule, day/seed partitions, parameter grid)
  azure2019_selection.npz                 the checksummed extract: per-minute invocation counts
  azure2019_selection.json                (45 apps x 14 days x 1440 min) and per-function-day
                                          duration percentile knots, plus selection metadata.
                                          THE REPLAY HALF IS FULLY REGENERABLE FROM THIS -- the
                                          2 GB source is not needed.
  traces/<split>/<trace_id>.jsonl         30 materialised traces (2 per family x split) as a
                                          representative subset. One JSON object per request:
                                          {"seq", "t_arrival", "service_time_s"}. A file's sha256
                                          equals its manifest entry's sha256.
  validation_results.json                 the full KS / Wasserstein / dispersion / ACF battery,
                                          per split and family, all-traces and rate-matched
  sensitivity_arrival_placement.json      the A1 sensitivity study: 4 arms x 339 replay traces
                                          (trace statistics) and 4 arms x 42 traces (downstream
                                          queueing), plus per-regime breakdown and the verdict

WHY ONLY 30 TRACES ARE MATERIALISED
  The full corpus is 5,497,539 requests. Every trace is a pure function of (spec, seed), so the
  specs plus the extract are a complete description. Regenerate:

    python -m kubegym.workloads.cli regenerate --corpus . --trace-id <id> --out trace.jsonl
    python -m kubegym.workloads.cli verify --corpus .     # all 767 traces + split checks
    python -m kubegym.workloads.cli build  --corpus .     # rebuild the whole corpus (~7 s)

  Verified at release: 767/767 traces regenerate to identical digests; 30/30 materialised file
  hashes match their manifest entries; 0 split overlaps on applications, days, segments or seeds.
