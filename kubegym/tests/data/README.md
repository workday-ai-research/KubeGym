# Test fixtures

`S1_dev.jsonl`, `S3_dev.jsonl` are copied VERBATIM from the source vLLM
autoscaling testbed's reference scenario set (`scenarios_traces.tar.gz`,
manifest schema `scenario-manifest-1`).  They are request-arrival traces only:
`seq`, `t_arrival`, `prompt_tokens`, `cls`, `requested_thinking_mode`.  Output
lengths are NOT in the trace; they are drawn at episode build time from the
length model (which is `calibrated: false`).

They are fixtures for the test suite, not the benchmark's workload corpus --
that is a separate phase and lands as its own provenance-tracked artifact.

Provenance of these files:

{
  "S1_dev": {
    "sha256": "2d8c19b0d6ccd8be26ce3d07bb6b897ea52bae2ab76178dc570d1c62737c59f7",
    "n_requests": 225,
    "horizon_s": 300.0,
    "max_t_arrival_s": 299.831
  },
  "S3_dev": {
    "sha256": "4c366a3e2d2e01f521f11b8ba71ab5cdd71ba1bc95dea099d57d067dd179f007",
    "n_requests": 221,
    "horizon_s": 300.0,
    "max_t_arrival_s": 299.838
  }
}

`thinking_flip_prob = 0.0` in the generator that produced them, so
`requested_thinking_mode` is a NOISELESS proxy for the length class.  Any
experiment that conditions on the class signal must say so; see
`class_predictor_accuracy` in the config.
