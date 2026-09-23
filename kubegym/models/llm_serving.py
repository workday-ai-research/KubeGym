"""LLM serving: the calibrated vLLM token/KV/preemption model.

This is a PORT, not a reimplementation.  The physics below are the same physics
as `sim_cluster.py` in the source testbed, moved behind the `ServiceModel` seam.
`tests/test_parity.py` runs the same trace and seed through both and compares
per-request completion times, replica-seconds and preemption counts.

PHYSICAL MODEL (unchanged from the source)
------------------------------------------
One scheduler step advances EVERY running sequence by one token and costs

    t_step(b) = a0 + a1 * b        seconds, for b running sequences

with a0 = 0.01657539 s and a1 = 0.00011063 s/seq, least-squares fitted to
measured single-replica decode throughput of 60.3 / 232.6 / 460.2 tok/s at
b = 1 / 4 / 8 (vLLM 0.25.1, Qwen3-1.7B, one L40S 46 GB; max relative error on
the three fitted points 1.05 %).  Per-token latency therefore degrades with
batch size and aggregate throughput rises sublinearly.

Concurrency is bounded by KV cache, not by a fixed batch limit.  A sequence
occupies `prompt_tokens + generated_tokens` KV tokens and that footprint GROWS
as it decodes.  When the sum would exceed the replica's KV budget (MEASURED:
55 200 tokens) the scheduler preempts, recompute-style: the victim returns to
the HEAD of the waiting queue keeping its generated prefix, and re-pays prefill
on `prompt + generated` tokens when it is re-admitted.  This is what makes
long-CoT traffic qualitatively different from short traffic: the same request
count consumes monotonically more KV as it runs, so a replica that is
comfortable at admission can be preempting 200 s later with no new arrivals.

Admission charges a prefill of `prompt + already-generated` tokens at
`prefill_tokens_per_s`, during which the replica does not decode.

KNOWN INFIDELITIES, with direction (see also INTERFACE.md and unverified_core.md)
--------------------------------------------------------------------------------
* `prefill_tokens_per_s = 6000` is a PLACEHOLDER, never measured.  Every TTFT
  number this model produces is placeholder-derived.
* Decode step time above batch 8 is an unmeasured affine extrapolation.  Real
  engines saturate, so it is OPTIMISTIC about large batches and biases results
  toward making under-provisioning look cheap.
* Chunked prefill is NOT modelled; a whole-prompt prefill stalls the replica.
  This OVERSTATES the inter-token spike an admission causes and UNDERSTATES the
  TTFT of a long prompt under load.
* vLLM's exact preemption victim-selection order was not read from source; this
  model evicts the most-recently-admitted sequence.
* Per-class output lengths come from `length_model.py`, which is
  `calibrated: false` -- the dominant uncalibrated input.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Sequence

from ..core.service import INF, BaseReplicaEngine, Request, Slot
from ..core.workload import JsonlTraceSource, WorkloadSource
from ..provenance import ProvenancedConfig
from .length_model import LengthModel, load_length_model

#: VERIFIED PRESENT on the live server, vLLM 0.25.1 (66 vllm:* families scanned).
METRIC_NAMES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
)

#: Names a controller might reach for that DO NOT EXIST on 0.25.1.  Reading one
#: must fail loudly, not read as zero.  Also enforced by the always-on floor in
#: `kubegym.core.state.RETIRED_METRIC_NAMES`.
RETIRED_METRIC_NAMES = {
    "vllm:gpu_cache_usage_perc":
        "does not exist on vLLM 0.25.1 (verified absent from all 66 vllm:* families on the "
        "live server); use vllm:kv_cache_usage_perc",
    "vllm:cpu_cache_usage_perc":
        "not exposed by the testbed build; do not depend on it",
}

CONFIG_FIELDS = (
    "kv_tokens_per_replica", "decode_step_time_s", "max_num_seqs", "prefill_tokens_per_s",
    "preemption_policy", "decode_batch_extrapolation_above_8", "output_length_distribution",
)


class LLMReplicaEngine(BaseReplicaEngine):
    """One vLLM engine instance: continuous batching under a KV budget."""

    def __init__(self, rid: int, slot: Slot, *, kv_budget: int, a0: float, a1: float,
                 max_num_seqs: int, prefill_tps: float):
        super().__init__(rid, slot)
        self.kv_budget = int(kv_budget)
        self.a0 = float(a0)
        self.a1 = float(a1)
        self.max_num_seqs = int(max_num_seqs)
        self.prefill_tps = float(prefill_tps)
        self.generation_tokens = 0

    # -- physics -------------------------------------------------------
    def kv_used(self) -> float:
        # `size` is prompt tokens and `progress` is generated tokens; both hold
        # integral values, so float arithmetic here is exact and is materially
        # faster than going through the int-casting property aliases. The KV
        # footprint GROWS as a sequence decodes -- that is the whole point.
        return sum(r.size + r.progress for r in self.running)

    def kv_frac(self) -> float:
        return min(1.0, self.kv_used() / float(self.kv_budget))

    def step_time(self) -> float:
        b = len(self.running)
        return self.a0 + self.a1 * b if b else INF

    def can_admit(self, r: Request) -> bool:
        if not self.accepting or len(self.running) >= self.max_num_seqs:
            return False
        return self.kv_used() + r.size + r.progress <= self.kv_budget

    def try_admit(self, t: float) -> float:
        """Admit as many waiting sequences as KV allows.  Returns prefill seconds."""
        spent = 0.0
        slot = self.slot
        running = self.running
        while slot and self.can_admit(slot[0]):
            r = slot.popleft()
            spent += (r.size + r.progress) / self.prefill_tps
            if r.t_admit is None:
                r.t_admit = t
            running.append(r)
        return spent

    def preempt_if_needed(self, t: float) -> None:
        """Recompute-style preemption: evict most-recently-admitted until KV fits."""
        while len(self.running) > 1 and self.kv_used() > self.kv_budget:
            victim = self.running.pop()          # most recently admitted
            victim.n_preemptions += 1
            self.n_preemptions += 1
            self.slot.push_front(victim)

    # -- ReplicaEngine surface ----------------------------------------
    def service_step(self, t: float) -> List[Request]:
        """One scheduler step: advance every running sequence by one token."""
        finished: List[Request] = []
        running = self.running
        nb = len(running)
        for r in running:
            # `_mark_output` inlined: this runs once per token per running
            # sequence and is the single hottest path in the simulator.
            last = r._t_last_token
            if r.t_first_token is None:
                r.t_first_token = t
                r._t_last_token = t
            else:
                gap = t - (last if last is not None else t)
                if gap > r.max_tbt:
                    r.max_tbt = gap
                r.sum_tbt += gap
                r.n_tbt += 1
                r._t_last_token = t
            r.progress += 1
            if r.progress >= r.demand:
                r.t_done = t
                finished.append(r)
        self.generation_tokens += nb
        self.work_units += nb
        for r in finished:
            running.remove(r)
            self.n_success += 1
        # Reschedule exactly as the source testbed does: a replica that emptied
        # its batch gets next_step_t = t, which the event loop ignores because it
        # only considers replicas with running work.
        self.next_step_t = t + (self.step_time() if self.running else 0.0)
        return finished

    def on_tick(self, t: float) -> None:
        """Admit what KV allows, charge prefill, preempt, set the decode clock."""
        spent = self.try_admit(t)
        if spent > 0:
            # prefill blocks this replica's decode for `spent` seconds
            self.next_step_t = max(self.next_step_t, t) + spent
        self.preempt_if_needed(t)
        if self.running and (self.next_step_t <= t + 1e-9 or self.next_step_t == INF):
            self.next_step_t = t + self.step_time()

    def metrics(self) -> Dict[str, float]:
        return {
            "vllm:num_requests_running": float(len(self.running)),
            "vllm:num_requests_waiting": float(len(self.slot)),
            "vllm:kv_cache_usage_perc": self.kv_frac(),
            "vllm:num_preemptions_total": float(self.n_preemptions),
            "vllm:generation_tokens_total": float(self.generation_tokens),
            "vllm:request_success_total": float(self.n_success),
        }


class LLMServingModel:
    """`ServiceModel` for token-generating LLM inference replicas."""

    name = "llm_serving"

    def __init__(self, cfg: ProvenancedConfig, *, length_model: Optional[LengthModel] = None):
        self.cfg = cfg.subset(CONFIG_FIELDS)
        self.kv_budget = int(cfg.f("kv_tokens_per_replica"))
        st = cfg.f("decode_step_time_s")
        self.a0 = float(st["a0_s"])
        self.a1 = float(st["a1_s_per_seq"])
        self.max_num_seqs = int(cfg.f("max_num_seqs"))
        self.prefill_tps = float(cfg.f("prefill_tokens_per_s"))
        self.length_model = length_model

    # -- ServiceModel surface -----------------------------------------
    def config_fields(self) -> Sequence[str]:
        return CONFIG_FIELDS

    def metric_names(self) -> Sequence[str]:
        return METRIC_NAMES

    def retired_metric_names(self) -> Dict[str, str]:
        return dict(RETIRED_METRIC_NAMES)

    def make_replica(self, rid: int, slot: Slot) -> LLMReplicaEngine:
        return LLMReplicaEngine(rid, slot, kv_budget=self.kv_budget, a0=self.a0, a1=self.a1,
                                max_num_seqs=self.max_num_seqs, prefill_tps=self.prefill_tps)

    def reset(self, seed: int) -> None:
        """No model-owned randomness: output lengths are drawn by the workload
        source at episode build time, which makes controller comparisons paired."""

    def provenance(self) -> Dict[str, Any]:
        p = {
            "service_model": self.name,
            "step_time_model": "t_step(b) = a0 + a1*b, a0=%.8f s, a1=%.8f s/seq" % (self.a0,
                                                                                   self.a1),
            "kv_tokens_per_replica": self.kv_budget,
            "prefill_tokens_per_s": self.prefill_tps,
            "known_infidelities": [
                "prefill_tokens_per_s is a PLACEHOLDER (never measured); all TTFT numbers "
                "are placeholder-derived",
                "decode step time above batch 8 is an unmeasured affine extrapolation; real "
                "engines saturate, so this is OPTIMISTIC about large batches",
                "chunked prefill not modelled: OVERSTATES the inter-token spike on admission, "
                "UNDERSTATES long-prompt TTFT under load",
                "preemption victim order (most-recently-admitted) was not read from vLLM source",
            ],
            "config_provenance": self.cfg.provenance(),
        }
        if self.length_model is not None:
            p["length_model"] = {"calibrated": self.length_model.calibrated,
                                 "label": self.length_model.label()}
        return p


# ---------------------------------------------------------------------------
# Workload source for LLM traces
# ---------------------------------------------------------------------------
class LLMTraceSource(JsonlTraceSource):
    """Trace source for the testbed's JSONL request traces.

    Each line supplies `seq`, `t_arrival`, `prompt_tokens`, `cls` and
    `requested_thinking_mode`.  The OUTPUT LENGTH is not in the trace: it is
    drawn once per request at build time from the length model, so that every
    controller in a comparison faces the identical length realisation for a
    given `(episode, seed)` -- a paired comparison, which matters because the
    length distribution is heavy-tailed.

    `class_predictor_accuracy < 1.0` corrupts the class signal handed to
    controllers with that per-request error probability, standing in for a real
    prompt-length predictor's error.  The true mode stays in
    `attrs["thinking_mode"]` and is visible only to a controller that declares
    `uses_privileged_info`.

    Seeding note: for `episode=0` the length-draw stream is byte-identical to the
    source testbed's `load_trace(seed=...)`, which is what makes the parity test
    comparable.
    """

    def __init__(self, paths: Sequence[str], *, length_model: LengthModel,
                 class_predictor_accuracy: float = 1.0, name: str = "llm_trace",
                 horizon_s: Optional[float] = None):
        super().__init__(paths, name=name, horizon_s=horizon_s)
        self.lm = length_model
        self.class_predictor_accuracy = float(class_predictor_accuracy)
        self._prng: Optional[random.Random] = None

    def begin_build(self, episode: int, seed: int, rng: random.Random) -> None:
        # Second, independent stream for predictor error, seeded exactly as the
        # source testbed's `load_trace` does.
        self._prng = random.Random((int(seed) * 7_654_321) ^ 0xC1A55
                                   ^ (int(episode) * 2_654_435_761))

    def make_request(self, rec: Any, i: int, rng: random.Random) -> Request:
        mode = rec["requested_thinking_mode"]
        cls = rec["cls"]
        n_out = self.lm.sample(cls, mode, rng, 1)[0]
        obs = mode
        if self.class_predictor_accuracy < 1.0 and self._prng is not None:
            if self._prng.random() > self.class_predictor_accuracy:
                obs = "no_think" if mode == "think" else "think"
        return Request(
            seq=int(rec["seq"]),
            t_arrival=float(rec["t_arrival"]),
            demand=float(int(n_out)),
            size=float(int(rec["prompt_tokens"])),
            cls=str(cls),
            attrs={"prompt_id": rec.get("prompt_id", ""), "thinking_mode": mode,
                   "observed_mode": obs},
        )

    def manifest(self) -> Dict[str, Any]:
        m = super().manifest()
        m.update({"length_model_calibrated": self.lm.calibrated,
                  "length_model_label": self.lm.label(),
                  "class_predictor_accuracy": self.class_predictor_accuracy,
                  "output_length_drawn_at": "episode build time (paired across controllers)"})
        return m


def make_llm_serving(cfg: ProvenancedConfig, *, measured_lengths_path: str = "",
                     ) -> LLMServingModel:
    """Build the LLM serving model, loading the length model if one exists.

    With no `measured_lengths.json` the length model reports
    `calibrated: false` and returns the explicitly-labelled placeholder
    lognormals.  It is NOT silently upgraded.
    """
    lm = load_length_model(measured_lengths_path) if measured_lengths_path else \
        load_length_model("measured_lengths.json")
    return LLMServingModel(cfg, length_model=lm)
