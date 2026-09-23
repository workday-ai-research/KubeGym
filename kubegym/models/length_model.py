#!/usr/bin/env python3
"""Per-class output-length model.

STATUS
------
The real per-class output-length distributions MUST be MEASURED on the served
model (Qwen/Qwen3-1.7B on the GPU testbed) in a later phase.  This module is the
interface for that measurement plus an explicitly-labelled PLACEHOLDER used only
so that downstream code (planner, simulator, plots) can be written and
unit-tested before the measurement exists.

Contract
--------
`load_length_model(path)` returns a `LengthModel` whose `.calibrated` flag is:

  True  -> every statistic came from `measured_lengths.json`, i.e. from decoding
           real prompts on the real model.  Numbers may be reported.
  False -> statistics are PLACEHOLDER.  Any number, table or figure derived from
           them MUST carry the string returned by `LengthModel.label()`.

Nothing in this file measures anything.  It contains no empirical claim about
Qwen3-1.7B or any other model.  The placeholder parameters are round numbers
chosen to be *obviously* synthetic (see PLACEHOLDER_PARAMS) so that an
accidentally-unlabelled number is easy to spot.

Expected schema of measured_lengths.json (written by the measurement phase):

{
  "meta": {
    "model": "Qwen/Qwen3-1.7B",
    "engine": "vllm x.y.z",
    "hardware": "a single NVIDIA L40S (46 GB)",
    "sampling": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768},
    "n_prompts_per_cell": 400,
    "corpus_sha256": "<sha256 of corpus.jsonl>",
    "measured_utc": "YYYY-MM-DDTHH:MM:SSZ"
  },
  "cells": {
    "<cls>|<thinking_mode>": {
      "n": 400,
      "samples": [123, 456, ...],          # optional, full sample of output token counts
      "mean": 0.0, "std": 0.0,
      "quantiles": {"0.05": .., "0.25": .., "0.5": .., "0.75": .., "0.95": .., "0.99": ..},
      "truncated_frac": 0.0                # fraction that hit max_tokens
    }
  }
}

`<thinking_mode>` is one of "think" / "no_think" (Qwen3 chat-template switch).
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

PLACEHOLDER_LABEL = ("PLACEHOLDER (not measured; per-class output-length distribution "
                     "pending live-model measurement)")

# Deliberately round, obviously-synthetic lognormal parameters.  These are NOT
# estimates of any model's behaviour and must never be reported as such.
# mu/sigma are on the natural-log scale of output tokens.
PLACEHOLDER_PARAMS: Dict[str, Dict[str, float]] = {
    "short|no_think": {"mu": math.log(100.0), "sigma": 0.5},
    "short|think":    {"mu": math.log(400.0), "sigma": 0.8},
    "long|no_think":  {"mu": math.log(400.0), "sigma": 0.8},
    "long|think":     {"mu": math.log(2000.0), "sigma": 1.0},
}
PLACEHOLDER_CAP = 32768  # generation cap assumed by the placeholder, in tokens


@dataclass
class CellStats:
    """Output-length statistics for one (class, thinking_mode) cell."""
    key: str
    n: int
    mean: float
    std: float
    quantiles: Dict[str, float]
    truncated_frac: float = 0.0
    samples: Optional[List[int]] = None
    source: str = "placeholder"          # "measured" | "placeholder"

    @property
    def measured(self) -> bool:
        return self.source == "measured"


@dataclass
class LengthModel:
    cells: Dict[str, CellStats]
    calibrated: bool
    meta: Dict = field(default_factory=dict)

    # -- labelling -------------------------------------------------------
    def label(self) -> str:
        """String that MUST be attached to any number derived from this model."""
        if self.calibrated:
            return "measured ({})".format(self.meta.get("model", "unknown model"))
        return PLACEHOLDER_LABEL

    def assert_calibrated(self, what: str = "this number") -> None:
        if not self.calibrated:
            raise RuntimeError(
                f"refusing to produce {what}: length model is {PLACEHOLDER_LABEL}. "
                "Run the measurement phase and pass measured_lengths.json."
            )

    # -- queries ---------------------------------------------------------
    def key(self, cls: str, thinking_mode: str) -> str:
        return f"{cls}|{thinking_mode}"

    def stats(self, cls: str, thinking_mode: str) -> CellStats:
        k = self.key(cls, thinking_mode)
        if k not in self.cells:
            raise KeyError(f"no length statistics for cell {k!r}; have {sorted(self.cells)}")
        return self.cells[k]

    def sample(self, cls: str, thinking_mode: str, rng: random.Random, n: int = 1) -> List[int]:
        """Draw output-token counts.  Uses the empirical sample when measured
        (bootstrap), otherwise the labelled placeholder lognormal."""
        st = self.stats(cls, thinking_mode)
        if st.samples:
            return [int(rng.choice(st.samples)) for _ in range(n)]
        p = PLACEHOLDER_PARAMS[self.key(cls, thinking_mode)]
        out = []
        for _ in range(n):
            v = int(round(math.exp(rng.gauss(p["mu"], p["sigma"]))))
            out.append(max(1, min(v, PLACEHOLDER_CAP)))
        return out

    def expected_tokens(self, cls: str, thinking_mode: str) -> float:
        return self.stats(cls, thinking_mode).mean

    def as_dict(self) -> Dict:
        return {
            "calibrated": self.calibrated,
            "label": self.label(),
            "meta": self.meta,
            "cells": {k: {kk: vv for kk, vv in vars(v).items() if kk != "samples"}
                      for k, v in self.cells.items()},
        }


def _placeholder_cells() -> Dict[str, CellStats]:
    cells = {}
    for k, p in PLACEHOLDER_PARAMS.items():
        mu, sg = p["mu"], p["sigma"]
        mean = math.exp(mu + sg * sg / 2)
        var = (math.exp(sg * sg) - 1) * math.exp(2 * mu + sg * sg)
        qs = {}
        for qq in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
            # lognormal quantile via the standard-normal quantile (Acklam-free:
            # use math.erf inverse by bisection to avoid a scipy dependency)
            z = _norm_ppf(qq)
            qs[str(qq)] = math.exp(mu + sg * z)
        cells[k] = CellStats(key=k, n=0, mean=mean, std=math.sqrt(var), quantiles=qs,
                             truncated_frac=0.0, samples=None, source="placeholder")
    return cells


def _norm_ppf(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def load_length_model(path: str | os.PathLike = "measured_lengths.json") -> LengthModel:
    """Load measured statistics if the file exists; otherwise return the
    explicitly-labelled placeholder."""
    if path and os.path.exists(path):
        with open(path) as f:
            blob = json.load(f)
        cells = {}
        for k, c in blob["cells"].items():
            cells[k] = CellStats(
                key=k, n=int(c["n"]), mean=float(c["mean"]), std=float(c["std"]),
                quantiles={str(q): float(v) for q, v in c.get("quantiles", {}).items()},
                truncated_frac=float(c.get("truncated_frac", 0.0)),
                samples=[int(x) for x in c["samples"]] if c.get("samples") else None,
                source="measured")
        return LengthModel(cells=cells, calibrated=True, meta=blob.get("meta", {}))
    return LengthModel(cells=_placeholder_cells(), calibrated=False,
                       meta={"note": PLACEHOLDER_LABEL,
                             "placeholder_params": PLACEHOLDER_PARAMS,
                             "placeholder_cap_tokens": PLACEHOLDER_CAP})


def summarize(lm: LengthModel) -> str:
    lines = [f"length model: {lm.label()}", ""]
    hdr = f"{'cell':<20}{'n':>6}{'mean':>10}{'p50':>10}{'p95':>10}{'p99':>10}"
    lines += [hdr, "-" * len(hdr)]
    for k in sorted(lm.cells):
        c = lm.cells[k]
        lines.append(f"{k:<20}{c.n:>6}{c.mean:>10.1f}"
                     f"{c.quantiles.get('0.5', float('nan')):>10.1f}"
                     f"{c.quantiles.get('0.95', float('nan')):>10.1f}"
                     f"{c.quantiles.get('0.99', float('nan')):>10.1f}")
    if not lm.calibrated:
        lines += ["", "!! " + PLACEHOLDER_LABEL,
                  "!! Do not report these values as findings."]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    lm = load_length_model(sys.argv[1] if len(sys.argv) > 1 else "measured_lengths.json")
    print(summarize(lm))
