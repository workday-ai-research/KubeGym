"""Deterministic RNG substreams for the workload corpus.

The scheme is inherited from the prior project's generator
(`workload.py::_substream`): every stochastic component of a trace draws from its
own SHA256-derived substream of the trace seed, so

  * adding a new stochastic component never shifts an existing one, and
  * changing one knob (say the burst magnitude) cannot perturb an unrelated
    stream (say the service-time draws).

Two additions here:

1. **Named parts, not a single name.**  `substream(seed, "arrival", app_hash,
   day, segment)` lets a corpus of hundreds of traces give every trace its own
   independent streams while remaining a pure function of the spec.

2. **A numpy generator with a version-stable draw surface.**  The reconstruction
   draws millions of variates, which is impractical in pure Python, so
   `np_substream` returns a `numpy.random.Generator(PCG64(...))` seeded from the
   same digest.  Everything downstream draws **only** `Generator.random(n)`
   (uniform doubles) and derives normals / geometrics / categoricals / Poissons
   from those uniforms by explicit inverse transform in this module.  That
   restriction is deliberate: it keeps the byte-level output independent of which
   numpy version implements `standard_normal`, `geometric`, `choice` or
   `poisson`, so a corpus checksum recorded today still verifies on a future
   numpy.  Do not call any other `Generator` method on a substream from here.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Sequence

import numpy as np

#: Every stream name used anywhere in this package.  Listed so the manifest can
#: record them and so a reviewer can see that the set is closed.
STREAM_NAMES = (
    "arrival",          # within-minute arrival placement (replay) / thinning (synthetic)
    "rate_process",     # the stochastic rate path of the `variable` family
    "func_attrib",      # which function of an application an invocation belongs to
    "service",          # service-time draws
    "mix",              # class-label draws (kept for parity with the prior generator)
    "segment",          # deterministic segment selection
)


def digest(seed: int, *parts: Any) -> bytes:
    """SHA256 of the canonical `seed|part|part|...` string."""
    payload = "|".join([str(int(seed))] + [str(p) for p in parts])
    return hashlib.sha256(payload.encode("utf-8")).digest()


def substream(seed: int, *parts: Any) -> random.Random:
    """A `random.Random` for one named component of one trace."""
    return random.Random(int.from_bytes(digest(seed, *parts)[:16], "big"))


def np_substream(seed: int, *parts: Any) -> np.random.Generator:
    """A numpy `Generator` for one named component of one trace.

    Only `.random(n)` may be called on the result -- see the module docstring.
    """
    return np.random.Generator(np.random.PCG64(
        int.from_bytes(digest(seed, *parts), "big")))


# ---------------------------------------------------------------------------
# Version-stable transforms of uniform doubles
# ---------------------------------------------------------------------------
# `scipy.special.ndtri` is a fixed mathematical function, unlike a library's
# sampling *algorithm*, so an inverse-transform normal is reproducible across
# numpy and scipy versions.  Its relative accuracy (~1e-15) is far below any
# effect a workload trace can express.

def normal_from_uniform(u: np.ndarray) -> np.ndarray:
    """Standard normals by inverse CDF of uniforms in (0, 1)."""
    from scipy.special import ndtri
    u = np.clip(np.asarray(u, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    return np.asarray(ndtri(u), dtype=np.float64)


def geometric_from_uniform(u: np.ndarray, mean: float) -> np.ndarray:
    """Geometric variates on {1, 2, ...} with the given mean, by inverse CDF.

    `p = 1/mean`; `k = ceil(log1p(-u) / log1p(-p))`, clamped to >= 1.
    """
    mean = float(mean)
    if mean < 1.0:
        raise ValueError(f"geometric mean must be >= 1, got {mean}")
    u = np.asarray(u, dtype=np.float64)
    if mean == 1.0:
        return np.ones(u.shape, dtype=np.int64)
    p = 1.0 / mean
    u = np.clip(u, 0.0, 1.0 - 1e-15)
    k = np.ceil(np.log1p(-u) / math.log1p(-p))
    return np.maximum(1, k.astype(np.int64))


def categorical_from_uniform(u: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    """Categorical indices by inverse CDF of uniforms, given unnormalised weights."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a non-empty 1-D sequence")
    if not np.all(w >= 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    cdf = np.cumsum(w) / w.sum()
    cdf[-1] = 1.0
    return np.searchsorted(cdf, np.asarray(u, dtype=np.float64),
                           side="right").clip(0, w.size - 1)


def poisson_from_uniform(u: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Poisson variates by inverse CDF (cumulative search).

    Vectorised over `u` and `lam`.  Used only for the `poisson` arrival-placement
    arm, where per-minute counts are re-drawn rather than preserved, and only for
    the modest `lam` a per-minute serverless count takes; the loop is bounded by
    `max(lam) + 12*sqrt(max(lam)) + 20` terms.
    """
    u = np.asarray(u, dtype=np.float64)
    lam = np.broadcast_to(np.asarray(lam, dtype=np.float64), u.shape).astype(np.float64)
    out = np.zeros(u.shape, dtype=np.int64)
    if u.size == 0:
        return out
    lmax = float(lam.max())
    kmax = int(lmax + 12.0 * math.sqrt(max(lmax, 1.0)) + 20.0)
    term = np.exp(-lam)              # P(X = 0)
    cdf = term.copy()
    active = u > cdf
    for k in range(1, kmax + 1):
        if not active.any():
            break
        term = term * lam / k
        cdf = cdf + term
        out[active] += 1
        active = u > cdf
    return out
