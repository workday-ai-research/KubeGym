"""Command line entry points for the workload corpus.

    # 1. one-off: select applications from the staged Azure tables and write the
    #    small checksummed extract the corpus is built from
    python -m kubegym.workloads.cli build-extract \
        --invocations azure2019_app_invocations_per_minute.parquet \
        --durations   azure2019_function_durations.parquet \
        --triggers    azure2019_app_trigger_mix.parquet \
        --staging-manifest azure2019_staging_manifest.json \
        --out corpus

    # 2. build the corpus (specs + manifest + a materialised subset)
    python -m kubegym.workloads.cli build --corpus corpus

    # 3. verify: regenerate every trace from its spec and compare checksums,
    #    and re-check every split-disjointness claim
    python -m kubegym.workloads.cli verify --corpus corpus

    # 4. the analyses and figures
    python -m kubegym.workloads.cli sensitivity --corpus corpus
    python -m kubegym.workloads.cli validate    --corpus corpus --out .
    python -m kubegym.workloads.cli figures     --corpus corpus --out figures

    # 5. materialise one trace by id (the regeneration command in the manifest)
    python -m kubegym.workloads.cli regenerate --corpus corpus \
        --trace-id azure-2e862038-d05-m0000-s200013 --out trace.jsonl

Step 1 needs the staged parquet tables; steps 2-5 need only `corpus/`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

EXTRACT_NPZ = "azure2019_selection.npz"
EXTRACT_JSON = "azure2019_selection.json"
MANIFEST = "corpus_manifest.json"
SENSITIVITY = "sensitivity_arrival_placement.json"
VALIDATION = "validation_results.json"


def _extract(corpus: str):
    from .azure2019 import AzureExtract
    return AzureExtract.load(os.path.join(corpus, EXTRACT_NPZ),
                             os.path.join(corpus, EXTRACT_JSON))


def cmd_build_extract(a) -> int:
    from .azure2019 import build_extract, sha256_file
    os.makedirs(a.out, exist_ok=True)
    ex, report = build_extract(a.invocations, a.durations, a.triggers, a.staging_manifest,
                              apps_per_cell=a.apps_per_cell,
                              peak_per_minute_cap=a.peak_per_minute_cap)
    paths = ex.save(os.path.join(a.out, EXTRACT_NPZ), os.path.join(a.out, EXTRACT_JSON))
    print(json.dumps({"extract": paths, "n_apps": len(ex.apps),
                      "selection": report["selection"]}, indent=1))
    return 0


def cmd_build(a) -> int:
    from .azure2019 import sha256_file
    from .corpus import CorpusConfig, build_corpus, write_manifest, verify_splits
    ex = _extract(a.corpus)
    cfg = CorpusConfig(horizon_s=a.horizon_s, segments_per_app_day=a.segments_per_app_day,
                       seeds_per_grid_point=a.seeds_per_grid_point,
                       materialise_per_family_split=a.materialise_per_family_split)
    paths = {f: {"path": f, "sha256": sha256_file(os.path.join(a.corpus, f))}
             for f in (EXTRACT_NPZ, EXTRACT_JSON)}
    man = build_corpus(ex, a.corpus, cfg, extract_paths=paths)
    sha = write_manifest(man, os.path.join(a.corpus, MANIFEST))
    splits = verify_splits(man)
    print(json.dumps({"manifest_sha256": sha, "n_traces": man["n_traces"],
                      "n_requests_total": man["n_requests_total"],
                      "by_family": man["n_traces_by_family"],
                      "by_split": man["n_traces_by_split"],
                      "materialised": len(man["materialised_files"]),
                      "splits_ok": splits["ok"]}, indent=1))
    return 0 if splits["ok"] else 1


def cmd_verify(a) -> int:
    from .corpus import load_manifest, verify_manifest, verify_splits
    ex = _extract(a.corpus)
    man = load_manifest(os.path.join(a.corpus, MANIFEST))
    chk = verify_manifest(man, ex, limit=a.limit, stride=a.stride, corpus_dir=a.corpus)
    splits = verify_splits(man)
    print(json.dumps({"regeneration": chk, "splits": splits}, indent=1))
    return 0 if (chk["ok"] and splits["ok"]) else 1


def cmd_sensitivity(a) -> int:
    from .corpus import load_manifest
    from .sensitivity import run_sensitivity, run_statistics_sensitivity, sensitivity_verdict
    ex = _extract(a.corpus)
    man = load_manifest(os.path.join(a.corpus, MANIFEST))
    sim = run_sensitivity(ex, man["traces"], per_regime=a.per_regime,
                          min_requests=a.min_requests, max_requests=a.max_requests,
                          simulate=True, progress=False)
    stat = run_statistics_sensitivity(ex, man["traces"], min_requests=10, progress_every=0)
    out = {"simulated": sim, "trace_statistics_all_replay_traces": stat,
           "verdict": sensitivity_verdict(sim["summary"])}
    p = os.path.join(a.corpus, SENSITIVITY)
    with open(p, "w", newline="\n") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(out["verdict"])
    return 0


def cmd_validate(a) -> int:
    from .corpus import load_manifest
    from .reports import build_validation, realise_all, validation_markdown
    ex = _extract(a.corpus)
    man = load_manifest(os.path.join(a.corpus, MANIFEST))
    realised = realise_all(man, ex, progress_every=0)
    res = build_validation(man, realised)
    with open(os.path.join(a.corpus, VALIDATION), "w", newline="\n") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
        fh.write("\n")
    sens_path = os.path.join(a.corpus, SENSITIVITY)
    sens = json.load(open(sens_path)) if os.path.exists(sens_path) else None
    if sens is not None:
        md = validation_markdown(res, sens, man)
        with open(os.path.join(a.out, "validation_report.md"), "w", newline="\n") as fh:
            fh.write(md)
    print(json.dumps({"validation_results": os.path.join(a.corpus, VALIDATION),
                      "report_written": sens is not None}, indent=1))
    return 0


def cmd_figures(a) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    from .corpus import load_manifest
    from .reports import realise_all
    from . import diagnostics as dg
    ex = _extract(a.corpus)
    man = load_manifest(os.path.join(a.corpus, MANIFEST))
    realised = realise_all(man, ex, progress_every=0)
    res = json.load(open(os.path.join(a.corpus, VALIDATION)))
    sens = json.load(open(os.path.join(a.corpus, SENSITIVITY)))
    os.makedirs(a.out, exist_ok=True)
    names = {}
    for name, fig in (("corpus_families.png", dg.figure_families(realised)),
                      ("corpus_validation.png", dg.figure_validation(realised, res)),
                      ("corpus_sensitivity.png", dg.figure_sensitivity(sens)),
                      ("corpus_population.png", dg.figure_population(ex, man))):
        p = os.path.join(a.out, name)
        fig.savefig(p, dpi=300)
        names[name] = p
    print(json.dumps(names, indent=1))
    return 0


def cmd_regenerate(a) -> int:
    from .corpus import canonical_lines, load_manifest, realise_entry, trace_digest
    ex = _extract(a.corpus)
    man = load_manifest(os.path.join(a.corpus, MANIFEST))
    hits = [e for e in man["traces"] if e["trace_id"] == a.trace_id]
    if not hits:
        print(f"no trace {a.trace_id!r} in the manifest", file=sys.stderr)
        return 2
    e = hits[0]
    out = realise_entry(e, ex)
    got = trace_digest(out["t_arrival"], out["service_time_s"])
    with open(a.out, "wb") as fh:
        fh.write(canonical_lines(out["t_arrival"], out["service_time_s"]))
    ok = got == e["sha256"]
    print(json.dumps({"trace_id": a.trace_id, "path": a.out,
                      "n_requests": int(out["t_arrival"].size),
                      "sha256": got, "manifest_sha256": e["sha256"], "match": ok}, indent=1))
    return 0 if ok else 1


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="kubegym.workloads.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-extract", help="select applications and write the extract")
    p.add_argument("--invocations", required=True)
    p.add_argument("--durations", required=True)
    p.add_argument("--triggers", required=True)
    p.add_argument("--staging-manifest", required=True)
    p.add_argument("--out", default="corpus")
    p.add_argument("--apps-per-cell", type=int, default=3)
    p.add_argument("--peak-per-minute-cap", type=float, default=4800.0)
    p.set_defaults(fn=cmd_build_extract)

    p = sub.add_parser("build", help="build the corpus from the extract")
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--horizon-s", type=float, default=3600.0)
    p.add_argument("--segments-per-app-day", type=int, default=2)
    p.add_argument("--seeds-per-grid-point", type=int, default=2)
    p.add_argument("--materialise-per-family-split", type=int, default=2)
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("verify", help="regenerate every trace and check checksums and splits")
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("sensitivity", help="arrival-placement sensitivity study")
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--per-regime", type=int, default=10)
    p.add_argument("--min-requests", type=int, default=50)
    p.add_argument("--max-requests", type=int, default=40000)
    p.set_defaults(fn=cmd_sensitivity)

    p = sub.add_parser("validate", help="distributional validation + report")
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--out", default=".")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("figures", help="diagnostics figures")
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--out", default="figures")
    p.set_defaults(fn=cmd_figures)

    p = sub.add_parser("regenerate", help="materialise one trace by id")
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--trace-id", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_regenerate)

    a = ap.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
