"""Provenance-tracked physical configuration.

WHY THIS EXISTS
---------------
A benchmark whose physics are partly measured and partly guessed is only
scientifically usable if the boundary is machine-readable.  This module
generalises the `cluster-config-2` scheme used by the vLLM testbed this package
was ported from: every physical constant is a *field* that carries its own value
**and its own provenance**, and the run-level `calibrated` flag is a pure
function of those per-field flags.

Per-field keys
--------------
    value               the constant (scalar, string, or nested dict)
    calibrated          bool -- True ONLY if the value came from a measurement
    source              free text: how it was obtained, with n / hardware / fit
    required_for_claims bool -- if True, a placeholder here poisons the whole run
    weak_evidence       optional bool -- measured, but from too few observations
    flagged_optimistic  optional bool -- known to bias results in a favourable
                        direction; the `source` string must say which direction

THE CLAIMS GATE
---------------
`ProvenancedConfig.calibrated` is the AND over every field with
`required_for_claims: True`.  One placeholder makes the entire run
non-reportable.  That is deliberate and must not be relaxed by downstream code:
`stamp()` attaches the flag to every result row so a non-calibrated number
cannot travel without its label.

NO SILENT UPGRADES
------------------
`override()` cannot raise `calibrated` from False to True unless the caller
passes an explicit `attest` string describing the new measurement.  Overriding a
*measured* field's value without `attest` drops it to `calibrated: False` -- the
old measurement no longer describes the value in use.  Every override is
recorded in `provenance()["overrides"]`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field as _dc_field
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = "kubegym-config-1"

#: Schemas this loader accepts.  `cluster-config-2` is the vLLM testbed schema
#: this package was ported from; it is read verbatim so ported configs keep
#: their provenance strings byte-for-byte.
ACCEPTED_SCHEMAS = ("kubegym-config-1", "cluster-config-2")

_FLAG_KEYS = ("calibrated", "required_for_claims", "weak_evidence", "flagged_optimistic")


class ProvenanceError(ValueError):
    """Raised when a config violates the provenance contract."""


@dataclass
class Field:
    """One physical constant plus its provenance."""

    name: str
    value: Any
    calibrated: bool = False
    source: str = ""
    required_for_claims: bool = False
    weak_evidence: bool = False
    flagged_optimistic: bool = False
    extra: Dict[str, Any] = _dc_field(default_factory=dict)

    @property
    def placeholder(self) -> bool:
        return not self.calibrated

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "value": self.value,
            "calibrated": self.calibrated,
            "required_for_claims": self.required_for_claims,
            "source": self.source,
        }
        if self.weak_evidence:
            d["weak_evidence"] = True
        if self.flagged_optimistic:
            d["flagged_optimistic"] = True
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, name: str, raw: Dict[str, Any]) -> "Field":
        if not isinstance(raw, dict) or "value" not in raw:
            raise ProvenanceError(
                f"field {name!r} must be an object with a 'value' key; got "
                f"{type(raw).__name__}. Bare constants are rejected on purpose: every "
                "constant carries provenance.")
        if not str(raw.get("source", "")).strip():
            raise ProvenanceError(
                f"field {name!r} has no 'source'. A constant without a stated origin cannot "
                "be distinguished from a guess, so it is refused.")
        extra = {k: v for k, v in raw.items()
                 if k not in _FLAG_KEYS and k not in ("value", "source")}
        return cls(
            name=name,
            value=raw["value"],
            calibrated=bool(raw.get("calibrated", False)),
            source=str(raw["source"]),
            required_for_claims=bool(raw.get("required_for_claims", False)),
            weak_evidence=bool(raw.get("weak_evidence", False)),
            flagged_optimistic=bool(raw.get("flagged_optimistic", False)),
            extra=extra,
        )


class ProvenancedConfig:
    """A bundle of `Field`s with a claims gate.

    Construct with `ProvenancedConfig.load(path)` or
    `ProvenancedConfig.from_dict(raw, path=...)`.

    Read values with `cfg["name"]` or `cfg.f("name")`.  `f` is kept for drop-in
    compatibility with the testbed's `ClusterConfig`, so controllers written
    against that class run unmodified against this one.
    """

    def __init__(self, fields: Dict[str, Field], *, path: str = "<memory>",
                 schema: str = SCHEMA_VERSION, calibration_note: str = "",
                 meta: Optional[Dict[str, Any]] = None,
                 overrides: Optional[List[Dict[str, Any]]] = None):
        self._fields = dict(fields)
        self.path = path
        self.schema = schema
        self.calibration_note = calibration_note
        self.meta = dict(meta or {})
        self._overrides: List[Dict[str, Any]] = list(overrides or [])

    # -- construction ---------------------------------------------------
    @classmethod
    def from_dict(cls, raw: Dict[str, Any], *, path: str = "<memory>") -> "ProvenancedConfig":
        schema = str(raw.get("schema", ""))
        if schema not in ACCEPTED_SCHEMAS:
            raise ProvenanceError(
                f"{path}: schema {schema!r} not accepted; expected one of {ACCEPTED_SCHEMAS}")
        if "fields" not in raw or not isinstance(raw["fields"], dict):
            raise ProvenanceError(f"{path}: expected a 'fields' object")
        fields = {k: Field.from_dict(k, v) for k, v in raw["fields"].items()}
        meta = {k: v for k, v in raw.items()
                if k not in ("fields", "schema", "calibration_note", "calibrated",
                             "uncalibrated_required_fields", "overrides")}
        return cls(fields, path=path, schema=schema,
                   calibration_note=str(raw.get("calibration_note", "")), meta=meta)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ProvenancedConfig":
        with open(path) as fh:
            raw = json.load(fh)
        return cls.from_dict(raw, path=str(path))

    # -- reads ----------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self._fields

    def __getitem__(self, name: str) -> Any:
        return self.field(name).value

    def field(self, name: str) -> Field:
        try:
            return self._fields[name]
        except KeyError:
            raise KeyError(f"no config field {name!r}; have {sorted(self._fields)}") from None

    def f(self, name: str) -> Any:
        """Value accessor, name-compatible with the testbed's `ClusterConfig.f`."""
        return self.field(name).value

    def get(self, name: str, default: Any = None) -> Any:
        return self._fields[name].value if name in self._fields else default

    def is_calibrated(self, name: str) -> bool:
        return self.field(name).calibrated

    def source(self, name: str) -> str:
        return self.field(name).source

    def names(self) -> List[str]:
        return sorted(self._fields)

    def fields(self) -> Dict[str, Field]:
        return dict(self._fields)

    # -- the claims gate ------------------------------------------------
    @property
    def calibrated(self) -> bool:
        """AND over every field marked `required_for_claims`.

        One placeholder in a required field makes the whole run non-reportable.
        """
        return all(fl.calibrated for fl in self._fields.values() if fl.required_for_claims)

    @property
    def uncalibrated_required(self) -> List[str]:
        return sorted(k for k, fl in self._fields.items()
                      if fl.required_for_claims and not fl.calibrated)

    def uncalibrated_required_detail(self) -> List[Dict[str, Any]]:
        return [{"field": k, "value": self._fields[k].value, "source": self._fields[k].source,
                 "flagged_optimistic": self._fields[k].flagged_optimistic}
                for k in self.uncalibrated_required]

    def require_calibrated(self, what: str = "this number") -> None:
        """Raise unless every required-for-claims field is measured."""
        if not self.calibrated:
            raise RuntimeError(
                f"refusing to produce {what}: uncalibrated required fields "
                f"{self.uncalibrated_required}. This config is a modelling scaffold; "
                "measure those constants before reporting.")

    def provenance(self) -> Dict[str, Any]:
        """The block every output file and result row must embed."""
        return {
            "schema": self.schema,
            "calibrated": self.calibrated,
            "uncalibrated_required_fields": self.uncalibrated_required,
            "calibration_note": self.calibration_note,
            "config_path": os.path.basename(str(self.path)),
            "measured_fields": sorted(k for k, v in self._fields.items() if v.calibrated),
            "placeholder_fields": sorted(k for k, v in self._fields.items() if not v.calibrated),
            "weak_evidence_fields": sorted(k for k, v in self._fields.items() if v.weak_evidence),
            "flagged_optimistic_fields": sorted(k for k, v in self._fields.items()
                                                if v.flagged_optimistic),
            "overrides": list(self._overrides),
        }

    def stamp(self, row: Dict[str, Any], *, prefix: str = "") -> Dict[str, Any]:
        """Attach the claims gate to a result row, in place.

        Every downstream result row (CSV line, JSON record, RL episode summary)
        must pass through this.  `calibrated=False` rows are not reportable.
        """
        p = self.provenance()
        row[prefix + "calibrated"] = p["calibrated"]
        row[prefix + "uncalibrated_required_fields"] = ";".join(p["uncalibrated_required_fields"])
        row[prefix + "config"] = p["config_path"]
        if p["flagged_optimistic_fields"]:
            row[prefix + "flagged_optimistic_fields"] = ";".join(p["flagged_optimistic_fields"])
        if p["overrides"]:
            row[prefix + "config_overridden"] = True
        return row

    # -- controlled mutation -------------------------------------------
    def override(self, name: str, value: Any, *, source: str,
                 attest: Optional[str] = None,
                 required_for_claims: Optional[bool] = None) -> "ProvenancedConfig":
        """Return a COPY with one field's value replaced.

        `calibrated` becomes True only when `attest` is supplied, and then the
        attestation text is appended to the source string.  Without `attest` the
        field is marked `calibrated: False` whatever it was before.  There is no
        code path from placeholder to measured that does not pass through
        `attest`.
        """
        old = self.field(name)
        attested = attest is not None
        new = Field(
            name=name,
            value=value,
            calibrated=bool(attested),
            source=(source if not attested
                    else f"{source} | ATTESTED MEASUREMENT: {attest}"),
            required_for_claims=(old.required_for_claims if required_for_claims is None
                                 else bool(required_for_claims)),
            weak_evidence=old.weak_evidence and attested,
            flagged_optimistic=old.flagged_optimistic,
            extra=dict(old.extra),
        )
        fields = dict(self._fields)
        fields[name] = new
        rec = {"field": name, "old_value": old.value, "new_value": value,
               "old_calibrated": old.calibrated, "new_calibrated": new.calibrated,
               "source": new.source, "attested": attested}
        return ProvenancedConfig(fields, path=self.path, schema=self.schema,
                                 calibration_note=self.calibration_note, meta=self.meta,
                                 overrides=self._overrides + [rec])

    def overridden(self, **values: Any) -> "ProvenancedConfig":
        """Convenience for experiment knobs: every override is marked uncalibrated.

        Use for sweeps and sensitivity analysis.  The resulting config reports
        `calibrated=False` for any required field it touched and records each
        override in `provenance()["overrides"]`.
        """
        cfg = self
        for k, v in values.items():
            cfg = cfg.override(k, v, source=(
                f"RUNTIME OVERRIDE (sweep / experiment knob); previous value {cfg.f(k)!r}. "
                "Not a measurement."))
        return cfg

    def subset(self, names: Iterable[str], *, strict: bool = True) -> "ProvenancedConfig":
        """The sub-config a component declares it consumes.

        A service model declares its field names via `config_fields()`; `subset`
        is how the harness checks the config supplies them and how a
        per-component provenance block is produced.
        """
        want = list(names)
        missing = [n for n in want if n not in self._fields]
        if missing and strict:
            raise KeyError(f"config {os.path.basename(str(self.path))} is missing required "
                           f"fields {missing}; have {sorted(self._fields)}")
        return ProvenancedConfig({n: self._fields[n] for n in want if n in self._fields},
                                 path=self.path, schema=self.schema,
                                 calibration_note=self.calibration_note, meta=self.meta,
                                 overrides=self._overrides)

    def merge(self, other: "ProvenancedConfig", *, prefix: str = "") -> "ProvenancedConfig":
        """Union of two configs; a duplicate name is an error, not a silent win."""
        fields = dict(self._fields)
        for k, v in other._fields.items():
            key = prefix + k
            if key in fields:
                raise ProvenanceError(
                    f"duplicate config field {key!r} while merging {other.path} into "
                    f"{self.path}; provenance would be ambiguous")
            fields[key] = v
        note = "; ".join(x for x in (self.calibration_note, other.calibration_note) if x)
        return ProvenancedConfig(fields, path=self.path, schema=self.schema,
                                 calibration_note=note, meta=self.meta,
                                 overrides=self._overrides + other._overrides)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.meta)
        d.update({
            "schema": self.schema,
            "calibrated": self.calibrated,
            "uncalibrated_required_fields": self.uncalibrated_required,
            "calibration_note": self.calibration_note,
            "fields": {k: self._fields[k].to_dict() for k in sorted(self._fields)},
        })
        if self._overrides:
            d["overrides"] = list(self._overrides)
        return d

    def save(self, path: str | os.PathLike) -> str:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return str(path)

    def report(self) -> str:
        """Human-readable calibration table."""
        lines = [f"config: {self.path}  schema={self.schema}",
                 f"CLAIMS GATE: calibrated={self.calibrated}"]
        if self.uncalibrated_required:
            lines.append(f"  blocked by: {', '.join(self.uncalibrated_required)}")
        lines += ["", f"{'field':<40}{'calib':>7}{'reqd':>6}  flags", "-" * 74]
        for k in sorted(self._fields):
            fl = self._fields[k]
            flags = ",".join(x for x, on in (("weak_evidence", fl.weak_evidence),
                                             ("flagged_optimistic", fl.flagged_optimistic)) if on)
            lines.append(f"{k:<40}{str(fl.calibrated):>7}{str(fl.required_for_claims):>6}  {flags}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"ProvenancedConfig({os.path.basename(str(self.path))!r}, "
                f"n_fields={len(self._fields)}, calibrated={self.calibrated})")


def load_config(path: str | os.PathLike) -> ProvenancedConfig:
    return ProvenancedConfig.load(path)
