# Changes made when preparing this release

Every edit applied to the working code when assembling this anonymous release. No configuration value, calibration flag, trace or reported number was changed.

- baselines/baselines/source_controllers.py: removed 5 classes and their REGISTRY entries. They implement a separate method that is not part of this work, are not used by any KubeGym baseline, and are not reported in the paper. No remaining code referenced them.
- baselines/baselines/source_controllers.py: controller count after removal
- baselines/baselines/source_controllers.py: renumbered after removal
- baselines/baselines/source_controllers.py: removed a pointer to the removed planning method
- baselines/baselines/source_controllers.py: pointed at a report that is not part of this release
- baselines/baselines/source_controllers.py: removed a reference to the removed method
- baselines/baselines/source_controllers.py: neutral wording
- baselines/baselines/source_controllers.py: Oracle docstring: removed comparison against the removed method; corrected the upper-bound claim to match the reported results
- baselines/baselines/source_controllers.py: CONTROLLER_ORDER: dropped the two removed controllers (would KeyError otherwise)
- baselines/baselines/source_controllers.py: __main__ demo: dropped a branch for the removed controllers
- baselines/baselines/source_controllers.py: removed two now-empty section headers; renumbered the Oracle section
- baselines/baselines/source_controllers.py: Oracle is a reference, not an upper bound (consistent with the docstring below)
- regression_fixture.json: re-pinned source_controllers_sha256 (1 occurrence(s)) 4185faabc833... -> 8b0357b05153... to match the edited file. No reported number changes.
- baselines/out/regression_fixture.json: re-pinned source_controllers_sha256 (1 occurrence(s)) 4185faabc833... -> 8b0357b05153... to match the edited file. No reported number changes.
- baselines/out/port_parity.json: re-pinned source_controllers_sha256 (1 occurrence(s)) 4185faabc833... -> 8b0357b05153... to match the edited file. No reported number changes.
- baselines/BASELINES.md: removed a phrase that identified a cited work as the authors' own
- baselines/unverified_baselines.md: removed an internal note about the authors' AI-use log, and rewrote the citation item without identifying phrasing
- docs/unverified_literature.md: collapsed sections A2-A3, which described two not-yet-public works in identifying detail, to a one-paragraph note
- kubegym/configs/llm_serving_l40s.json: output_length_distribution provenance string: removed a reference to the removed method (value, flags and calibration status unchanged)
- core_provenance.json: output_length_distribution provenance string: removed a reference to the removed method (value, flags and calibration status unchanged)
- CALIBRATION.md: regenerated from the edited config by tools/gen_calibration_report.py (--check passes)
- core_provenance.json: neutral wording for the source testbed
- corpus/README.txt: neutral wording
- baselines/README.md: file is no longer verbatim; say so
