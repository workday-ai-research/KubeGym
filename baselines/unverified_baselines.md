# Unverified claims and open items — baselines

Everything in this file is something a reader should not take as confirmed. It is
kept separate from `BASELINES.md` so the write-up can be read as verified and
this can be read as the register of what is not.

## A. Not verified against a primary source in this track

1. **arXiv:2605.26418 and arXiv:2608.07303**, cited in `BASELINES.md`, were checked
   against their indexed records (title, authors, year and abstract). The findings
   attributed to them are paraphrases of those abstracts.
2. **The TokenScale reference (arXiv 2512.03416)** appears in
   `gym_controllers.LeadingIndicatorGym` as the source of the structural idea,
   carried over from the prior project's controller file. The identifier was not
   verified, and nothing here is a measurement of that method — the controller is
   an adaptation driving on arrival rate, not tokens.
3. **The Kubernetes HPA documentation string** (`HPA_SEMANTICS_SOURCE`, page last
   modified 2026-03-15, retrieved 2026-08-12) was copied verbatim from the source
   implementation and cross-read against the primary-source extract shipped with
   the project. The live documentation page was **not** re-fetched in this track,
   so the "last modified" date is as recorded by the earlier phase.
4. **ADAPT / SageServe** are named in `ForecastThresholdGym` as the family the
   forecast-then-provision controller belongs to, inherited from the source file.
   No citation was verified and the implementation is a reimplementation, not the
   original authors' code.

## B. Claims that are true only inside this simulator

5. **Every result row has `calibrated=false`**, with
   `replica_cold_start_s`, `rs_max_concurrency`, `rs_service_time_distribution`
   and `slo` listed as unmeasured claims-relevant fields. No number in
   `reference_results.csv`, `results_summary.csv`, `regression_fixture.json` or any
   figure is an empirical performance claim about a real autoscaler.
6. **The azure_replay SLO axis is dominated by a placeholder.** The irreducible
   floor is 0.8279 there and the best deployable controller attains 0.8040
   (the privileged oracle 0.8223). Decomposing the shortfall from perfect
   attainment: total 0.1960, of which 0.1721 is imposed by the floor and only
   0.0239 is controllable — so **87.8 % of the shortfall is set by
   `rs_service_time_distribution`**, a placeholder constant, and the best
   controller reaches 97.1 % of the attainable ceiling. Any SLO-based ranking on
   that family is largely a statement about the placeholder.
7. **The `llm_serving` side study uses shipped or default controller parameters,
   not parameters tuned on `llm_serving` dev.** Its numbers are a variance
   measurement only. The mean costs there must not be read as a controller
   comparison.

## C. Scope limits that could be mistaken for results

8. **PPO and DQN were not trained on `llm_serving`.** The reason is wall-clock
   under budget parity (~4.3 h per algorithm/family at the same 1.2 M-step cap),
   not an experimental finding. Nothing follows about learned control there.
10. **Only PPO and DQN were trained.** A2C, SAC, TD3 and DDPG — which the prior
    study covered — were not run here. "RL does not win" in `BASELINES.md` means
    these two algorithms at this budget.
11. **The budget is 1.2 M environment steps per (method, family).** That is small
    by RL standards. The finding is about a matched and enforced cap, not about
    the asymptotic capability of either algorithm.
12. **`leading` is an adaptation, not a port.** Its parity check compares it
    against a reference implementation of the same adapted algorithm, which
    isolates the 4-lag observation restriction. It is **not** verified against the
    source controller's token-driven behaviour, because `request_service` has no
    token stream.
13. **`hpa` with `metric="kv"` is parity-checked only on `llm_serving`**, because
    the source reads a metric name that does not exist under `request_service`.
    The `kv` variant's port is unverified on the corpus tasks, where it was never
    the dev-selected configuration for any family.

## D. Workarounds whose side effects are argued, not proved in general

14. **`engine_guard`'s 1 ns event-time floor.** Its inertness is verified
    empirically and narrowly: all 30 tuned dev objectives, measured before the
    guard existed, reproduce bit-identically (max abs diff 0.0). That is evidence
    for the configurations and episodes checked, not a proof that the floor can
    never bind observably on some other trace. The correct fix is a relative
    completion tolerance in the core engine.
15. **`InfoEpisodeKeyGuard`** renames `info["episode"]`. It was verified only to
    the extent that training runs complete and evaluation reads the corpus episode
    index from the environment rather than from `info`. No systematic check that
    no other SB3 or Gymnasium code path depends on the original key.
16. **The HPA target-grid re-ranging is a judgement call.** The argument that a
    per-replica concurrency target of 30 is unreachable at
    `rs_max_concurrency = 4` is arithmetic, but the specific replacement grid
    (0.5–4.0) was chosen by the author of this track, not derived from a source.

## E. Statistical caveats

17. **The within-noise rule is one specific rule**: 95 % percentile bootstrap CI
    over episodes covering zero, or a point difference below the relevant
    across-seed SD. Both thresholds are conventions. A different rule would move
    some of the 47 flagged pairs.
18. **Five training seeds** is enough to see that seed dispersion is comparable to
    several between-method gaps, and not enough to estimate it precisely. The
    across-seed SDs reported (0.00–19.79 on cost) are themselves noisy at n=5.
19. **No multiple-comparison correction** is applied across the 140 pairwise
    tests. The within-noise flags should be read as descriptive, not as
    family-wise-error-controlled inference.
20. **The `azure_replay` per-regime table has 18–24 episodes per regime** and its
    heavy regimes are the ones carrying the cost, so those cell means rest on a
    few dozen traces each.
