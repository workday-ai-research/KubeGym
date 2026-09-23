# KubeGym reference baselines — reproduction

Contents:

    baselines/            controller ports, tuner, trainer, evaluator, aggregator, figures
    baselines/source_controllers.py   the source testbed's controller file (two unrelated controllers removed)
                                      (sha256 4185faabc833ba2759dd9f621c91e85638cb9314a2e4e5830295df8f814fb612)
    out/tuned_<family>.json           every tuning trial, the ledger, the selected parameters
    out/port_parity.json              shadowed-decision comparison + oracle property checks
    out/rl/train_<algo>_<family>.json hyperparameter search, per-seed curves, model checksums
    out/rl/<algo>_<family>_seed<k>.zip  trained policies
    out/regression_fixture.json       reference numbers with tolerances
    BASELINES.md                      methods, budget, protocol, honest interpretation
    unverified_baselines.md           what is NOT verified
    ai_use_log_baselines.md           AI-use record for the paper's disclosure

## Reproducing

    pip install -e /path/to/kubegym[all]        # or put the package root on PYTHONPATH
    export KUBEGYM_CORPUS_DIR=/path/to/corpus   # the extracted corpus directory
    export PYTHONPATH=$PWD:$PWD/baselines

    python -m baselines.parity_port                 # port verification (must report 0 disagreements)
    python -m baselines.tune                        # dev-only sweeps, all families
    python -m baselines.train_rl ppo                # ~0.6 h per family, CPU
    python -m baselines.train_rl dqn
    python -m baselines.evaluate                    # test split, once per method
    python -m baselines.aggregate                   # reference_results.csv, results_summary.csv, pairwise
    python -m baselines.fixture                     # rebuild the fixture
    python -c "from baselines.fixture import check; print(check())"   # verify against it
    python -m baselines.llm_side                    # work-seed variance side study

Figures are produced by `baselines/figures.py` (see BASELINES.md for the calls).

## Two things to know before running

1. `baselines/engine_guard.py` installs a one-line workaround for a
   non-terminating event loop in the shipped `request_service` engine. Without it
   the `azure_replay` evaluation hangs with no output. Its inertness on everything
   already measured is verified by `engine_guard.verify_inert()`. See BASELINES.md
   section 6.
2. The test split is opened exactly once per method, in `evaluate.py`.
   `runner.assert_not_test` raises if any other phase reaches for it. Do not
   remove that guard to "just check something" on test.
