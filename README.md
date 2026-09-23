# KubeGym

A reproducible benchmark for reinforcement learning in cloud resource management
(horizontal replica autoscaling). Anonymous code release accompanying an ICLR 2027 submission.

## What is here

| Path | Contents |
|---|---|
| `kubegym/` | The environment: discrete-event core, two service models, Gymnasium API, workload corpus code, tests |
| `kubegym/configs/` | The two physical configurations. Every constant carries its provenance |
| `corpus/` | Workload corpus: manifest for 767 one-hour traces, 30 shipped trace files, Azure selection |
| `baselines/` | Analytic controllers, PPO/DQN training, tuning, evaluation; trained policies in `baselines/out/rl/` |
| `tools/verify.py` | The reproduction harness |
| `CALIBRATION.md` | Which constants are measured and which are placeholders (generated from the configs) |
| `docs/` | API reference, environment cards, workload documentation, validation and parity reports |

## Install

```bash
pip install -e ".[all]"            # Python >= 3.10
export KUBEGYM_CORPUS_DIR=$PWD/corpus
```

## Quick start

```python
import gymnasium as gym
from kubegym.workloads import register_corpus_tasks

register_corpus_tasks()            # adds the 30 corpus task ids
env = gym.make("KubeGym/Corpus-Burst-Test-Delta-v0")
obs, info = env.reset(seed=0, options={"episode": 0, "work_seed": 0})
obs, reward, terminated, truncated, info = env.step(2)   # action 2 = "hold"
print(info["calibrated"])          # False -- see CALIBRATION.md
```

## Reproduce

```bash
python tools/verify.py
```

This runs nine checks and prints PASS, FAIL or SKIP for each. A skip is never counted as a pass.

- `parity` will **SKIP** here. It compares KubeGym against the original simulator it was ported
  from, which is not part of this release. The recorded result is in `parity_results.json` and
  `docs/parity_report.md`: exact on all 14 cases.
- All corpus traces regenerate from their specs. The check compares every SHA-256 against the
  manifest.

Baselines: see `baselines/README.md` and `baselines/BASELINES.md`.

## Read before citing any number

Both shipped configurations report `calibrated=false`, and every result row carries that flag.
These numbers are comparisons inside an uncalibrated simulator. They are **not** empirical
performance claims about real clusters. `CALIBRATION.md` lists exactly which constants are
placeholders and which way each one biases results.

## Data

`corpus/` is derived from the Azure Functions Trace 2019 (Microsoft, CC-BY). See `DATA_LICENSE.md`.
