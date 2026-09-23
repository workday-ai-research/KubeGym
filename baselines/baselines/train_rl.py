"""Train PPO and DQN on the corpus training split under an enforced budget.

BUDGET, AND WHY IT IS THE SAME CURRENCY AS THE ANALYTIC SWEEP
-------------------------------------------------------------
The budget is **environment steps taken on the train and dev splits**, capped at
`tune.CAP_ENV_STEPS` per (method, family) -- the same integer the analytic
controllers are capped at, charged through the same `runner.BudgetLedger`.  Every
step of hyperparameter search, of training, of learning-curve evaluation and of
model selection is charged.  Test-split steps are not charged, because they are
not search: they are the single measurement at the end.

This is the one accounting choice that makes "matched budget" checkable rather
than claimed.  It is deliberately *generous to RL relative to a per-trial
budget*: the analytic methods spend a fraction of the cap because their grids are
small, whereas both learners spend most of it.  The spend per method is reported
in `BASELINES.md` and in `rl_training.json`, so a reader can see who got more
search compute rather than taking a parity claim on trust.

ALLOCATION (identical for PPO and DQN, per family)
--------------------------------------------------
    search : N_SEARCH_CONFIGS configs x SEARCH_STEPS training steps, one seed,
             each evaluated once on the dev tuning episodes
    final  : the dev-selected config trained on N_SEEDS seeds x FINAL_STEPS,
             with a dev evaluation every EVAL_EVERY steps for the learning curve,
             and a final full dev evaluation for best-checkpoint selection

The two algorithms get the same number of configs, the same step counts, the same
seeds, the same policy architecture (`[64, 64]` MLP, `tanh`) and the same
observation and action spaces.  The hyperparameter grid has the same *shape* for
both -- {two learning rates} x {absolute mode} plus {delta mode at the lower
rate} -- with algorithm-appropriate rate values, because a shared numeric rate
would be a handicap for one of them rather than a fairness measure.

The action mode is inside the search grid on purpose.  `absolute` lets a policy
request any target in one step, matching what the analytic controllers can do;
`delta` is rate-limited to +-2 replicas per interval but has a much smaller
action space (5 vs 20).  Which is easier to learn is an empirical question, so it
is answered on dev rather than assumed, and the answer is reported.

WHAT IS NOT DONE, AND WHY
-------------------------
No reward normalisation, no observation normalisation wrapper, no frame stacking
beyond the task's own 4-lag observation.  The observation is already scaled by
a-priori constants (`kubegym.gym.obs`) and the per-step reward is O(1) on these
tasks, so a normaliser would add a moving statistic between the policy and the
benchmark's objective without changing its optimum -- and it would make the
learner's inputs a function of episode statistics the analytic controllers are
forbidden from using.  This is a fairness choice, and it is recorded as one.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .runner import (BudgetLedger, FAMILIES, assert_not_test, load_manifest,
                     make_env, objective, rollout, task_id)
from .tune import CAP_ENV_STEPS, dev_tuning_episodes

N_SEEDS = 5
N_SEARCH_CONFIGS = 3
SEARCH_STEPS = 100_000
FINAL_STEPS = 150_000
EVAL_EVERY = 15_000
#: dev episodes used for the learning curve (a subset, to keep curve cost small)
N_DEV_CURVE = 5
N_ENVS = 4
NET_ARCH = [64, 64]


def hp_grid(algo: str) -> List[Dict[str, Any]]:
    """The search grid.  Same shape for both algorithms."""
    if algo == "ppo":
        lo, hi = 3e-4, 1e-3
    elif algo == "dqn":
        lo, hi = 1e-4, 5e-4
    else:
        raise KeyError(algo)
    return [
        {"action_mode": "absolute", "learning_rate": lo},
        {"action_mode": "absolute", "learning_rate": hi},
        {"action_mode": "delta", "learning_rate": lo},
    ]


def _algo_kwargs(algo: str, hp: Dict[str, Any], seed: int) -> Dict[str, Any]:
    common = dict(policy="MlpPolicy", learning_rate=float(hp["learning_rate"]),
                  gamma=0.99, seed=int(seed), verbose=0, device="cpu",
                  policy_kwargs=dict(net_arch=list(NET_ARCH)))
    if algo == "ppo":
        return dict(common, n_steps=256, batch_size=256, n_epochs=10,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
                    max_grad_norm=0.5)
    return dict(common, buffer_size=100_000, learning_starts=2_000, batch_size=64,
                tau=1.0, train_freq=4, gradient_steps=1, target_update_interval=1_000,
                exploration_fraction=0.2, exploration_initial_eps=1.0,
                exploration_final_eps=0.05)


import gymnasium as _gym


class InfoEpisodeKeyGuard(_gym.Wrapper):
    """Rename `info["episode"]` so it does not collide with the SB3 convention.

    A REAL INTEROPERABILITY DEFECT IN THE ENVIRONMENT, worked around here and
    reported in `BASELINES.md`.  `KubeGymEnv` puts the corpus episode *index*
    under `info["episode"]` on every step (`GYM_API.md` section 6, "Identity").
    Gymnasium's `Monitor` wrapper -- which `stable_baselines3.common.env_util`
    applies by default -- uses the same key for the end-of-episode statistics
    *dict*, and SB3's logger then does `len(self.ep_info_buffer[0])`.  With the
    integer in place that raises `TypeError: object of type 'int' has no len()`
    on the first log dump, so **PPO and DQN cannot be trained on a KubeGym task
    through the standard SB3 helper without a wrapper**.  `positioning.md` claims
    the environment ships "with the standard `reset`/`step` contract so that
    existing RL tooling attaches without adapters" (section 1, item 1); on the
    `stable-baselines3` 2.9.0 / `gymnasium` 1.3.0 path measured here that is not
    quite true, and this class is the adapter.  `GYM_API.md` section 6 documents
    the colliding key itself (the "Identity" block lists `episode` among the
    per-step `info` keys); it does not discuss tooling compatibility.

    The identity information is preserved under `kubegym_episode`, so nothing is
    lost; only the key that Gymnasium reserves is vacated.
    """

    @staticmethod
    def _fix(info):
        if isinstance(info, dict) and "episode" in info and not isinstance(
                info["episode"], dict):
            info = dict(info)
            info["kubegym_episode"] = info.pop("episode")
        return info

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return obs, self._fix(info)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return obs, r, term, trunc, self._fix(info)


def _make_vec(family: str, action_mode: str, n_envs: int, seed: int):
    import gymnasium
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    from .runner import _ensure_registered
    _ensure_registered()
    assert_not_test("train", "RL training")
    tid = task_id(family, "train", action_mode)

    def factory():
        return InfoEpisodeKeyGuard(gymnasium.make(tid, split="train"))

    # DummyVecEnv, not SubprocVecEnv: a request_service control step is
    # ~0.7-2.2 ms, so subprocess IPC would cost more than the step it parallelises.
    return make_vec_env(factory, n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv)


def _eval_on_dev(model, family: str, action_mode: str, episodes: Sequence[int],
                 ledger: BudgetLedger, phase: str) -> Dict[str, Any]:
    assert_not_test("dev", "RL dev evaluation")
    env = make_env(family, "dev", action_mode)
    ws = int(env.unwrapped.task_spec.split_work_seeds["dev"][0])
    rows = []
    for ep in episodes:
        r = rollout(env, model, ep, ws)
        ledger.charge(r["n_control_steps"], phase)
        rows.append(r)
    env.close()
    return {"objective": objective(rows),
            "mean_cost_total": float(np.mean([r["cost_total"] for r in rows])),
            "mean_return": float(np.mean([r["return"] for r in rows])),
            "mean_slo_attainment": float(np.mean([r["slo_attainment"] for r in rows])),
            "mean_replica_seconds": float(np.mean([r["replica_seconds"] for r in rows])),
            "mean_scale_events": float(np.mean([r["scale_events"] for r in rows])),
            "n_episodes": len(rows)}


def _fit(algo: str, family: str, hp: Dict[str, Any], seed: int, total_steps: int,
         ledger: BudgetLedger, phase: str, curve_episodes: Optional[Sequence[int]] = None,
         eval_every: int = 0, save_path: Optional[str] = None) -> Dict[str, Any]:
    from stable_baselines3 import DQN, PPO
    cls = {"ppo": PPO, "dqn": DQN}[algo]
    venv = _make_vec(family, hp["action_mode"], N_ENVS, seed)
    model = cls(env=venv, **_algo_kwargs(algo, hp, seed))
    curve: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    done = 0
    chunk = eval_every if eval_every else total_steps
    while done < total_steps:
        n = min(chunk, total_steps - done)
        model.learn(total_timesteps=n, reset_num_timesteps=(done == 0),
                    progress_bar=False)
        done += n
        ledger.charge(n, phase + "_train")
        if eval_every and curve_episodes is not None:
            ev = _eval_on_dev(model, family, hp["action_mode"], curve_episodes,
                              ledger, phase + "_curve_dev")
            curve.append({"steps": done, **ev})
    venv.close()
    wall = time.perf_counter() - t0
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        model.save(save_path)
    return {"model": model, "curve": curve, "wall_s": round(wall, 1),
            "steps": done, "save_path": save_path}


def train_family(algo: str, family: str, out_dir: str = "out/rl",
                 cap: int = CAP_ENV_STEPS) -> Dict[str, Any]:
    import torch
    torch.set_num_threads(1)
    manifest = load_manifest()
    dev_eps = dev_tuning_episodes(family, manifest)
    curve_eps = dev_eps[:N_DEV_CURVE]
    ledger = BudgetLedger(method=algo, family=family, cap=cap)
    t_start = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)

    # ---- search phase: one seed, dev-selected -----------------------
    search: List[Dict[str, Any]] = []
    for i, hp in enumerate(hp_grid(algo)):
        fit = _fit(algo, family, hp, seed=0, total_steps=SEARCH_STEPS,
                   ledger=ledger, phase="search")
        ev = _eval_on_dev(fit["model"], family, hp["action_mode"], dev_eps,
                          ledger, "search_select_dev")
        search.append({"config": i, "hp": hp, "steps": fit["steps"],
                       "wall_s": fit["wall_s"], "dev": ev})
        ledger.log_trial(hp, ev["objective"], {"phase": "search", "steps": fit["steps"]})
        print(f"[{algo}/{family}] search {i} {hp} dev_obj={ev['objective']:.3f}", flush=True)
        del fit
    best = min(search, key=lambda s: (s["dev"]["objective"],
                                      json.dumps(s["hp"], sort_keys=True)))
    hp = dict(best["hp"])

    # ---- final phase: N_SEEDS seeds at the selected config ----------
    seeds: List[Dict[str, Any]] = []
    for seed in range(N_SEEDS):
        path = os.path.join(out_dir, f"{algo}_{family}_seed{seed}")
        fit = _fit(algo, family, hp, seed=seed, total_steps=FINAL_STEPS,
                   ledger=ledger, phase="final", curve_episodes=curve_eps,
                   eval_every=EVAL_EVERY, save_path=path)
        ev = _eval_on_dev(fit["model"], family, hp["action_mode"], dev_eps,
                          ledger, "final_select_dev")
        seeds.append({"seed": seed, "steps": fit["steps"], "wall_s": fit["wall_s"],
                      "curve": fit["curve"], "dev": ev, "model_path": path + ".zip",
                      "model_sha256": _sha256(path + ".zip")})
        ledger.log_trial(hp, ev["objective"], {"phase": "final", "seed": seed,
                                               "steps": fit["steps"]})
        print(f"[{algo}/{family}] seed {seed} dev_obj={ev['objective']:.3f} "
              f"wall={fit['wall_s']:.0f}s", flush=True)
        del fit
    ledger.wall_s = time.perf_counter() - t_start

    out = {
        "algo": algo, "family": family, "action_mode_selected": hp["action_mode"],
        "hp_selected": hp, "hp_grid": hp_grid(algo),
        "net_arch": NET_ARCH, "n_envs": N_ENVS, "n_seeds": N_SEEDS,
        "search_steps_per_config": SEARCH_STEPS, "final_steps_per_seed": FINAL_STEPS,
        "eval_every": EVAL_EVERY, "dev_episodes_selection": dev_eps,
        "dev_episodes_curve": curve_eps,
        "algo_kwargs": {k: v for k, v in _algo_kwargs(algo, hp, 0).items()
                        if k not in ("policy",)},
        "search": search, "seeds": seeds, "ledger": ledger.to_dict(),
        "notes": [
            "budget currency is train+dev environment steps; test steps are not charged",
            "no reward or observation normalisation wrapper (see module docstring)",
            "learning curves are dev-split evaluations, so they are not test leakage",
        ],
    }
    with open(os.path.join(out_dir, f"train_{algo}_{family}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    return out


def _sha256(path: str) -> Optional[str]:
    import hashlib
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    import sys
    algo = sys.argv[1]
    fams = sys.argv[2:] or list(FAMILIES)
    for fam in fams:
        d = train_family(algo, fam)
        print(f"[{algo}/{fam}] selected {d['hp_selected']} "
              f"spent={d['ledger']['spent_env_steps']}/{d['ledger']['cap_env_steps']} "
              f"wall={d['ledger']['wall_s']:.0f}s", flush=True)
