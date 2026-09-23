"""Tests for the corpus <-> Gym bridge (`kubegym.workloads.gym_tasks`).

The bridge is the only place the corpus and Gym subpackages meet, and it was
added after both were built independently: without it the 767-trace corpus is
unreachable through the Gym API. These tests pin the join itself, and the
documented replay semantics that differ from the builtin tasks.

Every test skips (rather than fails) when no corpus directory is configured, so
the suite still runs from a bare package checkout.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from kubegym.workloads import FAMILIES, SPLITS  # noqa: E402


def _corpus_dir():
    d = os.environ.get("KUBEGYM_CORPUS_DIR")
    if not d or not os.path.isfile(os.path.join(d, "corpus_manifest.json")):
        pytest.skip("KUBEGYM_CORPUS_DIR not set to a corpus directory")
    return d


@pytest.fixture(scope="module")
def registered():
    """Register the corpus tasks, then remove them again.

    Registration is global (it writes into `gymnasium.registry`), so leaving
    corpus tasks behind breaks sibling tests that assert the registry holds
    exactly the builtin set. Teardown follows the same convention as
    `test_gym_registry._unregister`.
    """
    d = _corpus_dir()
    from kubegym.workloads import register_corpus_tasks
    ids = register_corpus_tasks(d, override=True, max_episodes=3)
    try:
        yield ids
    finally:
        from kubegym.gym import tasks as t
        for name in [n for n in list(t._TASKS) if n.startswith("Corpus-")]:
            t._TASKS.pop(name, None)
        for gid in ids:
            t._IDS.pop(gid, None)
            gym.registry.pop(gid, None)


def test_every_family_and_split_registers(registered):
    from kubegym.gym import list_tasks
    names = {t["task"] for t in list_tasks() if t["task"].startswith("Corpus-")}
    # 5 families x 3 splits, both action modes share a task name
    assert len(names) == len(FAMILIES) * len(SPLITS), sorted(names)
    for split in SPLITS:
        assert any(n.endswith("-" + split.capitalize()) for n in names)


def test_registered_ids_are_makeable_and_declare_uncalibrated(registered):
    from kubegym.gym import list_tasks
    corpus = [t for t in list_tasks() if t["task"].startswith("Corpus-")]
    assert corpus
    for t in corpus:
        # The corpus is a reconstruction, never measured data: the claims gate
        # must stay closed for every corpus task.
        assert t["calibrated"] is False, t["id"]
        assert t["horizon_s"] == 3600.0
        env = gym.make(t["id"])
        env.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_one_episode_runs_and_reports_provenance(registered, family):
    from kubegym.gym import list_tasks
    from kubegym.workloads.gym_tasks import _TITLE
    want = f"Corpus-{_TITLE[family]}-Test"
    t = next((t for t in list_tasks()
              if t["task"] == want and t["action_mode"] == "delta"), None)
    if t is None:
        pytest.skip(f"{want} not registered (missing extract?)")
    env = gym.make(t["id"])
    obs, info = env.reset(seed=0, options={"episode": 0, "work_seed": 0})
    assert obs.shape == env.observation_space.shape
    done = False
    n = 0
    while not done:
        obs, r, term, trunc, info = env.step(2)   # delta 0 == hold
        assert np.isfinite(r)
        done = term or trunc
        n += 1
    assert n == int(t["n_control_steps"])
    assert info["calibrated"] is False
    es = info["episode_summary"]
    assert es["n_requests"] >= 0
    env.close()


def test_replay_is_seed_invariant_but_episode_sensitive(registered):
    """The documented departure from the builtin tasks.

    A corpus trace's realisation is fixed by its manifest seed so its sha256
    stays verifiable, so the env `seed` must NOT change the offered load. This
    is the property a user needs in order to know that dispersion has to be
    reported across episodes, not across work seeds.
    """
    from kubegym.gym import list_tasks
    t = next(t for t in list_tasks()
             if t["task"] == "Corpus-Burst-Test" and t["action_mode"] == "delta")
    env = gym.make(t["id"])

    def rollout(seed, episode):
        env.reset(seed=seed, options={"episode": episode, "work_seed": 0})
        out, done = [], False
        while not done:
            _, r, term, trunc, _ = env.step(3)
            out.append(r)
            done = term or trunc
        return np.asarray(out)

    a, b = rollout(1, 0), rollout(12345, 0)
    assert np.array_equal(a, b), "same episode under different seeds must replay identically"
    c = rollout(1, 1)
    assert not np.array_equal(a, c), "different episodes must not be identical"
    env.close()


def test_digest_verification_catches_generator_drift(registered):
    """Records are checksummed against the manifest, so drift raises."""
    import kubegym.workloads.corpus as corpus_mod
    from kubegym.workloads.gym_tasks import CorpusTraceSource, find_corpus_dir
    from kubegym.workloads.azure2019 import AzureExtract
    from kubegym.workloads.corpus import load_manifest

    root = find_corpus_dir(_corpus_dir())
    man = load_manifest(os.path.join(root, "corpus_manifest.json"))
    entries = [e for e in man["traces"]
               if e["family"] == "burst" and e["split"] == "test"][:1]
    src = CorpusTraceSource(entries, None, name="drift-probe")
    assert len(src.records(0)) > 0          # clean path works

    real = corpus_mod.trace_digest
    try:
        corpus_mod.trace_digest = lambda *a, **k: "deadbeef" * 8
        src2 = CorpusTraceSource(entries, None, name="drift-probe-2")
        with pytest.raises(RuntimeError, match="does not reproduce its manifest digest"):
            src2.records(0)
    finally:
        corpus_mod.trace_digest = real


def test_manifest_declares_the_reconstruction_is_not_measured(registered):
    from kubegym.workloads.gym_tasks import CorpusTraceSource, find_corpus_dir
    from kubegym.workloads.corpus import load_manifest
    root = find_corpus_dir(_corpus_dir())
    man = load_manifest(os.path.join(root, "corpus_manifest.json"))
    entries = [e for e in man["traces"] if e["family"] == "constant"][:2]
    m = CorpusTraceSource(entries, None, name="probe").manifest()
    assert m["calibrated"] is False
    assert m["is_measured_data"] is False
    assert "RECONSTRUCTION" in m["note"] or "reconstruction" in m["note"].lower()
    assert len(m["trace_sha256"]) == len(entries)


def test_info_does_not_shadow_gymnasium_reserved_episode_key(registered):
    """`info["episode"]` is reserved by RecordEpisodeStatistics / SB3 Monitor.

    Publishing the corpus episode index there made SB3's logger raise TypeError
    on its first dump, so neither PPO nor DQN could be trained through the
    standard helpers. This pins the fix: the index lives under
    `episode_index`, and `episode` is left for the wrappers.
    """
    from kubegym.gym import list_tasks
    t = next(t for t in list_tasks()
             if t["task"] == "Corpus-Constant-Test" and t["action_mode"] == "delta")
    env = gym.make(t["id"])
    _, info = env.reset(seed=0, options={"episode": 0, "work_seed": 0})
    assert "episode_index" in info
    assert "episode" not in info, (
        "info['episode'] is reserved by gymnasium's episode-statistics wrappers")
    _, _, _, _, info = env.step(2)
    assert "episode" not in info
    assert info["episode_index"] == 0
    env.close()


def test_episode_statistics_wrapper_composes(registered):
    """The env must survive the wrapper that actually uses the reserved key."""
    from gymnasium.wrappers import RecordEpisodeStatistics
    from kubegym.gym import list_tasks
    t = next(t for t in list_tasks()
             if t["task"] == "Corpus-Constant-Test" and t["action_mode"] == "delta")
    env = RecordEpisodeStatistics(gym.make(t["id"]))
    env.reset(seed=0, options={"episode": 0, "work_seed": 0})
    done, info = False, {}
    while not done:
        _, _, term, trunc, info = env.step(2)
        done = term or trunc
    # the wrapper's own dict must be present and well formed
    assert isinstance(info.get("episode"), dict), info.get("episode")
    assert {"r", "l"} <= set(info["episode"])
    env.close()
