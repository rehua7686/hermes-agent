"""Shared fixtures for pipeline optimization tests.

Mirrors the hermetic-test invariants from the top-level conftest while
adding pipeline-specific helpers (state, scorer, feature flags).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable so ``agent`` and ``hermes_state`` resolve.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent.memory_pipeline import PipelineState, SalienceScorer
from agent.pipeline.feature_flags import FeatureFlags


# ---------------------------------------------------------------------------
# Pipeline state (temp DB, hermetic)
# ---------------------------------------------------------------------------

@pytest.fixture()
def state(tmp_path, monkeypatch):
    """Create a PipelineState backed by a temporary database.

    ``hermes_state.apply_wal_with_fallback`` is monkeypatched to a no-op so
    the test never touches the real hermes home directory.
    """
    import hermes_state

    monkeypatch.setattr(
        hermes_state,
        "apply_wal_with_fallback",
        lambda conn, db_label="": "wal",
    )
    db_path = str(tmp_path / "pipeline_test.db")
    ps = PipelineState(db_path=db_path)
    yield ps
    ps.close()


# ---------------------------------------------------------------------------
# Salience scorer
# ---------------------------------------------------------------------------

@pytest.fixture()
def scorer() -> SalienceScorer:
    """Return a fresh SalienceScorer instance (stateless, cheap)."""
    return SalienceScorer()


# ---------------------------------------------------------------------------
# Feature flags (all off by default -- safe baseline)
# ---------------------------------------------------------------------------

@pytest.fixture()
def feature_flags() -> FeatureFlags:
    """Return a FeatureFlags with every flag set to False."""
    return FeatureFlags()


@pytest.fixture()
def feature_flags_enabled() -> FeatureFlags:
    """Return a FeatureFlags with every flag set to True (for bulk-enable tests)."""
    return FeatureFlags({name: True for name in FeatureFlags({}).get_all()})
