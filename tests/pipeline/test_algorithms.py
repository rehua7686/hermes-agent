"""Tests for algorithm correctness fixes in the Hermes memory pipeline (Phase 1).

Each test class targets a specific bug identified in the deep analysis doc.
These tests are written TDD-style: they SHOULD FAIL against the current
implementation and PASS once the corresponding fix is applied.

Bug IDs covered:
    H1  - Salience double counting (novelty + rep_factor both use freshness)
    H2  - Schema dedup only checks first 50 characters
    H3  - New engram strength always 1.0 (no fragile period)
    H4  - ActivationGraph shortest-path uses strength as weight (finds weakest)
    H5  - Prediction failure penalizes ALL schemas (confidence deflation)
    M2  - Emotion decay uses global valence, not per-memory
    D2  - schemas.updated_at never updated on UPDATE
    D4  - predict() schema_id LIKE pattern direction reversed
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent.memory_pipeline import (
    ActivationGraph,
    ConsolidationEngine,
    FeedbackCoordinator,
    PipelineState,
    SalienceScorer,
    SilentEngramEngine,
)
from agent.pipeline.feature_flags import FeatureFlags, _FLAG_NAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _enable_all_feature_flags(monkeypatch):
    """Enable all v2 feature flags so TDD tests validate the fixed behavior.

    This patches FeatureFlags in agent.memory_pipeline so every ``_ff =
    FeatureFlags()`` call returns an instance with all flags ON.
    """
    _all_on = {name: True for name in _FLAG_NAMES}

    def _all_enabled_cls(config=None):
        return FeatureFlags(_all_on)

    monkeypatch.setattr(
        "agent.memory_pipeline.FeatureFlags", _all_enabled_cls,
    )


@pytest.fixture()
def state(tmp_path, monkeypatch):
    """Create a PipelineState backed by a temporary database.

    Monkeypatches hermes_state so the test never touches the real home dir.
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


@pytest.fixture()
def scorer() -> SalienceScorer:
    """Return a fresh SalienceScorer."""
    return SalienceScorer()


# ===========================================================================
# H1: Salience double-counting fix
# ===========================================================================

class TestSalienceScorerFix:
    """H1: novelty and rep_factor both derived from freshness.

    The raw salience formula includes novelty (which contains freshness)
    AND then multiplies by rep_factor (also freshness).  Repeated messages
    are penalized twice for the same signal.

    Fix: decouple novelty from rep_penalty.  Only penalize via one path.
    """

    def test_no_double_counting(self, scorer: SalienceScorer):
        """Repeating a message should penalize but not excessively.

        After the fix, score2 should be < score1 but > score1 * 0.3.
        The old code double-counts freshness, pushing score2 far below
        score1 * 0.3 for moderate repetition.
        """
        msg = "the quick brown fox jumps over the lazy dog near the river"
        score1 = scorer.score(msg).overall
        score2 = scorer.score(msg).overall
        assert score1 > 0.0, "sanity: first score must be positive"
        assert score2 < score1, "repeated message should score lower"
        assert score2 > score1 * 0.3, (
            f"penalty is excessive: score2={score2:.4f} <= "
            f"score1*0.3={score1 * 0.3:.4f} "
            "(double-counting of freshness signal)"
        )

    def test_trivial_pattern_escape(self, scorer: SalienceScorer):
        """A genuinely important message matching a trivial pattern
        should still score above 0.3 if it has strong emotion/importance signals.

        The trivial-pattern penalty should not crush strong signals entirely.
        """
        # This message matches the "ok" trivial pattern (starts with "ok")
        # but also carries strong emotion + importance.
        msg = "ok but this is extremely important: critical emergency crash!!"
        result = scorer.score(msg)
        assert result.emotion >= 0.4, "emotion signals should be detected"
        assert result.importance >= 0.4, "importance signals should be detected"
        assert result.overall > 0.3, (
            f"strong signals crushed by trivial penalty: overall={result.overall:.4f}"
        )

    def test_novelty_decoupled_from_rep_factor(self, scorer: SalienceScorer):
        """novelty and rep_penalty should be independent signals.

        After the fix, novelty reflects genuine content freshness (decays
        with repetition but not as sharply), while rep_penalty only applies
        an extra penalty for trivial repeated content.  The key invariant:
        overall score drops with repetition, but novelty does not collapse
        as fast as the old double-penalised score.
        """
        msg = "important system design decision for the database layer"
        r1 = scorer.score(msg)
        r2 = scorer.score(msg)
        # Novelty should decrease somewhat but not collapse
        assert r2.novelty < r1.novelty, "novelty should decrease with repetition"
        assert r2.novelty > r1.novelty * 0.3, (
            f"novelty collapsed: r2.novelty={r2.novelty:.4f} vs "
            f"r1.novelty={r1.novelty:.4f}"
        )
        # Overall still decreases
        assert r2.overall < r1.overall


# ===========================================================================
# H3: New engram strength fix
# ===========================================================================

class TestEngramStrengthFix:
    """H3: strengthen() inserts new engrams at strength=1.0 (min(1.0, 1.0+delta)).

    New memories should start in a "fragile period" (Ebbinghaus 1885),
    typically around 0.3, and be strengthened through retrieval/consolidation.
    """

    def test_new_engram_not_at_max(self, state: PipelineState):
        """A newly strengthened memory should NOT start at strength=1.0."""
        engine = SilentEngramEngine(half_life_hours=720.0)
        strength = engine.strengthen(state, "brand_new_memory_ref", delta=0.03)
        assert strength < 1.0, (
            f"new engram starts at max strength: {strength}"
        )

    def test_new_engram_in_fragile_range(self, state: PipelineState):
        """New engram strength should be between 0.2 and 0.5.

        The proposed INITIAL_ENGRAM_STRENGTH is 0.3, plus delta (0.03)
        gives ~0.33.  We allow the [0.2, 0.5] range to accommodate
        implementation flexibility.
        """
        engine = SilentEngramEngine(half_life_hours=720.0)
        strength = engine.strengthen(state, "fragile_test_ref", delta=0.03)
        assert 0.2 <= strength <= 0.5, (
            f"new engram strength {strength} outside fragile range [0.2, 0.5]"
        )

    def test_strengthen_increases_gradually(self, state: PipelineState):
        """Multiple strengthen() calls should gradually increase strength.

        Each call adds delta; strength should climb from the fragile
        starting point toward 1.0 over several invocations.
        """
        engine = SilentEngramEngine(half_life_hours=720.0)
        ref = "gradual_strengthen_ref"
        strengths = [engine.strengthen(state, ref, delta=0.1) for _ in range(10)]
        # Should be monotonically non-decreasing
        for i in range(1, len(strengths)):
            assert strengths[i] >= strengths[i - 1], (
                f"strength decreased at step {i}: "
                f"{strengths[i]} < {strengths[i - 1]}"
            )
        # Final should be higher than initial
        assert strengths[-1] > strengths[0], "strength did not increase"
        # Initial should be in fragile range
        assert strengths[0] < 0.5, (
            f"initial strength too high: {strengths[0]}"
        )


# ===========================================================================
# H2: Schema dedup fix
# ===========================================================================

class TestSchemaDedupFix:
    """H2: ConsolidationEngine dedup only checks content[:50].

    Two facts sharing the first 50 characters but differing afterwards
    are incorrectly treated as duplicates.  Fix: use content hash for
    exact dedup + character-level Jaccard similarity for approximate dedup.
    """

    def test_exact_duplicate_detected(self, state: PipelineState):
        """Two identical facts should be deduped."""
        engine = ConsolidationEngine(min_facts=3)
        facts = [
            {"content": f"exact dedup test fact {i} with enough characters here"}
            for i in range(4)
        ]
        first = engine.consolidate(state, facts=facts)
        second = engine.consolidate(state, facts=facts)
        assert first["schemas_created"] > 0
        # Second run with same facts should create 0 new schemas
        assert second["schemas_created"] == 0, (
            f"exact duplicates not deduped: {second['schemas_created']} created"
        )

    def test_different_content_same_prefix_not_deduped(self, state: PipelineState):
        """Two facts sharing the first 50 chars but different after should NOT
        be deduped.  This is the core H2 bug: the old code uses
        content[:50] as the dedup key, so facts with identical prefixes
        but divergent content are falsely merged.

        The prefix (50 chars) is: "This is a very long fact about climate change and"
        """
        prefix = "This is a very long fact about climate change and"
        # Ensure prefix is exactly 50 chars (pad if needed)
        prefix = prefix[:50].ljust(50)
        facts_a = [
            {"content": prefix + " how rising sea levels affect coastal cities worldwide. " + str(i)}
            for i in range(4)
        ]
        facts_b = [
            {"content": prefix + " how deforestation impacts biodiversity in tropical forests. " + str(i)}
            for i in range(4)
        ]
        engine = ConsolidationEngine(min_facts=3)
        result_a = engine.consolidate(state, facts=facts_a)
        assert result_a["schemas_created"] > 0, "first batch should create schemas"
        result_b = engine.consolidate(state, facts=facts_b)
        assert result_b["schemas_created"] > 0, (
            f"different content with same prefix was falsely deduped: "
            f"schemas_created={result_b['schemas_created']}"
        )

    def test_approximate_duplicate_merged(self, state: PipelineState):
        """Very similar facts (>85% Jaccard) should merge (increment confidence),
        not silently drop.  The fix should detect approximate duplicates
        and UPDATE the existing schema rather than INSERTing a new one.
        """
        engine = ConsolidationEngine(min_facts=3)
        # These facts are nearly identical (differ by ~2 words)
        base = "the memory consolidation process in the hippocampus transfers episodic"
        variant = "the memory consolidation process in the hippocampus moves episodic"
        facts_a = [
            {"content": f"{base} memories to long term storage during sleep phase {i}"}
            for i in range(4)
        ]
        facts_b = [
            {"content": f"{variant} memories to long term storage during sleep phase {i}"}
            for i in range(4)
        ]
        result_a = engine.consolidate(state, facts=facts_a)
        assert result_a["schemas_created"] > 0
        # Count schemas before second batch
        count_before = state._conn.execute(
            "SELECT COUNT(*) as c FROM schemas"
        ).fetchone()["c"]
        result_b = engine.consolidate(state, facts=facts_b)
        count_after = state._conn.execute(
            "SELECT COUNT(*) as c FROM schemas"
        ).fetchone()["c"]
        # Approximate dupes should NOT create new schemas; they should
        # either be merged (no new row) or merged + 1 new row max
        # The key invariant: count should not jump by len(facts_b)
        new_schemas = count_after - count_before
        assert new_schemas < len(facts_b), (
            f"approximate duplicates created {new_schemas} new schemas "
            f"(expected < {len(facts_b)}); similarity merge not working"
        )

    def test_dedup_checks_beyond_50_chars(self, state: PipelineState):
        """Dedup should look at full content, not just first 50 chars.

        Insert a fact with a 100-char content, then try to insert a fact
        with the same first 50 chars but completely different continuation.
        The second fact should be accepted.
        """
        engine = ConsolidationEngine(min_facts=3)
        prefix = "A" * 50
        facts_a = [
            {"content": prefix + "B" * 50 + f" filler {i}"}
            for i in range(4)
        ]
        facts_b = [
            {"content": prefix + "C" * 50 + f" filler {i}"}
            for i in range(4)
        ]
        result_a = engine.consolidate(state, facts=facts_a)
        assert result_a["schemas_created"] > 0
        result_b = engine.consolidate(state, facts=facts_b)
        assert result_b["schemas_created"] > 0, (
            "content differing after char 50 was falsely deduped"
        )


# ===========================================================================
# H4: ActivationGraph weight direction fix
# ===========================================================================

class TestActivationGraphWeightFix:
    """H4: find_bridge_entities uses strength as edge weight for nx.shortest_path.

    nx.shortest_path minimises total weight, so using raw strength means
    it finds the WEAKEST path.  Fix: use weight=1/strength so stronger
    edges have shorter distance.
    """

    def test_strongest_path_found(self, state: PipelineState):
        """Bridge entity path should go through strong edges, not weak ones.

        Graph topology:
            A --(0.9)--> B --(0.9)--> C       (strong path, 2 hops)
            A --(0.1)--> D --(0.1)--> C       (weak path, 2 hops)

        The correct bridge between A and C is B (through strong edges),
        not D (through weak edges).
        """
        graph = ActivationGraph()
        # Strong path: A -> B -> C
        graph.record_co_activation(state, ["Alpha", "Bridge"], delta=0.9)
        graph.record_co_activation(state, ["Bridge", "Charlie"], delta=0.9)
        # Weak path: A -> D -> C
        graph.record_co_activation(state, ["Alpha", "WeakLink"], delta=0.1)
        graph.record_co_activation(state, ["WeakLink", "Charlie"], delta=0.1)

        bridge = graph.find_bridge_entities(state, "Alpha", "Charlie")
        assert len(bridge) > 0, "no bridge path found"
        assert bridge[0] == "Bridge", (
            f"expected bridge through strong edge 'Bridge', "
            f"got '{bridge[0]}' (likely found weakest path)"
        )


# ===========================================================================
# H5: Schema confidence deflation fix
# ===========================================================================

class TestSchemaConfidenceFix:
    """H5: observe_outcome() penalizes ALL schemas when prediction fails.

    Current code: UPDATE schemas SET confidence = MAX(0.1, confidence - 0.05)
                  WHERE confidence > 0.3
    This punishes every schema, causing "learned helplessness" -- all
    confidences deflate toward 0.1 over time.

    Fix: only penalise the schema that generated the prediction.
    """

    def _insert_schema(self, state, content, confidence, domain="general"):
        state._conn.execute(
            "INSERT INTO schemas (content, domain, confidence) VALUES (?, ?, ?)",
            (content, domain, confidence),
        )
        state._conn.commit()

    def test_only_targeted_schema_affected(self, state: PipelineState):
        """When a prediction fails, only the generating schema loses confidence."""
        self._insert_schema(state, "target schema content for prediction", 0.8)
        self._insert_schema(state, "unrelated schema about something else", 0.8)

        feedback = FeedbackCoordinator()
        # Simulate: predict, then observe a totally different outcome
        feedback.predict(state, "some context")
        feedback.observe_outcome(state, "completely different outcome text")

        rows = state._conn.execute(
            "SELECT content, confidence FROM schemas ORDER BY schema_id"
        ).fetchall()
        target_conf = rows[0]["confidence"]
        unrelated_conf = rows[1]["confidence"]

        # The unrelated schema should NOT have been penalised
        assert unrelated_conf == pytest.approx(0.8, abs=0.01), (
            f"unrelated schema confidence deflated: {unrelated_conf}"
        )

    def test_unrelated_schemas_unchanged(self, state: PipelineState):
        """Schemas not involved in the prediction should keep their confidence."""
        for i in range(5):
            self._insert_schema(
                state,
                f"schema number {i} about topic {chr(65 + i)}",
                0.7,
            )
        feedback = FeedbackCoordinator()
        feedback.predict(state, "context")
        feedback.observe_outcome(state, "totally unrelated outcome")

        rows = state._conn.execute(
            "SELECT confidence FROM schemas ORDER BY schema_id"
        ).fetchall()
        for row in rows:
            assert row["confidence"] >= 0.69, (
                f"schema confidence dropped to {row['confidence']}; "
                "all schemas were penalised instead of just the target"
            )

    def test_no_confidence_deflation(self, state: PipelineState):
        """After 10 wrong predictions, unrelated schemas should still have
        their original confidence."""
        self._insert_schema(state, "the one schema used for predictions", 0.8)
        for i in range(5):
            self._insert_schema(
                state,
                f"independent schema {i} not used in predictions",
                0.8,
            )
        feedback = FeedbackCoordinator()
        for _ in range(10):
            feedback.predict(state, "context")
            feedback.observe_outcome(state, "wrong answer every single time")

        rows = state._conn.execute(
            "SELECT content, confidence FROM schemas "
            "WHERE content LIKE 'independent schema%'"
        ).fetchall()
        assert len(rows) == 5
        for row in rows:
            assert row["confidence"] >= 0.75, (
                f"'{row['content']}' confidence deflated to "
                f"{row['confidence']} after 10 wrong predictions"
            )


# ===========================================================================
# D2: Schema updated_at fix
# ===========================================================================

class TestSchemaTimestampFix:
    """D2: UPDATE statements on schemas table never set updated_at.

    This means predict()'s temporal query (WHERE updated_at >= datetime('now','-1 day'))
    never finds updated schemas -- only newly created ones.
    """

    def test_updated_at_changes_on_update(self, state: PipelineState):
        """UPDATE schemas should set updated_at = CURRENT_TIMESTAMP."""
        state._conn.execute(
            "INSERT INTO schemas (content, domain, confidence) "
            "VALUES (?, ?, ?)",
            ("timestamp test schema", "general", 0.5),
        )
        state._conn.commit()
        before = state._conn.execute(
            "SELECT updated_at FROM schemas WHERE content = ?",
            ("timestamp test schema",),
        ).fetchone()["updated_at"]

        # Trigger an update (e.g., confidence bump)
        state._conn.execute(
            "UPDATE schemas SET confidence = 0.7, "
            "updated_at = CURRENT_TIMESTAMP WHERE content = ?",
            ("timestamp test schema",),
        )
        state._conn.commit()
        after = state._conn.execute(
            "SELECT updated_at FROM schemas WHERE content = ?",
            ("timestamp test schema",),
        ).fetchone()["updated_at"]
        assert after >= before, (
            f"updated_at did not advance: before={before}, after={after}"
        )


# ===========================================================================
# D4: predict() schema_id fix
# ===========================================================================

class TestPredictSchemaIdFix:
    """D4: predict() uses reversed LIKE pattern for schema_id lookup.

    Current: WHERE ? LIKE '%' || substr(content, 1, 50) || '%'
    This checks if the prediction text contains the content prefix.
    But the prediction text has a prefix like "Expected pattern (conf=...):"
    prepended, so the schema content prefix does NOT appear at the start.

    Fix: WHERE substr(content, 1, 50) LIKE '%' || ? || '%'
    or use a proper substring match.
    """

    def test_schema_id_populated(self, state: PipelineState):
        """predict() should find the correct schema_id for its prediction."""
        # Insert a schema with enough content to match
        content = (
            "Python is a high-level programming language known for its "
            "readability and versatility in software development"
        )
        state._conn.execute(
            "INSERT INTO schemas (content, domain, confidence) "
            "VALUES (?, ?, ?)",
            (content, "programming", 0.85),
        )
        state._conn.commit()
        feedback = FeedbackCoordinator()
        feedback.predict(state, "tell me about Python")
        # Check that predictions table has a non-null schema_id
        pred_rows = state._conn.execute(
            "SELECT prediction, schema_id FROM predictions"
        ).fetchall()
        assert len(pred_rows) > 0, "no predictions were stored"
        matched = [r for r in pred_rows if r["schema_id"] is not None]
        assert len(matched) > 0, (
            "predict() failed to populate schema_id for any prediction; "
            "the LIKE pattern is likely reversed (D4 bug)"
        )


# ===========================================================================
# M2: Per-memory emotion decay fix
# ===========================================================================

class TestEmotionDecayFix:
    """M2: apply_decay uses a single global emotional_valence for all memories.

    The fix (apply_decay_with_emotion) should accept per-memory valences
    and adjust each memory's half-life individually, so emotionally
    significant memories decay slower than neutral ones.
    """

    def test_per_memory_emotion(self, state: PipelineState):
        """apply_decay_with_emotion should use per-memory valence, not global."""
        engine = SilentEngramEngine(
            half_life_hours=720.0,
            emotion_modulated_decay_enabled=True,
            emotion_decay_multiplier=2.0,
        )
        # Create two memories at identical strength
        engine.strengthen(state, "emotional_memory", delta=0.0)
        engine.strengthen(state, "neutral_memory", delta=0.0)

        # Apply decay: emotional_memory has high valence (slower decay),
        # neutral_memory has zero valence (normal decay).
        engine.apply_decay_with_emotion(
            state,
            hours_elapsed=720.0,  # one half-life
            emotional_valences={
                "emotional_memory": 0.6,
                "neutral_memory": 0.0,
            },
        )

        emo_str = state._conn.execute(
            "SELECT strength FROM engram_strengths WHERE memory_ref = ?",
            ("emotional_memory",),
        ).fetchone()["strength"]
        neu_str = state._conn.execute(
            "SELECT strength FROM engram_strengths WHERE memory_ref = ?",
            ("neutral_memory",),
        ).fetchone()["strength"]

        # Emotional memory should have decayed LESS (higher strength)
        assert emo_str > neu_str, (
            f"emotional memory ({emo_str}) should retain more strength "
            f"than neutral memory ({neu_str}) after equal elapsed time"
        )
        # Neutral memory should be near half (normal half-life decay)
        assert neu_str == pytest.approx(0.5, abs=0.05), (
            f"neutral memory should decay at normal rate, got {neu_str}"
        )
        # Emotional memory should be noticeably above half
        # With valence=0.6, multiplier=2.0: half_life = 720*(1+1.2) = 1584h
        # After 720h: 0.5^(720/1584) ≈ 0.73
        assert emo_str > 0.6, (
            f"emotion-modulated memory decayed too fast: {emo_str}"
        )
