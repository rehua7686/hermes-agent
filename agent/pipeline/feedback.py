"""FeedbackCoordinator -- Layer 5 predictive processing + learning.

Three interconnected feedback loops:
1. SalienceLearner: learns which signals predict useful memories
2. PredictiveModel: generates expectations from schemas
3. CrossDomainBridge: discovers unexpected connections

Scientific basis: Predictive coding (Friston 2010).
Thread-safe: _pending_predictions protected by _lock.

Contains:
- FeedbackCoordinator class
- All v2_confidence fixes (targeted schema confidence)
- All v2_predict fixes (schema_id extraction)
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from agent.pipeline.feature_flags import FeatureFlags

logger = logging.getLogger(__name__)


class FeedbackCoordinator:
    """Three interconnected feedback loops.

    1. SalienceLearner: learns which signals predict useful memories
    2. PredictiveModel: generates expectations from schemas
    3. CrossDomainBridge: discovers unexpected connections
    Scientific basis: Predictive coding (Friston 2010).
    Thread-safe: _pending_predictions protected by _lock.
    """

    def __init__(self) -> None:
        self._pending_predictions: list[str] = []
        self._reconsolidation: 'ReconsolidationEngine | None' = None
        self._lock = threading.Lock()

    def predict(self, state: 'PipelineState', context: str) -> list[str]:
        """Generate predictions from existing schemas.

        Bitemporal support: predictions are drawn from both high-confidence
        schemas AND recently-updated schemas (temporal query).  Recently
        updated schemas are tagged with a recency marker so downstream
        consumers know they describe near-current situations.
        """
        if not state:
            return []
        try:
            _ff = FeatureFlags()
            _use_db = _ff.is_enabled('v2_concurrency') or hasattr(state, 'db')
            # C3 fix: use state.db.execute() to eliminate lock-order inversion
            if _use_db:
                conf_rows = state.db.execute(
                    "SELECT content, confidence, updated_at FROM schemas "
                    "WHERE confidence > 0.3 ORDER BY confidence DESC LIMIT 3"
                ).fetchall()
                recent_rows = state.db.execute(
                    "SELECT content, confidence, updated_at FROM schemas "
                    "WHERE updated_at >= datetime('now', '-1 day') "
                    "ORDER BY updated_at DESC LIMIT 3"
                ).fetchall()
            else:
                with state._lock:
                    conf_rows = state._conn.execute(
                        "SELECT content, confidence, updated_at FROM schemas "
                        "WHERE confidence > 0.3 ORDER BY confidence DESC LIMIT 3"
                    ).fetchall()
                    recent_rows = state._conn.execute(
                        "SELECT content, confidence, updated_at FROM schemas "
                        "WHERE updated_at >= datetime('now', '-1 day') "
                        "ORDER BY updated_at DESC LIMIT 3"
                    ).fetchall()
            # Merge, deduplicating by content prefix
            seen_prefixes: set[str] = set()
            predictions: list[str] = []
            for row in conf_rows:
                prefix = row["content"][:50]
                if prefix in seen_prefixes:
                    continue
                seen_prefixes.add(prefix)
                predictions.append(
                    f"Expected pattern (conf={row['confidence']:.2f}): "
                    f"{row['content'][:100]}"
                )
            for row in recent_rows:
                prefix = row["content"][:50]
                if prefix in seen_prefixes:
                    continue
                seen_prefixes.add(prefix)
                predictions.append(
                    f"Recent pattern (updated={row['updated_at']}, "
                    f"conf={row['confidence']:.2f}): {row['content'][:100]}"
                )
            # Fix 3: Include recently consolidated schemas as predictions
            _db_exec = state.db.execute if _use_db else state._conn.execute
            try:
                last_run = _db_exec(
                    "SELECT timestamp FROM consolidation_runs "
                    "ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if last_run:
                    recent_schemas = _db_exec(
                        "SELECT content, confidence FROM schemas "
                        "WHERE created_at >= ? OR updated_at >= ? "
                        "ORDER BY confidence DESC LIMIT 3",
                        (last_run["timestamp"], last_run["timestamp"]),
                    ).fetchall()
                    for srow in recent_schemas:
                        sprefix = srow["content"][:50]
                        if sprefix not in seen_prefixes:
                            seen_prefixes.add(sprefix)
                            predictions.append(
                                f"Consolidated schema "
                                f"(conf={srow['confidence']:.2f}): "
                                f"{srow['content'][:100]}"
                            )
            except Exception as e:
                logger.debug(
                    "Consolidated schema prediction failed: %s", e)
            # Insert predictions into predictions table
            try:
                _ff7 = FeatureFlags()
                for pred_text in predictions:
                    if _ff7.is_enabled('v2_predict'):
                        # FIX 7 (D4): Extract actual schema content from
                        # prediction text (strip prefix) for matching
                        _match_text = pred_text
                        _colon = pred_text.find("): ")
                        if _colon > 0:
                            _match_text = pred_text[_colon + 3:]
                        row = _db_exec(
                            "SELECT schema_id FROM schemas "
                            "WHERE ? LIKE '%' || "
                            "substr(content, 1, 50) || '%' "
                            "ORDER BY confidence DESC "
                            "LIMIT 1",
                            (_match_text,),
                        ).fetchone()
                    else:
                        row = _db_exec(
                            "SELECT schema_id FROM schemas "
                            "WHERE ? LIKE '%' || "
                            "substr(content, 1, 50) || '%' "
                            "ORDER BY confidence DESC "
                            "LIMIT 1",
                            (pred_text,),
                        ).fetchone()
                    sid = (row["schema_id"]
                           if row else None)
                    _db_exec(
                        "INSERT INTO predictions "
                        "(schema_id, prediction, "
                        " context) "
                        "VALUES (?, ?, ?)",
                        (sid, pred_text, context),
                    )
                if _use_db:
                    state.db.commit()
                else:
                    state._conn.commit()
            except Exception as e:
                logger.debug(
                    "Prediction insert failed: %s", e)
            with self._lock:
                self._pending_predictions = predictions
            return predictions
        except Exception as e:
            logger.debug("Prediction failed: %s", e)
            return []

    def observe_outcome(self, state: 'PipelineState',
                        actual: str) -> float:
        """Compare predictions against actual outcome. Returns error score."""
        with self._lock:
            pending = list(self._pending_predictions)
        if not pending or not state:
            return 0.0
        try:
            _ff = FeatureFlags()
            _use_db = _ff.is_enabled('v2_concurrency') or hasattr(state, 'db')
            actual_tokens = set(actual.lower().split())
            max_error = 0.0
            for pred in pending:
                pred_tokens = set(pred.lower().split())
                if not pred_tokens or not actual_tokens:
                    continue
                overlap = len(pred_tokens & actual_tokens) / max(
                    1, len(pred_tokens | actual_tokens))
                error = 1.0 - overlap
                max_error = max(max_error, error)

            # Update schema confidence based on prediction error
            _ff5 = FeatureFlags()
            if max_error > 0.5:
                # High error: schema was wrong, decrease confidence
                if _ff5.is_enabled('v2_confidence'):
                    # FIX 5 (H5): Only penalize schemas that generated
                    # predictions AND whose content overlaps with outcome
                    _db_exec = state.db.execute if _use_db else state._conn.execute
                    _pred_rows = _db_exec(
                        "SELECT DISTINCT schema_id FROM predictions "
                        "WHERE schema_id IS NOT NULL"
                    ).fetchall()
                    _sids = [r["schema_id"] for r in _pred_rows]
                    if _sids:
                        _placeholders = ",".join("?" for _ in _sids)
                        _target_rows = _db_exec(
                            f"SELECT schema_id, content FROM schemas "
                            f"WHERE schema_id IN ({_placeholders})",
                            _sids,
                        ).fetchall()
                        _actual_tokens = set(actual.lower().split())
                        _targeted_sids = []
                        for _r in _target_rows:
                            _schema_tokens = set(
                                _r["content"].lower().split())
                            if (_actual_tokens and _schema_tokens
                                    and len(_actual_tokens & _schema_tokens)
                                    / max(1, len(_actual_tokens
                                                | _schema_tokens)) > 0.05):
                                _targeted_sids.append(_r["schema_id"])
                        if _targeted_sids:
                            _ph2 = ",".join(
                                "?" for _ in _targeted_sids)
                            _db_exec(
                                f"UPDATE schemas SET "
                                f"confidence = MAX(0.1, "
                                f"  confidence - 0.05), "
                                f"updated_at = CURRENT_TIMESTAMP "
                                f"WHERE schema_id IN ({_ph2})",
                                _targeted_sids,
                            )
                    if _use_db:
                        state.db.commit()
                    else:
                        state._conn.commit()
                else:
                    if _use_db:
                        state.db.execute(
                            "UPDATE schemas SET confidence = MAX(0.1, confidence - 0.05), "
                            "updated_at = CURRENT_TIMESTAMP "
                            "WHERE confidence > 0.3"
                        )
                        state.db.commit()
                    else:
                        with state._lock:
                            state._conn.execute(
                                "UPDATE schemas SET confidence = MAX(0.1, confidence - 0.05), "
                                "updated_at = CURRENT_TIMESTAMP "
                                "WHERE confidence > 0.3"
                            )
                            state._conn.commit()
                # Fix 4: High prediction error triggers reconsolidation
                if self._reconsolidation:
                    try:
                        self._reconsolidation.check_retrieval(
                            state,
                            pending[0][:200] if pending else "",
                            actual[:200],
                        )
                    except Exception as e:
                        logger.debug(
                            "Prediction-error reconsolidation "
                            "failed: %s", e)
            elif max_error < 0.2:
                # Low error: schema was right, increase confidence
                if _use_db:
                    state.db.execute(
                        "UPDATE schemas SET confidence = MIN(1.0, confidence + 0.03), "
                        "updated_at = CURRENT_TIMESTAMP "
                        "WHERE confidence > 0.3"
                    )
                    state.db.commit()
                else:
                    with state._lock:
                        state._conn.execute(
                            "UPDATE schemas SET confidence = MIN(1.0, confidence + 0.03), "
                            "updated_at = CURRENT_TIMESTAMP "
                            "WHERE confidence > 0.3"
                        )
                        state._conn.commit()

            with self._lock:
                self._pending_predictions = []
            return max_error
        except Exception as e:
            logger.debug("Observe outcome failed: %s", e)
            return 0.0

    def discover_bridges(self, state: 'PipelineState') -> int:
        """Discover cross-domain connections between entities."""
        if not state:
            return 0
        try:
            with state._lock:
                # Find entities that appear in multiple domains
                rows = state._conn.execute(
                    "SELECT entity, COUNT(DISTINCT domain_a) as domain_count "
                    "FROM cross_domain_links GROUP BY entity "
                    "HAVING domain_count >= 2"
                ).fetchall()
                return len(rows)
        except Exception as e:
            logger.debug("Bridge discovery failed: %s", e)
            return 0
