"""PipelineState -- persistent storage, database accessor, error handling.

Contains:
- PipelineState: SQLite-backed persistent state for the memory pipeline
- DatabaseAccessor: Thread-safe SQLite wrapper (C1 fix)
- PipelineErrorHandler: Consecutive failure tracking, auto-disable
- _PIPELINE_SCHEMA: All CREATE TABLE statements
- _ZH_STOPWORDS: Shared Chinese stopword set
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ===========================================================================
# Shared Chinese stopwords (used by activation and consolidation)
# ===========================================================================

_ZH_STOPWORDS: set[str] = {
    '的', '了', '是', '在', '我', '你', '他', '她', '它', '们',
    '这', '那', '有', '和', '与', '及', '或', '但', '而', '就',
    '都', '要', '会', '能', '可以', '不', '没', '也', '还', '把',
    '被', '让', '给', '从', '到', '对',
}


# ===========================================================================
# Pipeline Schema (database provisioning)
# ===========================================================================

_PIPELINE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS salience_weights (
    signal_type TEXT PRIMARY KEY, weight REAL NOT NULL,
    sample_count INTEGER DEFAULT 0, success_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS salience_encoding_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL,
    fact_ref TEXT, emotion_score REAL, novelty_score REAL,
    importance_score REAL, overall_score REAL, was_helpful INTEGER DEFAULT -1,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS engram_strengths (
    memory_ref TEXT PRIMARY KEY, provider TEXT NOT NULL,
    strength REAL DEFAULT 1.0, last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    decay_half_life_hours REAL DEFAULT 720.0, access_count INTEGER DEFAULT 0,
    emotional_valence REAL DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS schemas (
    schema_id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL,
    domain TEXT DEFAULT 'general', confidence REAL DEFAULT 0.5,
    source_count INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, hrr_vector BLOB
);
CREATE TABLE IF NOT EXISTS schema_sources (
    schema_id INTEGER REFERENCES schemas(schema_id),
    memory_ref TEXT NOT NULL, provider TEXT NOT NULL,
    contribution REAL DEFAULT 1.0, PRIMARY KEY (schema_id, memory_ref)
);
CREATE TABLE IF NOT EXISTS reconsolidation_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT, memory_ref TEXT,
    old_content TEXT, new_content TEXT, prediction_error REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS consolidation_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
    memories_processed INTEGER DEFAULT 0, schemas_created INTEGER DEFAULT 0,
    schemas_updated INTEGER DEFAULT 0, conflicts_found INTEGER DEFAULT 0,
    duration_ms INTEGER DEFAULT 0, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT, schema_id INTEGER,
    prediction TEXT NOT NULL, context TEXT DEFAULT '', outcome TEXT DEFAULT '',
    error_score REAL DEFAULT 0.0, resolved INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, resolved_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS salience_feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT, memory_ref TEXT,
    signal_type TEXT, signal_value REAL, was_helpful INTEGER DEFAULT 0,
    was_retrieved INTEGER DEFAULT 0, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS activation_edges (
    source_entity TEXT NOT NULL, target_entity TEXT NOT NULL,
    strength REAL DEFAULT 0.1, co_activation_count INTEGER DEFAULT 1,
    last_activated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_entity, target_entity)
);
CREATE TABLE IF NOT EXISTS cross_domain_links (
    link_id INTEGER PRIMARY KEY AUTOINCREMENT, entity TEXT NOT NULL,
    domain_a TEXT NOT NULL, domain_b TEXT NOT NULL,
    fact_refs_a TEXT DEFAULT '', fact_refs_b TEXT DEFAULT '',
    strength REAL DEFAULT 0.5, discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS entity_frequencies (
    entity TEXT PRIMARY KEY, count INTEGER DEFAULT 0,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS cooccurrence_frequencies (
    entity_a TEXT NOT NULL, entity_b TEXT NOT NULL,
    count INTEGER DEFAULT 0,
    PRIMARY KEY (entity_a, entity_b)
);
CREATE INDEX IF NOT EXISTS idx_engram_strength ON engram_strengths(strength DESC);
CREATE INDEX IF NOT EXISTS idx_engram_provider ON engram_strengths(provider);
CREATE INDEX IF NOT EXISTS idx_schemas_domain ON schemas(domain);
CREATE INDEX IF NOT EXISTS idx_schemas_confidence ON schemas(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_activation_source ON activation_edges(source_entity);
CREATE INDEX IF NOT EXISTS idx_activation_target ON activation_edges(target_entity);
CREATE INDEX IF NOT EXISTS idx_cross_links_entity ON cross_domain_links(entity);
CREATE INDEX IF NOT EXISTS idx_salience_feedback_ref ON salience_feedback(memory_ref, was_retrieved);
CREATE INDEX IF NOT EXISTS idx_schemas_updated_at ON schemas(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_consolidation_runs_ts ON consolidation_runs(timestamp DESC);
"""


# ===========================================================================
# DatabaseAccessor -- thread-safe SQLite wrapper (C1 fix)
# ===========================================================================


class DatabaseAccessor:
    """Thread-safe SQLite accessor with consistent locking.

    Wraps all connection access behind a single lock to prevent concurrent
    write corruption and lock-order inversions (C1, C3 fixes).
    """

    def __init__(self, conn, lock):
        self._conn = conn
        self._lock = lock

    def execute(self, sql, params=()):
        with self._lock:
            return self._conn.execute(sql, params)

    def execute_many(self, sql, params_seq):
        with self._lock:
            self._conn.executemany(sql, params_seq)

    def transaction(self, operations):
        """operations: list of (sql, params) tuples. All-or-nothing."""
        with self._lock:
            try:
                for sql, params in operations:
                    self._conn.execute(sql, params)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def commit(self):
        with self._lock:
            self._conn.commit()


# ===========================================================================
# PipelineState -- persistent storage
# ===========================================================================

class PipelineState:
    """Persistent state for the memory pipeline (pipeline_state.db).

    Design: single connection + threading.RLock, WAL mode, independent
    from any provider's database.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "pipeline_state.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=10.0,
        )
        self._lock = threading.RLock()
        self._conn.row_factory = sqlite3.Row
        self.db = DatabaseAccessor(self._conn, self._lock)
        self._init_tables()

    def _init_tables(self) -> None:
        from hermes_state import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="pipeline_state.db")
        with self._lock:
            self._conn.executescript(_PIPELINE_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception as e:
            logger.debug("PipelineState close failed: %s", e)


# ===========================================================================
# PipelineErrorHandler -- consecutive failure tracking, auto-disable
# ===========================================================================

class PipelineErrorHandler:
    """Tracks consecutive failures and auto-disables after threshold.

    Provides level-based logging:
    - WARNING for database errors (transient, may self-resolve)
    - ERROR for resource errors (persistent, needs intervention)

    After 10 consecutive failures, returns should_disable() = True so
    the caller can gracefully degrade.
    """

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(self) -> None:
        self._consecutive_failures: int = 0
        self._disabled: bool = False
        self._lock = threading.Lock()

    def record_failure(self, error: Exception, category: str = "db") -> None:
        """Record a failure. category='db' logs WARNING, other logs ERROR."""
        with self._lock:
            self._consecutive_failures += 1
            if category == "db":
                logger.warning(
                    "Pipeline %s failure (%d/%d): %s",
                    category, self._consecutive_failures,
                    self._MAX_CONSECUTIVE_FAILURES, error,
                )
            else:
                logger.error(
                    "Pipeline %s failure (%d/%d): %s",
                    category, self._consecutive_failures,
                    self._MAX_CONSECUTIVE_FAILURES, error,
                )
            if self._consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
                self._disabled = True
                logger.error(
                    "Pipeline auto-disabled after %d consecutive failures",
                    self._consecutive_failures,
                )

    def record_success(self) -> None:
        """Reset consecutive failure counter on success."""
        with self._lock:
            self._consecutive_failures = 0

    def should_disable(self) -> bool:
        """True if the pipeline should be disabled due to repeated failures."""
        with self._lock:
            return self._disabled

    def reset(self) -> None:
        """Manual reset (e.g., after operator intervention)."""
        with self._lock:
            self._consecutive_failures = 0
            self._disabled = False
