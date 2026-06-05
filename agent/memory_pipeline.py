"""MemoryPipeline -- organic memory infrastructure inside MemoryManager.

NOT a MemoryProvider.  No name, no tools, no system_prompt_block.
Pure interceptor: executes organic logic before/after MemoryManager lifecycle methods.

All methods are best-effort: exceptions are caught and logged at debug level,
never blocking upstream providers.

Design philosophy (浑然一体):
    Memory's organic properties (salience gating, silent engrams, consolidation,
    reconsolidation, predictive feedback, spreading activation) are infrastructure
    of the entire cognitive system -- not features of a specific storage backend.
    Just as synaptic plasticity is a universal property of neural circuits, not
    a "plugin" for the hippocampus, the MemoryPipeline lives inside MemoryManager
    and operates on ALL memory pathways regardless of which provider is active.

Architecture:
    MemoryManager
        └── MemoryPipeline (interceptor layer, NOT a provider)
            ├── SalienceScorer      (Layer 1: sensory gate)
            ├── SilentEngramEngine  (Layer 2: availability continuum)
            ├── ConsolidationEngine (Layer 3: sleep-like consolidation)
            ├── ReconsolidationEngine (Layer 4: prediction-error updates)
            ├── FeedbackCoordinator (Layer 5: predictive processing + learning)
            └── ActivationGraph     (Layer 6: spreading activation)
        └── providers[] (builtin + one external)

5 Architectural Invariants Preserved:
    1. MemoryProvider ABC contract unchanged
    2. Single external provider limit unchanged
    3. Tool registry unchanged (pipeline exposes no tools)
    4. ContextEngine orthogonality preserved
    5. run_agent.py integration points unchanged

Module split (8 pipeline submodules):
    agent.pipeline.state         -- PipelineState, DatabaseAccessor, PipelineErrorHandler
    agent.pipeline.salience      -- SalienceScorer, SalienceResult
    agent.pipeline.engram        -- SilentEngramEngine
    agent.pipeline.consolidation -- ConsolidationEngine, DeepConsolidationEngine
    agent.pipeline.reconsolidation -- ReconsolidationEngine, NLIDetector
    agent.pipeline.feedback      -- FeedbackCoordinator
    agent.pipeline.activation    -- ActivationGraph, EntityExtractor
    agent.pipeline.sleep         -- SleepScheduler
"""

from __future__ import annotations

import json
import logging
import re
import threading
from hashlib import sha256
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Feature flag gate
from agent.pipeline.feature_flags import FeatureFlags

# -- Re-export all public symbols from submodules for backward compatibility --
from agent.pipeline.state import (  # noqa: F401
    DatabaseAccessor,
    PipelineErrorHandler,
    PipelineState,
    _ZH_STOPWORDS,
)
from agent.pipeline.salience import (  # noqa: F401
    SalienceResult,
    SalienceScorer,
    _EMOTION_PATTERNS,
    _EMOTION_PATTERNS_ZH,
    _IMPORTANCE_PATTERNS,
    _IMPORTANCE_PATTERNS_ZH,
    _RECENCY_PATTERNS,
    _RECENCY_PATTERNS_ZH,
    _TRIVIAL_PATTERNS,
    _TRIVIAL_PATTERNS_ZH,
)
from agent.pipeline.engram import SilentEngramEngine, INITIAL_ENGRAM_STRENGTH, MemoryStrength  # noqa: F401
from agent.pipeline.consolidation import (  # noqa: F401
    ConsolidationEngine,
    DeepConsolidationEngine,
)
from agent.pipeline.reconsolidation import ReconsolidationEngine, NLIDetector  # noqa: F401
from agent.pipeline.feedback import FeedbackCoordinator  # noqa: F401
from agent.pipeline.activation import ActivationGraph, EntityExtractor  # noqa: F401
from agent.pipeline.sleep import SleepScheduler  # noqa: F401

# Backward-compatible re-export of the schema (used by tests)
from agent.pipeline.state import _PIPELINE_SCHEMA  # noqa: F401


# ===========================================================================
# Helper: salience-to-engram-strength mapping
# ===========================================================================

def _salience_to_engram_strength(salience: float) -> float:
    """Map a salience score to an initial engram strength.

    High-salience memories start at full strength, moderate ones at 0.7,
    and low-salience ones at 0.4.  This mirrors the allocation logic
    in ``MemoryPipeline._score_and_record_salience``.
    """
    if salience > 0.5:
        return 1.0
    elif salience > 0.2:
        return 0.7
    return 0.4


# ===========================================================================
# MemoryPipeline -- the interceptor layer
# ===========================================================================

class MemoryPipeline:
    """Organic memory pipeline -- internal infrastructure of MemoryManager.

    NOT a MemoryProvider.  Pure interceptor wrapping MemoryManager lifecycle.
    All methods best-effort: exceptions caught at debug level, never blocking.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict = config or {}
        self._state: PipelineState | None = None
        self._enabled: bool = self._config.get("enabled", True)
        self._session_id: str = ""
        # All 6 layers + episodic + dreaming
        self._salience: SalienceScorer | None = None
        self._engrams: SilentEngramEngine | None = None
        self._consolidation: ConsolidationEngine | None = None
        self._reconsolidation: ReconsolidationEngine | None = None
        self._feedback: FeedbackCoordinator | None = None
        self._activation: ActivationGraph | None = None
        self._episodic = None   # EpisodicTimeline (from holographic plugin)
        self._dreaming = None   # DreamEngine (from holographic plugin)
        self._evolution = None  # SelfEvolution (from holographic plugin)
        self._scheduler: SleepScheduler | None = None
        self._llm_client = None
        self._deep_consolidation: DeepConsolidationEngine | None = None
        self._background_threads: list[threading.Thread] = []

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize pipeline state and all organic modules."""
        if not self._enabled:
            return
        self._session_id = session_id
        db_path = self._config.get("db_path") or None
        self._background_threads: list[threading.Thread] = []

        # C2 fix: close connection on init failure (v2_concurrency)
        _ff = FeatureFlags()
        if _ff.is_enabled('v2_concurrency'):
            try:
                self._state = PipelineState(db_path=db_path)
                self._init_core_layers()
                self._init_plugin_layers()
            except Exception:
                if self._state is not None:
                    self._state.close()
                    self._state = None
                raise
        else:
            try:
                self._state = PipelineState(db_path=db_path)
                self._init_core_layers()
                self._init_plugin_layers()
            except Exception:
                if self._state is not None:
                    self._state.close()
                    self._state = None
                raise

        logger.debug("MemoryPipeline initialized (session=%s, layers=%d)",
                      session_id, sum(1 for x in [self._salience, self._engrams,
                      self._consolidation, self._reconsolidation,
                      self._feedback, self._activation,
                      self._episodic, self._dreaming,
                      self._scheduler] if x))

    def _init_core_layers(self) -> None:
        """Initialize layers 1-6 from config (salience through activation)."""
        # Layer 1: SalienceScorer
        sal_cfg = self._config.get("salience", {})
        if sal_cfg.get("enabled", True):
            self._salience = SalienceScorer(
                novelty_window=sal_cfg.get("novelty_window", 50))

        # Layer 2: SilentEngramEngine
        eng_cfg = self._config.get("silent_engram", {})
        if eng_cfg.get("enabled", True):
            self._engrams = SilentEngramEngine(
                half_life_hours=eng_cfg.get("half_life_hours", 720.0),
                emotion_modulated_decay_enabled=eng_cfg.get(
                    "emotion_modulated_decay_enabled", False),
                emotion_decay_multiplier=eng_cfg.get(
                    "emotion_decay_multiplier", 2.0))

        # Layer 3: ConsolidationEngine (base + optional deep)
        con_cfg = self._config.get("consolidation", {})
        if con_cfg.get("enabled", True):
            self._consolidation = ConsolidationEngine(
                min_facts=con_cfg.get("min_facts_for_consolidation", 5))
            if con_cfg.get("deep_consolidation_enabled", False):
                self._deep_consolidation = DeepConsolidationEngine(
                    llm_client=self._llm_client,
                    min_facts=con_cfg.get("min_facts_for_consolidation", 5))

        # Layer 4: ReconsolidationEngine
        rec_cfg = self._config.get("reconsolidation", {})
        if rec_cfg.get("enabled", True):
            self._reconsolidation = ReconsolidationEngine(
                error_threshold=rec_cfg.get("prediction_error_threshold", 0.3),
                semantic_conflict_enabled=rec_cfg.get(
                    "semantic_conflict_enabled", False),
                semantic_conflict_threshold=rec_cfg.get(
                    "semantic_conflict_threshold", 0.7))

        # Layer 5: FeedbackCoordinator
        if self._config.get("feedback", {}).get("enabled", True):
            self._feedback = FeedbackCoordinator()
            if self._reconsolidation:
                self._feedback._reconsolidation = self._reconsolidation

        # Layer 6: ActivationGraph
        act_cfg = self._config.get("activation", {})
        if act_cfg.get("enabled", True):
            self._activation = ActivationGraph(
                edge_decay_hours=act_cfg.get("edge_decay_hours", 168.0),
                pagerank_damping=act_cfg.get("pagerank_damping", 0.85),
                pagerank_max_iter=act_cfg.get("pagerank_max_iter", 20),
                pagerank_enabled=act_cfg.get("pagerank_enabled", False),
            )

    def _init_plugin_layers(self) -> None:
        """Initialize layers 7-9 from config (episodic, dreaming, hippocampal, sleep)."""
        # Layer 7: EpisodicTimeline (what-where-when binding)
        epi_cfg = self._config.get("episodic", {})
        if epi_cfg.get("enabled", False):
            _mod = self._load_holographic_plugin(
                "holographic_episodic", "episodic.py")
            if _mod:
                self._episodic = _mod.EpisodicTimeline(
                    self._state._conn, self._state._lock)
                self._episodic.init_tables()

        # Layer 8: DreamEngine (structured selective replay)
        dream_cfg = self._config.get("dreaming", {})
        if dream_cfg.get("enabled", False):
            _mod = self._load_holographic_plugin(
                "holographic_dreaming", "dreaming.py")
            if _mod:
                self._dreaming = _mod.DreamEngine(
                    self._state._conn, self._state._lock,
                    cooldown_hours=dream_cfg.get("cooldown_hours", 1.0),
                    mode1_top_k=dream_cfg.get("mode1_top_k", 10),
                    mode2_top_k=dream_cfg.get("mode2_top_k", 5),
                    mode3_idle_hours=dream_cfg.get("mode3_idle_hours", 24.0),
                    mode3_min_schema_conf=dream_cfg.get(
                        "mode3_min_schema_conf", 0.7),
                )
                self._dreaming.init_tables()

        # Layer 8b: SelfEvolution (homeostatic self-regulation)
        evo_cfg = self._config.get('self_evolution', {})
        if evo_cfg.get('self_evolution_enabled', False):
            _mod = self._load_holographic_plugin(
                'self_evolution', 'self_evolution.py')
            if _mod:
                self._evolution = _mod.SelfEvolution(
                    self._state._conn, self._state._lock,
                    evo_cfg,
                    pipeline_conn=self._state._conn,
                )
                self._evolution.init_tables()

        # Layer 9a: HippocampalIndex (sparse index for pattern completion)
        hippo_cfg = self._config.get("hippocampal_index", {})
        self._hippocampal = None
        if hippo_cfg.get("enabled", False):
            _mod = self._load_holographic_plugin(
                "hippocampal_index", "hippocampal_index.py")
            if _mod:
                self._hippocampal = _mod.HippocampalIndex(
                    self._state._conn, self._state._lock)
                self._hippocampal.init_tables()

        # Layer 9: SleepScheduler (automatic sleep-driven consolidation)
        sleep_cfg = self._config.get("sleep", {})
        if sleep_cfg.get("enabled", False):
            self._scheduler = SleepScheduler(
                idle_threshold_minutes=sleep_cfg.get(
                    "idle_minutes", 5.0),
                salience_threshold=sleep_cfg.get(
                    "salience_threshold", 10.0),
            )
            self._scheduler._state = self._state
            self._scheduler._session_id = self._session_id

    def shutdown(self) -> None:
        """Flush and close pipeline state."""
        # H6 fix: join background threads before closing state (v2_concurrency)
        _ff = FeatureFlags()
        if _ff.is_enabled('v2_concurrency'):
            for t in getattr(self, '_background_threads', []):
                t.join(timeout=5.0)
        else:
            for t in getattr(self, '_background_threads', []):
                t.join(timeout=5.0)
        if self._state is not None:
            self._state.close()
            self._state = None
        logger.debug("MemoryPipeline shut down")

    # -- Shared helpers --

    def _load_holographic_plugin(self, module_name: str, file_name: str) -> 'Any | None':
        """Dynamically load a module from the holographic plugin directory.

        Args:
            module_name: Name to register the module as.
            file_name: Python file name (e.g. "episodic.py").

        Returns:
            The loaded module, or None on failure.
        """
        try:
            import importlib.util
            plugin_dir = (Path(__file__).resolve().parent.parent
                          / "plugins" / "memory" / "holographic")
            _spec = importlib.util.spec_from_file_location(
                module_name, str(plugin_dir / file_name))
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            return _mod
        except Exception as e:
            logger.debug("Plugin %s load failed: %s", file_name, e)
            return None

    def _score_and_record_salience(
        self, content: str, provider_tag: str = "salience_init",
    ) -> 'SalienceResult | None':
        """Score content for salience and record engram strength.

        Computes the salience score, then initializes or upgrades the
        engram_strengths row for the content's memory_ref so that
        high-salience content starts strong and low-salience content
        starts modestly.

        Args:
            content: Text to score.
            provider_tag: Value for the ``provider`` column in
                engram_strengths (used for provenance tracking).

        Returns:
            The SalienceResult, or None if scoring is unavailable.
        """
        if not self._salience or not self._state:
            return None
        try:
            result = self._salience.score(content)
            if self._engrams:
                init_str = _salience_to_engram_strength(result.overall)
                ref = sha256(content.encode()).hexdigest()[:16]
                with self._state._lock:
                    self._state._conn.execute(
                        "INSERT INTO engram_strengths "
                        "(memory_ref, provider, strength) "
                        "VALUES (?, ?, ?) "
                        "ON CONFLICT(memory_ref) DO UPDATE SET "
                        "strength = MAX(strength, excluded.strength), "
                        "last_accessed = CURRENT_TIMESTAMP",
                        (ref, provider_tag, init_str),
                    )
                    self._state._conn.commit()
            return result
        except Exception as e:
            logger.debug("_score_and_record_salience failed: %s", e)
            return None

    def _record_salience_log(self, result: 'SalienceResult', source: str = "builtin") -> None:
        """Persist a salience score to the encoding log table."""
        if not self._state:
            return
        try:
            with self._state._lock:
                self._state._conn.execute(
                    "INSERT INTO salience_encoding_log "
                    "(source, emotion_score, novelty_score, "
                    "importance_score, overall_score) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (source, result.emotion, result.novelty,
                     result.importance, result.overall),
                )
                self._state._conn.commit()
        except Exception as e:
            logger.debug("Salience log insert failed: %s", e)

    def _update_salience_weights(self, error_score: float) -> None:
        """Adjust salience signal weights after a conflict or feedback event.

        High error nudges weights down; low error nudges them up.
        """
        if not self._state:
            return
        try:
            adj = -0.02 if error_score > 0.5 else 0.01
            for sig in ("emotion", "novelty", "importance"):
                with self._state._lock:
                    self._state._conn.execute(
                        "INSERT INTO salience_weights "
                        "(signal_type, weight, "
                        "sample_count, success_count) "
                        "VALUES (?, 0.5, 1, ?) "
                        "ON CONFLICT(signal_type) "
                        "DO UPDATE SET "
                        "weight = MAX(0.1, "
                        "  MIN(1.0, weight + ?)), "
                        "sample_count = sample_count + 1, "
                        "success_count = success_count + ?, "
                        "updated_at = CURRENT_TIMESTAMP",
                        (sig,
                         1 if adj > 0 else 0,
                         adj,
                         1 if adj > 0 else 0),
                    )
                    self._state._conn.commit()
        except Exception as e:
            logger.debug("Salience weight update failed: %s", e)

    def _extract_entities_from_text(self, text: str) -> list[str]:
        """Extract capitalized and Chinese entities from text."""
        entities = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        zh_entities = [
            e for e in re.findall(r'[一-鿿]{2,6}', text)
            if e not in _ZH_STOPWORDS
        ]
        entities.extend(zh_entities)
        return entities

    # -- Pre-interceptors --

    def pre_turn_start(self, turn: int, message: str) -> None:
        """Reset salience novelty window periodically, decay activation edges."""
        if self._salience and turn > 0 and turn % 100 == 0:
            try:
                self._salience.reset()
            except Exception as e:
                logger.debug("SalienceScorer reset failed: %s", e)
        if self._activation and self._state:
            try:
                self._activation.decay_edges(self._state, hours_elapsed=0.1)
            except Exception as e:
                logger.debug("Activation decay failed: %s", e)

    def pre_sync(self, user: str, asst: str,
                 embed_fn: 'Any | None' = None,
                 llm_client: 'Any | None' = None) -> dict | None:
        """Score user content for salience, persist signals.

        When semantic conflict detection is enabled in the reconsolidation
        config, this method also runs semantic conflict detection against
        existing schemas before the sync proceeds.

        Args:
            user: The user message content.
            asst: The assistant message content.
            embed_fn: Optional callable(text) -> list[float] for embeddings.
            llm_client: Optional LLM client with .complete(prompt) method.

        Returns:
            Metadata dict including salience scores and, when applicable,
            semantic conflict results, or None on failure.
        """
        if not self._salience:
            return None
        # Capture llm_client for deep consolidation
        if llm_client is not None:
            self._llm_client = llm_client
            if self._deep_consolidation:
                self._deep_consolidation._llm = llm_client
        try:
            # Score salience and initialize engram strength (consolidated)
            result = self._score_and_record_salience(user, "salience_init")
            if result is None:
                return None
            meta: dict = {
                "salience_overall": result.overall,
                "salience_emotion": result.emotion,
                "salience_novelty": result.novelty,
                "salience_importance": result.importance,
                "salience_is_trivial": result.is_trivial,
                "salience_temporal_recency_boost": result.temporal_recency_boost,
            }
            # --- Sleep scheduler: accumulate salience, maybe trigger sleep ---
            if self._scheduler:
                try:
                    self._scheduler.on_message(result.overall)
                    if self._scheduler.should_sleep():
                        import threading as _slp_thread
                        _slp_thread.Thread(
                            target=self._scheduler.sleep_cycle,
                            args=(self._consolidation, self._dreaming),
                            daemon=True,
                        ).start()
                except Exception as e:
                    logger.debug("SleepScheduler failed: %s", e)

            # Emotion-modulated engram decay
            if self._engrams and self._state:
                try:
                    _ff8b = FeatureFlags()
                    if _ff8b.is_enabled('v2_emotion_decay'):
                        # FIX 8 (M2): Use per-memory emotion, not global
                        with self._state._lock:
                            _erows = self._state._conn.execute(
                                "SELECT memory_ref, emotional_valence "
                                "FROM engram_strengths "
                                "WHERE emotional_valence != 0.0"
                            ).fetchall()
                        _valences = {
                            r["memory_ref"]: r["emotional_valence"]
                            for r in _erows
                        }
                        decay_affected = self._engrams.apply_decay_with_emotion(
                            self._state,
                            hours_elapsed=1.0,
                            emotional_valences=_valences,
                        )
                    else:
                        decay_affected = self._engrams.apply_decay(
                            self._state,
                            hours_elapsed=1.0,
                            emotional_valence=result.emotion,
                        )
                    meta["decay_affected"] = decay_affected
                except Exception as e:
                    logger.debug("Emotion-modulated engram decay failed: %s", e)

            # Activation expansion
            if self._activation and self._state:
                try:
                    expansions = self._activation.expand_query(
                        self._state, user)
                    if expansions:
                        meta["activation_expansions"] = expansions
                except Exception as e:
                    logger.debug("Activation query expansion failed: %s", e)

            self._record_salience_log(result)

            # --- Semantic conflict detection ---
            self._detect_and_log_conflict(user, meta, embed_fn, llm_client)

            return meta
        except Exception as e:
            logger.debug("SalienceScorer.score failed: %s", e)
            return None

    def _detect_and_log_conflict(
        self, user: str, meta: dict,
        embed_fn: 'Any | None', llm_client: 'Any | None',
    ) -> None:
        """Run semantic conflict detection and log any detected conflict."""
        if not (self._reconsolidation
                and self._reconsolidation._semantic_enabled
                and self._state):
            return
        try:
            with self._state._lock:
                rows = self._state._conn.execute(
                    "SELECT content FROM schemas "
                    "ORDER BY confidence DESC LIMIT 20"
                ).fetchall()
            existing_contents = [r["content"] for r in rows if r["content"]]
            if not existing_contents:
                return

            error_score, action = self._reconsolidation.detect_semantic_conflict(
                user, existing_contents,
                embed_fn=embed_fn, llm_client=llm_client)
            meta["semantic_conflict_score"] = error_score
            meta["semantic_conflict_action"] = action

            if action != "no_conflict" and error_score > 0.2:
                with self._state._lock:
                    self._state._conn.execute(
                        "INSERT INTO reconsolidation_log "
                        "(memory_ref, old_content, new_content, prediction_error) "
                        "VALUES (?, ?, ?, ?)",
                        ("pre_sync",
                         existing_contents[0][:500],
                         user[:500],
                         error_score),
                    )
                    self._state._conn.commit()
                logger.debug(
                    "Semantic conflict detected: action=%s, score=%.2f",
                    action, error_score)

            self._update_salience_weights(error_score)
        except Exception as e:
            logger.debug("Semantic conflict detection failed: %s", e)

    def pre_memory_write(
        self, action: str, target: str, content: str, metadata: dict
    ) -> dict | None:
        """Salience gate -- score content, attach metadata."""
        if not self._salience or action not in ("add", "replace"):
            return None
        try:
            result = self._salience.score(content)
            if self._state:
                with self._state._lock:
                    self._state._conn.execute(
                        "INSERT INTO salience_encoding_log "
                        "(source, fact_ref, emotion_score, novelty_score, "
                        "importance_score, overall_score) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (target, content[:100], result.emotion, result.novelty,
                         result.importance, result.overall),
                    )
                    self._state._conn.commit()
            return {
                **metadata,
                "pipeline_salience": result.overall,
                "pipeline_emotion": result.emotion,
                "pipeline_novelty": result.novelty,
                "pipeline_importance": result.importance,
                "pipeline_temporal_recency_boost": result.temporal_recency_boost,
            }
        except Exception as e:
            logger.debug("SalienceScorer pre_memory_write failed: %s", e)
            return None

    def pre_compress(self, messages: list) -> str:
        """Extract key insights before context compression."""
        if not self._consolidation:
            return ""
        try:
            return self._consolidation.extract_insights(messages)
        except Exception as e:
            logger.debug("Consolidation extract_insights failed: %s", e)
            return ""

    # -- Post-interceptors --

    def post_prefetch(self, query: str, provider_results: list[str]) -> str:
        """Augment prefetch with predictions and spreading activation."""
        parts = []
        try:
            # Layer 5: predictions from schemas
            if self._feedback and self._state:
                predictions = self._feedback.predict(self._state, query)
                for pred in predictions:
                    parts.append(pred)

            # Layer 6: spreading activation
            if self._activation and self._state:
                expansions = self._activation.expand_query(self._state, query)
                parts.extend(expansions)
        except Exception as e:
            logger.debug("Pipeline post_prefetch failed: %s", e)
        return "\n".join(parts)

    def post_tool_call(self, name: str, args: dict, result: str) -> None:
        """Record retrieval for reconsolidation, co-activation."""
        if not self._state:
            return
        try:
            # Layer 4: reconsolidation check
            if self._reconsolidation and name == "fact_store":
                action = args.get("action", "")
                if action in ("search", "probe"):
                    self._reconsolidation.check_retrieval(
                        self._state, args.get("query", ""), result,
                        engrams=self._engrams)

            # Layer 6: record co-activation from search results
            if self._activation and name == "fact_store":
                query = args.get("query", "")
                entities = self._extract_entities_from_text(query)
                if len(entities) >= 2:
                    self._activation.record_co_activation(self._state, entities)

            # GAP 1: record co-activation from result content entities
            if self._activation and name == "fact_store":
                try:
                    result_entities = self._extract_entities_from_text(result)
                    if len(result_entities) >= 2:
                        self._activation.record_co_activation(
                            self._state, result_entities, delta=0.05)
                except Exception as e:
                    logger.debug(
                        "post_tool_call result co-activation "
                        "failed: %s", e)
            # GAP 3: 'add' action
            if name == "fact_store" and args.get("action") == "add":
                self._handle_fact_add(args, result)

            # GAP 5: fact_feedback
            if name == "fact_feedback":
                self._handle_fact_feedback(args)
        except Exception as e:
            logger.debug("Pipeline post_tool_call failed: %s", e)

    def _handle_fact_add(self, args: dict, result: str) -> None:
        """Process fact_store add action: salience, engram, co-activation, episodic."""
        content = args.get('content', '')
        if not content:
            return
        # 1. Score salience and init engram (consolidated)
        self._score_and_record_salience(content, 'add_action')

        # 2. Record co-activation of entities from content
        if self._activation:
            try:
                ents = self._extract_entities_from_text(content)
                if len(ents) >= 2:
                    self._activation.record_co_activation(
                        self._state, ents)
            except Exception as e:
                logger.debug('post_tool_call add co-activation failed: %s', e)

        # 3. Parse fact_id from result
        fact_id = self._parse_fact_id(result)
        if fact_id is None:
            return

        # 4. Append to current episode
        if self._episodic:
            try:
                self._episodic.append_fact(fact_id)
            except Exception as e:
                logger.debug('post_tool_call add episodic append failed: %s', e)

        # 5. Hippocampal index
        if self._hippocampal:
            try:
                ents = self._extract_entities_from_text(content)
                self._hippocampal.index_memory(
                    str(fact_id), content, entities=ents)
            except Exception as e:
                logger.debug('post_tool_call add hippocampal index failed: %s', e)

    def _handle_fact_feedback(self, args: dict) -> None:
        """Process fact_feedback: record feedback and update salience weights."""
        action_val = args.get('action', '')
        was_helpful = 1 if action_val == 'helpful' else 0
        fact_id = args.get('fact_id', 0)
        memory_ref = str(fact_id)

        if not self._state:
            return
        with self._state._lock:
            self._state._conn.execute(
                "INSERT INTO salience_feedback "
                "(memory_ref, signal_type, "
                " signal_value, was_helpful, "
                " was_retrieved) "
                "VALUES (?, 'fact_feedback', 1.0, ?, 1)",
                (memory_ref, was_helpful),
            )
            self._state._conn.commit()

        adj = 0.02 if was_helpful else -0.02
        self._update_salience_weights(-adj)

    @staticmethod
    def _parse_fact_id(result: str) -> 'int | None':
        """Extract fact_id from a JSON tool result string."""
        try:
            parsed = json.loads(result)
            return parsed.get('fact_id')
        except (json.JSONDecodeError, TypeError):
            return None

    def post_session_end(self, messages: list) -> None:
        """Consolidation, engram decay, bridge discovery, dreaming."""
        if not self._state:
            return
        try:
            # Layer 2: apply engram decay (1 hour worth)
            if self._engrams:
                self._engrams.apply_decay(self._state, hours_elapsed=1.0)

            # Layer 3: run consolidation (deep if available, else base)
            _engine = (
                self._deep_consolidation
                if (self._deep_consolidation
                    and self._deep_consolidation._llm)
                else self._consolidation)
            if _engine:
                facts = []
                for msg in messages[-10:]:
                    content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    if content and len(content) > 20:
                        facts.append({"content": content, "domain": "general"})
                _engine.consolidate(self._state, facts)

            # Layer 5: discover cross-domain bridges
            if self._feedback:
                self._feedback.discover_bridges(self._state)

            # Layer 6: decay activation edges
            if self._activation:
                self._activation.decay_edges(self._state, hours_elapsed=1.0)

            # Layer 7: close episodic episode
            if self._episodic:
                try:
                    summary = f"Session {self._session_id}: {len(messages)} messages"
                    self._episodic.close_episode(summary=summary)
                    # Fix 7: Episodic to Consolidation: mini-consolidation
                    # on episode facts when episode closes
                    if self._consolidation:
                        try:
                            epi_facts = []
                            for msg in messages[-5:]:
                                c = (msg.get("content", "")
                                     if isinstance(msg, dict)
                                     else str(msg))
                                if c and len(c) > 15:
                                    epi_facts.append(
                                        {"content": c,
                                         "domain": "episode"})
                            min_req = max(
                                2,
                                self._consolidation._min_facts // 2)
                            if len(epi_facts) >= min_req:
                                self._consolidation.consolidate(
                                    self._state, facts=epi_facts)
                        except Exception as e:
                            logger.debug(
                                "Episodic mini-consolidation "
                                "failed: %s", e)
                except Exception as e:
                    logger.debug("Episodic close_episode failed: %s", e)

            # Layer 8: run dream cycle if conditions met
            if self._dreaming:
                try:
                    if self._dreaming.should_dream():
                        import threading as _t
                        _t.Thread(
                            target=self._run_dream_postprocessing,
                            daemon=True,
                        ).start()
                except Exception as e:
                    logger.debug("Dream cycle failed: %s", e)

            # Layer 8b: run self-evolution cycle if conditions met
            if self._evolution:
                import threading as _t2
                _t2.Thread(
                    target=self._evolution.run_evolution_cycle,
                    daemon=True,
                ).start()
        except Exception as e:
            logger.debug("Pipeline post_session_end failed: %s", e)

    def _run_dream_postprocessing(self) -> None:
        """Run dream cycle with post-processing (schema boost + predictions).

        Extracted from post_session_end for readability.  Runs in a
        daemon thread: boosts schema confidences after replay and adds
        dream hypotheses as pending predictions.
        """
        try:
            dr = self._dreaming.dream_cycle(self._session_id)
            if self._state:
                with self._state._lock:
                    self._state._conn.execute(
                        "UPDATE schemas SET "
                        "confidence = MIN(1.0, confidence + 0.02), "
                        "updated_at = CURRENT_TIMESTAMP "
                        "WHERE confidence > 0.5"
                    )
                    self._state._conn.commit()
            if self._feedback and dr and getattr(dr, "hypotheses", 0) > 0:
                try:
                    hyps = self._dreaming.get_hypotheses(limit=3)
                    if hyps:
                        with self._feedback._lock:
                            self._feedback._pending_predictions.extend(
                                [f"Dream: {h['content']}" for h in hyps])
                except Exception as e:
                    logger.debug("Dream prediction extension failed: %s", e)
        except Exception as e:
            logger.debug("Dream post-processing failed: %s", e)

    def post_session_switch(self, new_id: str, **kwargs) -> None:
        """Propagate session switch to pipeline internals.

        Updates the cached session_id, closes the current episodic episode
        and opens a new one, and resets the sleep scheduler so accumulated
        salience does not bleed across sessions.
        """
        old_id = self._session_id
        self._session_id = new_id

        # Close old episode, start new one
        if self._episodic:
            try:
                self._episodic.close_episode(
                    summary=f"Session {old_id} ended (switch)")
                self._episodic.start_episode(new_id)
            except Exception as e:
                logger.debug("Episode switch failed: %s", e)

        # Update sleep scheduler so salience accumulators reset
        if self._scheduler:
            try:
                self._scheduler._session_id = new_id
                self._scheduler.reset()
            except Exception as e:
                logger.debug("Scheduler reset failed: %s", e)

    def post_delegation(self, task: str, result: str, **kwargs) -> None:
        """No-op for now. Phase 2+: score subagent result."""
        pass

    def augment_system_prompt(self) -> str:
        """Inject organic memory status into system prompt."""
        if not self._state:
            return ""
        try:
            with self._state._lock:
                engram_count = self._state._conn.execute(
                    "SELECT COUNT(*) FROM engram_strengths"
                ).fetchone()[0]
                schema_count = self._state._conn.execute(
                    "SELECT COUNT(*) FROM schemas"
                ).fetchone()[0]
                edge_count = self._state._conn.execute(
                    "SELECT COUNT(*) FROM activation_edges"
                ).fetchone()[0]
            if engram_count == 0 and schema_count == 0:
                return ""
            return (
                f"[Organic Memory: {engram_count} engrams, "
                f"{schema_count} schemas, {edge_count} activation edges]"
            )
        except Exception as e:
            logger.debug("augment_system_prompt failed: %s", e)
            return ""


# ===========================================================================
# Config loader
# ===========================================================================

def _load_pipeline_config() -> dict:
    """Load memory.pipeline config from $HERMES_HOME/config.yaml."""
    try:
        from hermes_cli.config import cfg_get, load_config
        config = load_config()
        return cfg_get(config, "memory", "pipeline", default={}) or {}
    except Exception as e:
        logger.debug("Failed to load pipeline config: %s", e)
        return {}


# ===========================================================================
# Public API (backward-compatible __all__)
# ===========================================================================

__all__ = [
    'MemoryPipeline',
    'PipelineState',
    'DatabaseAccessor',
    'PipelineErrorHandler',
    'SalienceScorer',
    'SalienceResult',
    'SilentEngramEngine',
    'INITIAL_ENGRAM_STRENGTH',
    'MemoryStrength',
    'ConsolidationEngine',
    'DeepConsolidationEngine',
    'ReconsolidationEngine',
    'NLIDetector',
    'FeedbackCoordinator',
    'ActivationGraph',
    'EntityExtractor',
    'SleepScheduler',
    '_salience_to_engram_strength',
    '_load_pipeline_config',
    '_PIPELINE_SCHEMA',
    '_ZH_STOPWORDS',
    'FeatureFlags',
]
