"""ActivationGraph -- Layer 6 spreading activation.

Hebbian co-activation graph for spreading activation. When entities are
co-retrieved, their connection strengthens. Activation spreads through
the graph to pre-activate related memories.

Scientific basis: Collins & Loftus (1975) spreading activation.

Contains:
- ActivationGraph class
- EntityExtractor class (new, gated behind v2_ner)
- _ZH_STOPWORDS constant (re-exported from state)
- All v2_activation fixes (inverse weight for shortest path)
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from agent.pipeline.feature_flags import FeatureFlags
from agent.pipeline.state import _ZH_STOPWORDS

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Named entity extractor for activation graph queries.

    Extraction rules (applied in order):
      1. Acronyms: [A-Z]{2,10}  (e.g. NASA, GPU, API)
      2. Capitalized names: [A-Z][a-z]+  (e.g. Python, London)
      3. Quoted text: "..." or "..." or '...'
      4. Chinese entities: CJK multi-char sequences, minus stopwords
      5. spaCy NER (when available and v2_ner enabled): PERSON, ORG, GPE,
         PRODUCT, WORK_OF_ART, LOC, EVENT

    Gated behind the v2_ner feature flag.  When the flag is off, callers
    should fall back to the inline regex extraction in
    ActivationGraph.expand_query().
    """

    def __init__(self) -> None:
        self._zh_stopwords = _ZH_STOPWORDS
        self._spacy_nlp = None
        self._spacy_loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> list[str]:
        """Extract entities from text.

        Returns a deduplicated list of entity strings.  When the v2_ner
        flag is enabled and spaCy is available, NER results are merged
        with the rule-based extraction.
        """
        _ff = FeatureFlags()
        if not _ff.is_enabled('v2_ner'):
            return self._simple_extract(text)

        # Rule-based extraction (always runs)
        entities = self._rule_based_extract(text)

        # spaCy NER (if available)
        spacy_entities = self._spacy_extract(text)
        entities.extend(spacy_entities)

        # Deduplicate preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                deduped.append(e)
        return deduped

    # ------------------------------------------------------------------
    # Rule-based extraction
    # ------------------------------------------------------------------

    def _simple_extract(self, text: str) -> list[str]:
        """Legacy simple extraction: capitalized words + Chinese entities."""
        entities = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        zh_entities = [
            e for e in re.findall(r'[一-鿿]{2,6}', text)
            if e not in self._zh_stopwords
        ]
        entities.extend(zh_entities)
        return entities

    def _rule_based_extract(self, text: str) -> list[str]:
        """Enhanced rule-based entity extraction.

        Extracts:
          - Acronyms: [A-Z]{2,10} (NASA, GPU, API)
          - Capitalized names: [A-Z][a-z]+ (Python, London)
          - Quoted text: "..." or "..." or '...'
          - Chinese entities: CJK multi-char sequences, minus stopwords
        """
        entities: list[str] = []

        # 1. Acronyms: 2-10 uppercase letters (must be word-bounded)
        acronyms = re.findall(r'\b([A-Z]{2,10})\b', text)
        entities.extend(acronyms)

        # 2. Capitalized names (single or multi-word)
        names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        entities.extend(names)

        # 3. Quoted text
        quoted = re.findall(r'[""“]([^""”]+)[""”]', text)
        quoted += re.findall(r"'([^']+)'", text)
        for q in quoted:
            stripped = q.strip()
            if len(stripped) >= 2:
                entities.append(stripped)

        # 4. Chinese entities (CJK multi-char, minus stopwords)
        zh_entities = [
            e for e in re.findall(r'[一-鿿]{2,6}', text)
            if e not in self._zh_stopwords
        ]
        entities.extend(zh_entities)

        return entities

    # ------------------------------------------------------------------
    # spaCy NER (lazy loading)
    # ------------------------------------------------------------------

    def _ensure_spacy(self) -> None:
        """Lazily load the spaCy model on first use."""
        if self._spacy_loaded:
            return
        self._spacy_loaded = True
        try:
            import spacy
            # Try small model first, fall back to any available
            for model_name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
                try:
                    self._spacy_nlp = spacy.load(model_name)
                    logger.debug("Loaded spaCy model: %s", model_name)
                    return
                except OSError:
                    continue
            # If no model found, try blank English pipeline
            self._spacy_nlp = spacy.blank("en")
            logger.debug("Using blank spaCy English pipeline")
        except ImportError:
            logger.debug("spaCy not installed; NER unavailable.")
            self._spacy_nlp = None

    # Entity types we want from spaCy NER
    _SPACY_ENTITY_TYPES = frozenset({
        "PERSON", "ORG", "GPE", "PRODUCT",
        "WORK_OF_ART", "LOC", "EVENT",
    })

    def _spacy_extract(self, text: str) -> list[str]:
        """Extract named entities using spaCy NER."""
        self._ensure_spacy()
        if self._spacy_nlp is None:
            return []
        try:
            doc = self._spacy_nlp(text)
            entities: list[str] = []
            for ent in doc.ents:
                if ent.label_ in self._SPACY_ENTITY_TYPES:
                    cleaned = ent.text.strip()
                    if cleaned and len(cleaned) >= 2:
                        entities.append(cleaned)
            return entities
        except Exception as e:
            logger.debug("spaCy NER extraction failed: %s", e)
            return []


class ActivationGraph:
    """Hebbian co-activation graph for spreading activation.

    When entities are co-retrieved, their connection strengthens.
    Activation spreads through the graph to pre-activate related memories.
    Scientific basis: Collins & Loftus (1975) spreading activation.
    """

    def __init__(self, edge_decay_hours: float = 168.0,
                 pagerank_damping: float = 0.85,
                 pagerank_max_iter: int = 20,
                 pagerank_enabled: bool = False) -> None:
        self._decay_hours = edge_decay_hours
        self._pr_damping = pagerank_damping
        self._pr_max_iter = pagerank_max_iter
        self._pr_enabled = pagerank_enabled

    def record_co_activation(self, state: 'PipelineState',
                             entities: list[str], delta: float = 0.1) -> None:
        """Strengthen edges between co-activated entities (Hebbian learning)."""
        if not state or len(entities) < 2:
            return
        try:
            with state._lock:
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        a, b = sorted([entities[i], entities[j]])
                        state._conn.execute(
                            "INSERT INTO activation_edges "
                            "(source_entity, target_entity, strength, co_activation_count) "
                            "VALUES (?, ?, ?, 1) "
                            "ON CONFLICT(source_entity, target_entity) DO UPDATE SET "
                            "strength = MIN(1.0, strength + ?), "
                            "co_activation_count = co_activation_count + 1, "
                            "last_activated = CURRENT_TIMESTAMP",
                            (a, b, delta, delta),
                        )
                state._conn.commit()
        except Exception as e:
            logger.debug("Co-activation recording failed: %s", e)

    def get_neighbors(self, state: 'PipelineState',
                      entity: str, min_strength: float = 0.3,
                      limit: int = 5) -> list[dict]:
        """Get strongly connected neighbors of an entity."""
        if not state:
            return []
        try:
            with state._lock:
                rows = state._conn.execute(
                    "SELECT target_entity AS neighbor, strength FROM activation_edges "
                    "WHERE source_entity = ? AND strength >= ? "
                    "UNION ALL "
                    "SELECT source_entity AS neighbor, strength FROM activation_edges "
                    "WHERE target_entity = ? AND strength >= ? "
                    "ORDER BY strength DESC LIMIT ?",
                    (entity, min_strength, entity, min_strength, limit),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.debug("Get neighbors failed: %s", e)
            return []

    def expand_query(self, state: 'PipelineState',
                     query: str, limit: int = 3) -> list[str]:
        """Expand a query using spreading activation.

        Extracts entities from query, finds their neighbors, returns
        additional context strings.  Uses EntityExtractor when v2_ner
        is enabled, otherwise falls back to inline regex.
        """
        if not state:
            return []
        try:
            _ff = FeatureFlags()
            if _ff.is_enabled('v2_ner'):
                extractor = EntityExtractor()
                entities = extractor.extract(query)
            else:
                entities = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
                zh_entities = [e for e in re.findall(r'[一-鿿]{2,6}', query)
                               if e not in _ZH_STOPWORDS]
                entities.extend(zh_entities)
            expansions = []
            for entity in entities[:3]:
                neighbors = self.get_neighbors(state, entity, limit=limit)
                for n in neighbors:
                    expansions.append(
                        f"[co-activated: {entity} → {n['neighbor']} "
                        f"(strength={n['strength']:.2f})]"
                    )
            return expansions
        except Exception as e:
            logger.debug("Query expansion failed: %s", e)
            return []

    def spread_activation_pagerank(
        self, state: 'PipelineState',
        seed_entities: list[str],
        damping: float | None = None,
        max_iter: int | None = None,
    ) -> dict[str, float]:
        """Personalized PageRank spreading activation over the co-activation graph.

        Builds a NetworkX graph from activation_edges, runs personalized
        PageRank with *seed_entities* as the personalisation vector, and
        returns an entity->score mapping (seeds excluded).

        Requires ``networkx``; returns an empty dict when the library is
        absent so callers degrade gracefully.
        """
        if not state or not seed_entities:
            return {}
        try:
            import networkx as nx
        except ImportError:
            logger.debug(
                "networkx not installed -- PageRank spreading activation "
                "unavailable; falling back to direct neighbors.")
            return {}

        d = damping if damping is not None else self._pr_damping
        iters = max_iter if max_iter is not None else self._pr_max_iter
        seed_set = set(seed_entities)

        # --- Build graph from DB ---
        try:
            with state._lock:
                rows = state._conn.execute(
                    "SELECT source_entity, target_entity, strength "
                    "FROM activation_edges WHERE strength > 0.01",
                ).fetchall()
        except Exception as e:
            logger.debug("PageRank graph load failed: %s", e)
            return {}

        if not rows:
            return {}

        G: nx.Graph = nx.Graph()
        for row in rows:
            G.add_edge(
                row["source_entity"], row["target_entity"],
                weight=float(row["strength"]),
            )

        # Ensure seeds present (even if isolated)
        for s in seed_entities:
            if s not in G:
                G.add_node(s)

        # Personalisation vector: equal weight on seeds, 0 elsewhere
        personalization = {n: (1.0 if n in seed_set else 0.0) for n in G}

        try:
            scores = nx.pagerank(
                G, alpha=damping, max_iter=iters,
                personalization=personalization, weight="weight",
            )
        except Exception as e:
            logger.debug("PageRank computation failed: %s", e)
            return {}

        # Exclude seeds, sort descending
        return {
            ent: score
            for ent, score in sorted(
                scores.items(), key=lambda kv: kv[1], reverse=True
            )
            if ent not in seed_set
        }

    def expand_query_deep(self, state: 'PipelineState',
                          query: str, limit: int = 5) -> list[str]:
        """Expand query using Personalized PageRank instead of direct neighbours.

        Extracts entities from the query, runs PageRank spreading
        activation, and returns formatted expansion strings for the
        top-*limit* scored entities.

        Falls back to ``expand_query`` when PageRank is disabled or
        ``networkx`` is unavailable.
        """
        if not state:
            return []
        if not self._pr_enabled:
            return self.expand_query(state, query, limit=limit)
        try:
            _ff_ner = FeatureFlags()
            if _ff_ner.is_enabled('v2_ner'):
                extractor = EntityExtractor()
                entities = extractor.extract(query)
            else:
                entities = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
                zh_entities = [e for e in re.findall(r'[一-鿿]{2,6}', query)
                               if e not in _ZH_STOPWORDS]
                entities.extend(zh_entities)
            if not entities:
                return []

            scores = self.spread_activation_pagerank(
                state, entities[:5])
            if not scores:
                return self.expand_query(state, query, limit=limit)

            expansions: list[str] = []
            for ent, score in list(scores.items())[:limit]:
                expansions.append(
                    f"[pagerank: {ent} (score={score:.4f})]"
                )
            return expansions
        except Exception as e:
            logger.debug("Deep query expansion failed: %s", e)
            return self.expand_query(state, query, limit=limit)

    def find_bridge_entities(self, state: 'PipelineState',
                             entity_a: str,
                             entity_b: str) -> list[str]:
        """Discover bridge entities on the shortest path between two nodes.

        Returns the intermediate entities (excluding *entity_a* and
        *entity_b*) along the shortest weighted path in the
        co-activation graph.  Requires ``networkx``.

        Returns an empty list when no path exists or the library is
        missing.
        """
        if not state or not entity_a or not entity_b:
            return []
        try:
            import networkx as nx
        except ImportError:
            logger.debug(
                "networkx not installed -- bridge entity discovery "
                "unavailable.")
            return []

        try:
            with state._lock:
                rows = state._conn.execute(
                    "SELECT source_entity, target_entity, strength "
                    "FROM activation_edges WHERE strength > 0.01",
                ).fetchall()
        except Exception as e:
            logger.debug("Bridge entity graph load failed: %s", e)
            return []

        if not rows:
            return []

        G: nx.Graph = nx.Graph()
        _ff4 = FeatureFlags()
        for row in rows:
            _str = float(row["strength"])
            if _ff4.is_enabled('v2_activation'):
                # FIX 4 (H4): Inverse weight so shortest_path finds strongest path
                G.add_edge(
                    row["source_entity"], row["target_entity"],
                    weight=1.0 / max(_str, 0.01),
                    strength=_str,
                )
            else:
                G.add_edge(
                    row["source_entity"], row["target_entity"],
                    weight=_str,
                )

        if entity_a not in G or entity_b not in G:
            return []

        try:
            path = nx.shortest_path(
                G, source=entity_a, target=entity_b, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        except Exception as e:
            logger.debug("Shortest-path computation failed: %s", e)
            return []

        # Exclude endpoints
        return [node for node in path if node not in (entity_a, entity_b)]

    def decay_edges(self, state: 'PipelineState',
                    hours_elapsed: float = 1.0) -> int:
        """Decay all edge strengths. Returns affected rows."""
        if not state:
            return 0
        try:
            decay_factor = 0.5 ** (hours_elapsed / self._decay_hours)
            with state._lock:
                cursor = state._conn.execute(
                    "UPDATE activation_edges SET "
                    "strength = MAX(0.01, strength * ?) "
                    "WHERE strength > 0.01",
                    (decay_factor,),
                )
                state._conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.debug("Edge decay failed: %s", e)
            return 0

    # ------------------------------------------------------------------
    # PMI-weighted co-activation (gated behind v2_pmi)
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid activation: 1 / (1 + e^(-x))."""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)

    def update_coactivation(self, state: 'PipelineState',
                            entity_a: str, entity_b: str) -> float:
        """Record a co-occurrence and return the PMI-based edge weight.

        Gated behind ``v2_pmi``.  When the flag is off, returns 0.0 and
        makes no database changes.

        Steps:
        1. Canonicalise: sort pair so (a, b) always has a < b.
        2. Increment entity_frequencies for both entities.
        3. Increment cooccurrence_frequencies for the pair.
        4. Compute PMI = log2(P(a,b) / (P(a) * P(b))).
        5. Edge weight = sigmoid(PMI).
        6. Upsert into activation_edges with the PMI-derived weight.

        Returns
        -------
        float
            The sigmoid(PMI) edge weight, or 0.0 if the flag is off or
            computation fails.
        """
        _ff = FeatureFlags()
        if not _ff.is_enabled('v2_pmi'):
            return 0.0

        if not state or not entity_a or not entity_b:
            return 0.0

        # Canonical ordering for undirected pair
        a, b = sorted([entity_a, entity_b])

        try:
            with state._lock:
                # 1. Increment single-entity frequencies
                for e in (a, b):
                    state._conn.execute(
                        "INSERT INTO entity_frequencies (entity, count, last_seen) "
                        "VALUES (?, 1, CURRENT_TIMESTAMP) "
                        "ON CONFLICT(entity) DO UPDATE SET "
                        "count = count + 1, last_seen = CURRENT_TIMESTAMP",
                        (e,),
                    )

                # 2. Increment co-occurrence
                state._conn.execute(
                    "INSERT INTO cooccurrence_frequencies "
                    "(entity_a, entity_b, count) "
                    "VALUES (?, ?, 1) "
                    "ON CONFLICT(entity_a, entity_b) DO UPDATE SET "
                    "count = count + 1",
                    (a, b),
                )

                # 3. Compute totals for PMI
                total_row = state._conn.execute(
                    "SELECT SUM(count) AS total FROM entity_frequencies"
                ).fetchone()
                total = total_row["total"] if total_row and total_row["total"] else 0

                if total == 0:
                    state._conn.commit()
                    return 0.0

                count_a = state._conn.execute(
                    "SELECT count FROM entity_frequencies WHERE entity = ?",
                    (a,),
                ).fetchone()
                count_b = state._conn.execute(
                    "SELECT count FROM entity_frequencies WHERE entity = ?",
                    (b,),
                ).fetchone()
                count_ab = state._conn.execute(
                    "SELECT count FROM cooccurrence_frequencies "
                    "WHERE entity_a = ? AND entity_b = ?",
                    (a, b),
                ).fetchone()

                f_a = count_a["count"] if count_a else 0
                f_b = count_b["count"] if count_b else 0
                f_ab = count_ab["count"] if count_ab else 0

                if f_a == 0 or f_b == 0 or f_ab == 0:
                    state._conn.commit()
                    return 0.0

                p_a = f_a / total
                p_b = f_b / total
                p_ab = f_ab / total

                # PMI = log2(P(a,b) / (P(a) * P(b)))
                pmi = math.log2(p_ab / (p_a * p_b))

                # Edge weight = sigmoid(PMI)
                weight = self._sigmoid(pmi)

                # 4. Upsert into activation_edges
                state._conn.execute(
                    "INSERT INTO activation_edges "
                    "(source_entity, target_entity, strength, "
                    " co_activation_count) "
                    "VALUES (?, ?, ?, 1) "
                    "ON CONFLICT(source_entity, target_entity) DO UPDATE SET "
                    "strength = ?, "
                    "co_activation_count = co_activation_count + 1, "
                    "last_activated = CURRENT_TIMESTAMP",
                    (a, b, weight, weight),
                )

                state._conn.commit()
                return weight

        except Exception as e:
            logger.debug("PMI co-activation update failed: %s", e)
            return 0.0

    def get_pmi(self, state: 'PipelineState',
                entity_a: str, entity_b: str) -> float:
        """Return the PMI between two entities, or 0.0 if data is missing.

        Gated behind ``v2_pmi``.  Useful for callers that want the raw
        PMI value rather than the sigmoid-mapped weight.
        """
        _ff = FeatureFlags()
        if not _ff.is_enabled('v2_pmi'):
            return 0.0

        if not state or not entity_a or not entity_b:
            return 0.0

        a, b = sorted([entity_a, entity_b])

        try:
            with state._lock:
                total_row = state._conn.execute(
                    "SELECT SUM(count) AS total FROM entity_frequencies"
                ).fetchone()
                total = total_row["total"] if total_row and total_row["total"] else 0
                if total == 0:
                    return 0.0

                count_a = state._conn.execute(
                    "SELECT count FROM entity_frequencies WHERE entity = ?",
                    (a,),
                ).fetchone()
                count_b = state._conn.execute(
                    "SELECT count FROM entity_frequencies WHERE entity = ?",
                    (b,),
                ).fetchone()
                count_ab = state._conn.execute(
                    "SELECT count FROM cooccurrence_frequencies "
                    "WHERE entity_a = ? AND entity_b = ?",
                    (a, b),
                ).fetchone()

                f_a = count_a["count"] if count_a else 0
                f_b = count_b["count"] if count_b else 0
                f_ab = count_ab["count"] if count_ab else 0

                if f_a == 0 or f_b == 0 or f_ab == 0:
                    return 0.0

                p_a = f_a / total
                p_b = f_b / total
                p_ab = f_ab / total

                return math.log2(p_ab / (p_a * p_b))

        except Exception as e:
            logger.debug("PMI query failed: %s", e)
            return 0.0
