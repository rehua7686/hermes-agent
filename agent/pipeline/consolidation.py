"""ConsolidationEngine -- Layer 3 sleep-like consolidation.

Consolidates episodic memories into semantic schemas via a three-phase
process: select, transfer, integrate.

Contains:
- ConsolidationEngine class
- DeepConsolidationEngine class (LLM-assisted abstraction)
- All v2_dedup fixes (SHA256 + Jaccard)
"""

from __future__ import annotations

import logging
import re
from hashlib import sha256

from agent.pipeline.feature_flags import FeatureFlags
from agent.pipeline.state import _ZH_STOPWORDS

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """Consolidates episodic memories into semantic schemas.

    Three-phase process mimicking sleep consolidation:
    1. Select: pick salient unconsolidated facts
    2. Transfer: group by entity/category, create schema candidates
    3. Integrate: merge with existing schemas or create new ones
    Scientific basis: F6 (Diekelmann & Born 2019 Nature Reviews Neuroscience).
    """

    def __init__(self, min_facts: int = 5) -> None:
        self._min_facts = min_facts

    def consolidate(self, state: 'PipelineState',
                    facts: list[dict] | None = None) -> dict:
        """Run consolidation. Returns summary dict.

        In Phase 1-2, this operates on pipeline_state.db schemas.
        In Phase 3+, it will pull facts from providers.

        Bitemporal consideration: facts are sorted by their effective
        timestamp (event_time > ingestion_time > created_at) so that
        temporally proximate facts are grouped together during schema
        creation.  Temporal proximity also boosts the initial confidence
        of a new schema -- facts within the same time window are more
        likely to describe the same underlying situation.
        """
        if not state:
            return {"schemas_created": 0, "schemas_updated": 0}
        created, updated = 0, 0
        try:
            with state._lock:
                # Consolidation runs whenever we have enough new facts
                if facts and len(facts) >= self._min_facts:
                    # Sort facts by temporal proximity: prefer event_time,
                    # fall back to ingestion_time, then created_at.
                    def _sort_key(f: dict) -> str:
                        return (f.get("event_time")
                                or f.get("ingestion_time")
                                or f.get("created_at")
                                or "")
                    facts_sorted = sorted(facts, key=_sort_key)

                    # Fix 2: Prefer facts with LOWER engram strength
                    # (they need consolidation most -- like sleep
                    # prioritizes fragile memories for replay)
                    # Batch-fetch all strengths in one query
                    refs_to_strength: dict[str, float] = {}
                    refs = [sha256(f.get('content', '').encode()
                                   ).hexdigest()[:16]
                            for f in facts_sorted]
                    unique_refs = list(set(refs))
                    if unique_refs:
                        placeholders = ",".join(
                            "?" for _ in unique_refs)
                        rows = state._conn.execute(
                            f"SELECT memory_ref, strength "
                            f"FROM engram_strengths "
                            f"WHERE memory_ref IN ({placeholders})",
                            unique_refs,
                        ).fetchall()
                        refs_to_strength = {
                            r["memory_ref"]: r["strength"]
                            for r in rows
                        }
                    def _engram_strength_key(fact_d: dict) -> float:
                        ref = sha256(
                            fact_d.get('content', '').encode()
                        ).hexdigest()[:16]
                        return refs_to_strength.get(ref, 1.0)
                    facts_sorted.sort(key=_engram_strength_key)

                    # Get existing schema contents for dedup
                    _ff3 = FeatureFlags()
                    if _ff3.is_enabled('v2_dedup'):
                        # FIX 3 (H2): SHA256 hash + Jaccard similarity dedup
                        _existing_rows = state._conn.execute(
                            "SELECT schema_id, content FROM schemas "
                            "ORDER BY updated_at DESC LIMIT 100"
                        ).fetchall()
                        _norm = lambda t: " ".join(t.lower().split())
                        existing_hashes = {
                            sha256(_norm(r["content"]).encode()).hexdigest()[:16]
                            for r in _existing_rows
                        }
                        existing_schemas = [
                            (r["schema_id"], r["content"]) for r in _existing_rows
                        ]
                    else:
                        _existing_rows_old = state._conn.execute(
                            "SELECT content FROM schemas ORDER BY updated_at DESC LIMIT 20"
                        ).fetchall()
                        existing_contents_old = {
                            r["content"][:50] for r in _existing_rows_old
                        }

                    for fact in facts_sorted[:10]:
                        content = fact.get("content", "")
                        domain = fact.get("domain", "general")
                        if not content or len(content) < 10:
                            continue
                        if _ff3.is_enabled('v2_dedup'):
                            h = sha256(
                                " ".join(content.lower().split()).encode()
                            ).hexdigest()[:16]
                            if h in existing_hashes:
                                updated += 1
                                continue
                            # Word-level Jaccard for approximate dedup
                            _is_dup = False
                            _cw = set(content.lower().split())
                            for _sid, _ec in existing_schemas:
                                _ew = set(_ec.lower().split())
                                _jaccard = (
                                    len(_cw & _ew) / len(_cw | _ew)
                                    if _cw and _ew else 0.0
                                )
                                if _jaccard > 0.85:
                                    _is_dup = True
                                    state._conn.execute(
                                        "UPDATE schemas SET "
                                        "confidence = MIN(1.0, confidence + 0.05), "
                                        "updated_at = CURRENT_TIMESTAMP "
                                        "WHERE schema_id = ?",
                                        (_sid,),
                                    )
                                    updated += 1
                                    break
                            if _is_dup:
                                continue
                        else:
                            if content[:50] in existing_contents_old:
                                updated += 1
                                continue
                        base_conf = self._temporal_confidence_boost(fact)
                        state._conn.execute(
                            "INSERT INTO schemas (content, domain, confidence) "
                            "VALUES (?, ?, ?)",
                            (content, domain, base_conf),
                        )
                        schema_id = state._conn.execute(
                            "SELECT last_insert_rowid()"
                        ).fetchone()[0]
                        mem_ref = sha256(
                            content.encode()
                        ).hexdigest()[:16]
                        state._conn.execute(
                            "INSERT OR IGNORE INTO "
                            "schema_sources "
                            "(schema_id, memory_ref, "
                            " provider) "
                            "VALUES (?, ?, ?)",
                            (schema_id, mem_ref,
                             "pipeline"),
                        )
                        created += 1

                    # OPT 6: Write cross-domain links for new schemas
                    _new_schemas = []
                    for _fact in facts_sorted[:10]:
                        _c = _fact.get('content', '')
                        _d = _fact.get('domain', 'general')
                        if not _c or len(_c) < 10:
                            continue
                        _is_new = True
                        if _ff3.is_enabled('v2_dedup'):
                            _ch = sha256(
                                " ".join(_c.lower().split()).encode()
                            ).hexdigest()[:16]
                            _is_new = _ch not in existing_hashes
                        else:
                            _is_new = _c[:50] not in existing_contents_old
                        if _is_new:
                            _new_schemas.append((_c, _d))

                    if _new_schemas:
                        _entity_domains: dict[str, set[str]] = {}
                        for _c, _d in _new_schemas:
                            _ents = set(
                                re.findall(r'[A-Z][a-z]{2,}', _c))
                            _ents.update(
                                e for e in re.findall(
                                    r'[一-鿿]{2,6}', _c)
                                if e not in _ZH_STOPWORDS)
                            for _e in _ents:
                                _entity_domains.setdefault(
                                    _e, set()).add(_d)

                        for _e, _doms in _entity_domains.items():
                            _ph = ','.join('?' for _ in _doms)
                            _rows = state._conn.execute(
                                'SELECT DISTINCT domain FROM schemas '
                                f'WHERE domain NOT IN ({_ph}) '
                                'AND content LIKE ?',
                                list(_doms) + [f'%{_e}%'],
                            ).fetchall()
                            _all_doms = _doms | {
                                r['domain'] for r in _rows}
                            if len(_all_doms) >= 2:
                                _sorted = sorted(_all_doms)
                                for _i in range(len(_sorted)):
                                    for _j in range(
                                            _i + 1, len(_sorted)):
                                        state._conn.execute(
                                            'INSERT INTO '
                                            'cross_domain_links '
                                            '(entity, domain_a, '
                                            'domain_b, strength) '
                                            'VALUES (?, ?, ?, 0.5)',
                                            (_e, _sorted[_i],
                                             _sorted[_j]),
                                        )

                # Log the run (single commit for atomicity)
                state._conn.execute(
                    "INSERT INTO consolidation_runs "
                    "(session_id, memories_processed, schemas_created, schemas_updated) "
                    "VALUES (?, ?, ?, ?)",
                    (getattr(self, '_session_id', '') or "", len(facts or []), created, updated),
                )
                state._conn.commit()
        except Exception as e:
            logger.debug("Consolidation failed: %s", e)
        return {"schemas_created": created, "schemas_updated": updated}

    @staticmethod
    def _temporal_confidence_boost(fact: dict) -> float:
        """Compute initial confidence with temporal proximity boost.

        Facts with a recent event_time get higher confidence because
        they describe a temporally grounded event.
        """
        base_conf = 0.5
        event_ts = fact.get("event_time") or fact.get("ingestion_time")
        if event_ts:
            try:
                from datetime import datetime, timezone
                evt = datetime.fromisoformat(str(event_ts))
                if evt.tzinfo is None:
                    evt = evt.replace(tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - evt).total_seconds() / 3600.0
                if age_hours < 1.0:
                    base_conf = 0.65
                elif age_hours < 24.0:
                    base_conf = 0.55
            except Exception:
                pass
        return base_conf

    def extract_insights(self, messages: list) -> str:
        """Extract key facts from messages about to be discarded by compression."""
        insights = []
        for msg in messages[-5:]:  # last 5 messages
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if len(content) > 50:
                # Extract first sentence as insight
                first_sentence = content.split(".")[0][:200]
                if first_sentence.strip():
                    insights.append(f"- {first_sentence.strip()}")
        return "\n".join(insights) if insights else ""



# ===========================================================================
# Layer 3b: DeepConsolidationEngine (LLM-assisted abstraction)
# ===========================================================================


class DeepConsolidationEngine(ConsolidationEngine):
    """LLM-assisted deep consolidation for abstract schema generation.


    Extends the base ConsolidationEngine with a second pass that uses an
    LLM to synthesise higher-level abstract schemas from the concrete
    schemas produced by the base consolidation.
    """


    def __init__(self, llm_client=None, min_facts: int = 3) -> None:
        super().__init__(min_facts=min_facts)
        self._llm = llm_client


    def consolidate(self, state, facts=None) -> dict:
        """Run base consolidation, then LLM-assisted abstraction."""
        result = super().consolidate(state, facts)
        if not self._llm or not state:
            return result
        try:
            abstract_created = self._generate_abstract_schemas(state)
            result["abstract_schemas_created"] = abstract_created
        except Exception as e:
            logger.debug("Deep consolidation abstraction failed: %s", e)
        return result


    def _generate_abstract_schemas(self, state) -> int:
        """Use LLM to produce abstract schemas from existing schemas."""
        created = 0
        with state._lock:
            rows = state._conn.execute(
                "SELECT schema_id, content, domain, confidence "
                "FROM schemas ORDER BY updated_at DESC LIMIT 10"
            ).fetchall()
        if len(rows) < 2:
            return 0
        schema_desc = []
        for r in rows:
            schema_desc.append(
                f"[{r['domain']}] (conf={r['confidence']:.2f}) "
                f"{r['content'][:200]}"
            )
        _nl = chr(10)
        prompt = (
            "You are a memory consolidation assistant. Given these "
            "memory schemas, generate 1-3 higher-level abstract "
            "schemas that capture key patterns or themes."
            + _nl + _nl
            + "SCHEMAS:" + _nl + _nl.join(schema_desc) + _nl + _nl
            + "Format each abstract schema as: DOMAIN|STATEMENT"
            + _nl + "One per line."
        )
        try:
            response = self._llm.complete(prompt)
        except Exception as e:
            logger.debug("LLM abstract schema call failed: %s", e)
            return 0
        if not response:
            return 0
        existing_prefixes = set()
        with state._lock:
            for r in state._conn.execute(
                    "SELECT content FROM schemas").fetchall():
                existing_prefixes.add(r["content"][:50])
        for line in response.strip().splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            domain = parts[0].strip().lower()
            content_val = parts[1].strip()
            if not content_val or len(content_val) < 10:
                continue
            if content_val[:50] in existing_prefixes:
                continue
            with state._lock:
                state._conn.execute(
                    "INSERT INTO schemas (content, domain, confidence) "
                    "VALUES (?, ?, ?)",
                    (content_val, domain, 0.70),
                )
                state._conn.commit()
            existing_prefixes.add(content_val[:50])
            created += 1
        return created
