"""ReconsolidationEngine -- Layer 4 prediction-error updates.

When new information contradicts existing memories, the system enters
a "reconsolidation" mode: evaluating the conflict and updating.

Contains:
- ReconsolidationEngine class
- NLIDetector class (new, gated behind v2_nli)
- All existing conflict detection logic
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from agent.pipeline.feature_flags import FeatureFlags

logger = logging.getLogger(__name__)


class ReconsolidationEngine:
    """Prediction-error driven memory updates.

    When new information contradicts existing memories, the system enters
    a "reconsolidation" mode: evaluating the conflict and updating.
    Scientific basis: F8 (Sinclair & Barense 2019 Trends in Neurosciences).
    """

    def __init__(self, error_threshold: float = 0.3,
                 semantic_conflict_enabled: bool = False,
                 semantic_conflict_threshold: float = 0.7) -> None:
        self._threshold = error_threshold
        self._semantic_enabled = semantic_conflict_enabled
        self._semantic_threshold = semantic_conflict_threshold

    def check_retrieval(self, state: 'PipelineState',
                        query: str, result: str,
                        engrams: 'SilentEngramEngine | None' = None) -> None:
        """Record retrieval event for potential reconsolidation."""
        if not state:
            return
        try:
            from hashlib import sha256
            ref = sha256(query.encode()).hexdigest()[:16]
            if engrams:
                engrams.strengthen(state, ref)
            else:
                from agent.pipeline.engram import SilentEngramEngine
                SilentEngramEngine().strengthen(state, ref)
        except Exception as e:
            logger.debug("Reconsolidation check failed: %s", e)

    def detect_conflict(self, new_content: str,
                        existing_contents: list[str],
                        embed_fn: 'Any | None' = None,
                        llm_client: 'Any | None' = None) -> 'float | tuple[float, str]':
        """Detect prediction error between new and existing content.

        If semantic conflict detection is enabled and embed_fn is provided,
        delegates to detect_semantic_conflict for deeper analysis.

        Returns:
            float (legacy): error score [0, 1], high = high conflict
            tuple[float, str] (semantic): (error_score, action)
                action is one of: "update", "keep_both", "supersede", "no_conflict"
        """
        # Route to semantic detection when enabled and embeddings available
        if self._semantic_enabled and embed_fn is not None:
            return self.detect_semantic_conflict(
                new_content, existing_contents,
                embed_fn=embed_fn, llm_client=llm_client)

        # Legacy token-overlap heuristic
        if not existing_contents:
            return 0.0
        new_tokens = set(new_content.lower().split())
        max_overlap = 0.0
        for existing in existing_contents:
            existing_tokens = set(existing.lower().split())
            if not new_tokens or not existing_tokens:
                continue
            overlap = len(new_tokens & existing_tokens) / max(
                1, len(new_tokens | existing_tokens))
            max_overlap = max(max_overlap, overlap)
        # High overlap = low conflict, low overlap = high conflict
        return 1.0 - max_overlap

    def detect_semantic_conflict(
        self,
        new_content: str,
        existing_contents: list[str],
        embed_fn: 'Any | None' = None,
        llm_client: 'Any | None' = None,
    ) -> tuple[float, str]:
        """Three-stage semantic conflict detection.

        Stage 1 (Fast Filter): Compute embedding similarity between the new
        content and each existing content.  Only candidates with cosine
        similarity exceeding semantic_conflict_threshold proceed.

        Stage 2 (NLI Contradiction): When the v2_nli flag is enabled,
        run a cross-encoder NLI model on each candidate to get
        contradiction / entailment / neutral scores.

        Stage 3 (LLM Judgment): When llm_client is available, ask the LLM
        whether the new content truly contradicts the high-similarity
        candidates.

        Args:
            new_content: The incoming information to evaluate.
            existing_contents: List of existing memory contents to check against.
            embed_fn: Optional callable(text) -> list[float] that returns
                      an embedding vector for a given text.
            llm_client: Optional object with a .complete(prompt: str) -> str
                        method for LLM-based judgment.

        Returns:
            (error_score, action) where action is one of:
            "update", "keep_both", "supersede", "no_conflict".
        """
        if not existing_contents:
            return (0.0, "no_conflict")

        # --- Stage 1: Fast embedding similarity filter ---
        if embed_fn is None:
            score = self._token_overlap_conflict(new_content, existing_contents)
            action = "update" if score > 0.7 else "no_conflict"
            return (score, action)

        try:
            new_vec = embed_fn(new_content)
        except Exception as e:
            logger.debug("Embedding computation failed for new_content: %s", e)
            score = self._token_overlap_conflict(new_content, existing_contents)
            action = "update" if score > 0.7 else "no_conflict"
            return (score, action)

        if not new_vec:
            return (0.0, "no_conflict")

        candidates: list[tuple[str, float]] = []  # (content, similarity)
        for existing in existing_contents:
            try:
                existing_vec = embed_fn(existing)
            except Exception:
                continue
            if not existing_vec:
                continue
            sim = self._cosine_similarity(new_vec, existing_vec)
            if sim > self._semantic_threshold:
                candidates.append((existing, sim))

        if not candidates:
            return (0.0, "no_conflict")

        # --- Stage 2: NLI contradiction detection (v2_nli) ---
        _ff = FeatureFlags()
        if _ff.is_enabled('v2_nli'):
            nli = NLIDetector()
            best_contra = 0.0
            for cand_content, _ in candidates:
                scores = nli.detect_contradiction(new_content, cand_content)
                contra = scores.get("contradiction", 0.0)
                best_contra = max(best_contra, contra)
            if best_contra >= nli._threshold:
                # Strong NLI contradiction signal -- short-circuit
                error_score = best_contra
                action = "supersede" if best_contra > 0.9 else "update"
                return (error_score, action)
            # NLI says no contradiction; if entailment is high, it's
            # consistent information -- lower the error score.
            if best_contra < 0.3:
                # Could be entailment or neutral; let LLM/heuristic decide
                pass

        # --- Stage 3: LLM judgment ---
        if llm_client is not None:
            try:
                prompt = self._build_conflict_prompt(new_content, candidates)
                response = llm_client.complete(prompt)
                error_score, action = self._parse_llm_conflict_response(
                    response, candidates)
                return (error_score, action)
            except Exception as e:
                logger.debug("LLM conflict judgment failed: %s", e)

        # --- Heuristic fallback (no LLM) ---
        max_sim = max(s for _, s in candidates)
        error_score = 1.0 - max_sim
        if max_sim > 0.9:
            action = "update"
        elif max_sim > self._semantic_threshold + 0.1:
            action = "keep_both"
        else:
            action = "no_conflict"
        return (error_score, action)

    # -- internal helpers for semantic conflict detection --

    def _token_overlap_conflict(self, new_content: str,
                                existing_contents: list[str]) -> float:
        """Legacy token-overlap conflict score."""
        new_tokens = set(new_content.lower().split())
        max_overlap = 0.0
        for existing in existing_contents:
            existing_tokens = set(existing.lower().split())
            if not new_tokens or not existing_tokens:
                continue
            overlap = len(new_tokens & existing_tokens) / max(
                1, len(new_tokens | existing_tokens))
            max_overlap = max(max_overlap, overlap)
        return 1.0 - max_overlap

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 for zero-length or mismatched vectors.
        """
        if len(vec_a) != len(vec_b) or not vec_a:
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _build_conflict_prompt(
        new_content: str,
        candidates: list[tuple[str, float]],
    ) -> str:
        """Build LLM prompt for semantic conflict judgment.

        The prompt instructs the LLM to analyse whether the new content
        genuinely contradicts, supersedes, or complements each candidate.

        Args:
            new_content: The incoming information.
            candidates: List of (existing_content, similarity_score) tuples.

        Returns:
            A prompt string ready to send to the LLM.
        """
        parts: list[str] = []
        parts.append(
            "You are a memory conflict analyst.  Determine whether the "
            "NEW CONTENT genuinely contradicts any of the EXISTING KNOWLEDGE "
            "entries below, or whether it simply updates / extends / is "
            "independent of them."
        )
        parts.append("")
        parts.append("NEW CONTENT:")
        parts.append(new_content)
        parts.append("")
        parts.append("EXISTING KNOWLEDGE:")
        for idx, (content, sim) in enumerate(candidates, 1):
            parts.append(f"  [{idx}] (similarity={sim:.2f}) {content[:500]}")
        parts.append("")
        parts.append(
            "Respond with EXACTLY one of these labels on its own line:")
        parts.append(
            "  update      - new content directly corrects/overwrites an existing entry")
        parts.append(
            "  keep_both   - new content and existing entries are complementary; keep both")
        parts.append(
            "  supersede   - new content is a newer version that should replace existing")
        parts.append(
            "  no_conflict - no meaningful conflict detected")
        parts.append("")
        parts.append(
            "Then on a second line, provide a conflict severity score from 0.0 "
            "(no conflict) to 1.0 (severe conflict).")
        parts.append("Example response:")
        parts.append("update")
        parts.append("0.85")
        return "\n".join(parts)

    @staticmethod
    def _parse_llm_conflict_response(
        response: str,
        candidates: list[tuple[str, float]],
    ) -> tuple[float, str]:
        """Parse the LLM conflict judgment response.

        Args:
            response: Raw LLM output string.
            candidates: The candidates list (used for fallback heuristic).

        Returns:
            (error_score, action) tuple.
        """
        if not response:
            max_sim = max((s for _, s in candidates), default=0.0)
            return (1.0 - max_sim, "no_conflict")

        resp_lines = response.strip().splitlines()
        valid_actions = {"update", "keep_both", "supersede", "no_conflict"}
        action = "no_conflict"
        error_score = 0.0

        for line in resp_lines:
            stripped = line.strip().lower()
            if stripped in valid_actions:
                action = stripped
                break

        # Try to extract numeric score from any line
        for line in resp_lines:
            stripped = line.strip()
            match = re.search(r"(0\.\d+|1\.0+)", stripped)
            if match:
                try:
                    error_score = float(match.group(1))
                    error_score = max(0.0, min(1.0, error_score))
                    break
                except ValueError:
                    pass

        # Fallback: derive score from action if no numeric found
        if error_score == 0.0 and action != "no_conflict":
            max_sim = max((s for _, s in candidates), default=0.0)
            if action == "update":
                error_score = max(0.7, 1.0 - max_sim)
            elif action == "supersede":
                error_score = max(0.5, 1.0 - max_sim)
            elif action == "keep_both":
                error_score = max(0.2, 1.0 - max_sim)

        return (error_score, action)


# ===========================================================================
# NLIDetector -- Natural Language Inference conflict detection (v2_nli)
# ===========================================================================

class NLIDetector:
    """NLI-based contradiction detector with lazy model loading.

    Uses a cross-encoder NLI model (cross-encoder/nli-deberta-v3-small
    from sentence-transformers) to detect genuine contradictions between
    new and existing memories.  Gated behind the v2_nli feature flag.

    Model is loaded lazily on first call to avoid startup cost.  Falls
    back gracefully when the model or library is unavailable.
    """

    # Default model identifier; overridden via constructor or env var.
    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"

    def __init__(self, nli_model=None, threshold: float = 0.7) -> None:
        self._model = nli_model
        self._loaded = nli_model is not None
        self._threshold = threshold

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self):
        """Lazily load the cross-encoder NLI model on first use."""
        if self._loaded:
            return
        self._loaded = True
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.DEFAULT_MODEL)
            logger.info("Loaded NLI model: %s", self.DEFAULT_MODEL)
        except ImportError:
            logger.debug(
                "sentence-transformers not installed; NLI detection "
                "unavailable.")
            self._model = None
        except Exception as e:
            logger.debug("Failed to load NLI model %s: %s",
                         self.DEFAULT_MODEL, e)
            self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_contradiction(
        self,
        premise: str,
        hypothesis: str,
    ) -> dict[str, float]:
        """Detect whether hypothesis contradicts premise.

        Returns:
            dict with keys ``contradiction``, ``entailment``, ``neutral``
            (each a float in [0, 1]).  Returns all zeros when the feature
            flag is off or the model is unavailable.
        """
        _zero = {"contradiction": 0.0, "entailment": 0.0, "neutral": 0.0}
        _ff = FeatureFlags()
        if not _ff.is_enabled('v2_nli'):
            return _zero

        self._ensure_model()
        if self._model is None:
            return _zero

        try:
            result = self._model.predict([(premise, hypothesis)])
            # CrossEncoder.predict on NLI models returns logits array
            if isinstance(result, (list, tuple)):
                # Typical shape: array of [entailment, neutral, contradiction]
                scores = result[0] if hasattr(result[0], '__len__') else result
                if len(scores) >= 3:
                    # Softmax-normalize raw logits to probabilities
                    import math
                    exps = [math.exp(s) for s in scores[:3]]
                    total = sum(exps)
                    ent, neu, con = [e / total for e in exps]
                    return {
                        "contradiction": con,
                        "entailment": ent,
                        "neutral": neu,
                    }
                elif len(scores) == 1:
                    return {"contradiction": float(scores[0]),
                            "entailment": 0.0, "neutral": 0.0}
            # If model returns a dict directly
            if isinstance(result, dict):
                return {
                    "contradiction": result.get("contradiction", 0.0),
                    "entailment": result.get("entailment", 0.0),
                    "neutral": result.get("neutral", 0.0),
                }
            return _zero
        except Exception as e:
            logger.debug("NLI detection failed: %s", e)
            return _zero

    def is_contradiction(self, premise: str, hypothesis: str) -> bool:
        """Convenience: return True if contradiction score exceeds threshold."""
        scores = self.detect_contradiction(premise, hypothesis)
        return scores["contradiction"] >= self._threshold

    def batch_detect(
        self,
        premise: str,
        hypotheses: list[str],
    ) -> list[dict[str, float]]:
        """Batch contradiction detection against multiple hypotheses."""
        return [self.detect_contradiction(premise, h) for h in hypotheses]
