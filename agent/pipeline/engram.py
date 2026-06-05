"""SilentEngramEngine -- Layer 2 availability continuum.

Manages memory strength decay and recovery. Memories decay via power-law
but NEVER reach zero. Forgotten facts become "silent engrams" that can
be recovered via context similarity.

Contains:
- SilentEngramEngine class
- INITIAL_ENGRAM_STRENGTH constant (0.3)
- apply_decay() and apply_decay_with_emotion() methods
- All v2_engram fixes (fragile new engrams)
- All v2_emotion_decay fixes (per-memory emotional valence)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from agent.pipeline.feature_flags import FeatureFlags

logger = logging.getLogger(__name__)

# New engrams start in fragile period (Ebbinghaus 1885)
INITIAL_ENGRAM_STRENGTH = 0.3


# ===========================================================================
# FSRS Memory Strength Model (gated behind v2_fsrs)
# ===========================================================================

@dataclass
class MemoryStrength:
    """Free Spaced Repetition Scheduler (FSRS) memory strength model.

    Tracks three variables per memory:
    - difficulty: how hard the item is to learn (0..1, higher = harder)
    - stability: days until retrievability drops to 90% (positive float)
    - retrievability: probability of recall right now (0..1)

    Decay formula: R(t) = (1 + t / (9 * S))^(-1)
        where t = elapsed days, S = stability

    Review formula (simplified FSRS v4):
        New stability = S * (e^(w) * (11 - D) * S^(-w2) * (e^(w3 * (1-R)) - 1) * h + 1)
        Simplified: S' = S * (1 + w * (11 - D) * (1 - R))
        where w is a tunable weight (default 0.5)

    Gated behind ``v2_fsrs`` feature flag.  When the flag is off,
    callers should fall back to the existing SilentEngramEngine.
    """

    difficulty: float = 0.3
    stability: float = 1.0
    retrievability: float = 1.0

    def decay(self, elapsed_days: float) -> float:
        """Compute retrievability after *elapsed_days* without review.

        Uses the FSRS power-law decay:
            R = (1 + t / (9 * S))^(-1)

        Returns the new retrievability (does NOT mutate this instance).
        """
        if elapsed_days <= 0.0:
            return self.retrievability
        if self.stability <= 0.0:
            return 0.0
        return (1.0 + elapsed_days / (9.0 * self.stability)) ** -1.0

    def review(self, rating: int) -> 'MemoryStrength':
        """Process a review and return an updated MemoryStrength.

        Parameters
        ----------
        rating : int
            Review quality rating: 1 = Again, 2 = Hard, 3 = Good, 4 = Easy.

        Returns
        -------
        MemoryStrength
            A NEW instance with updated difficulty, stability, and
            retrievability set to 1.0 (just reviewed).

        Notes
        -----
        Simplified FSRS formulas:
        - difficulty_delta = -0.1 * (rating - 3)  (Easy lowers, Again raises)
        - stability scales with (11 - difficulty) and rating quality
        """
        if rating < 1:
            rating = 1
        if rating > 4:
            rating = 4

        # Update difficulty: mean-reverting toward (4 - rating) / 3
        target_difficulty = (4 - rating) / 3.0
        new_difficulty = self.difficulty + 0.1 * (target_difficulty - self.difficulty)
        new_difficulty = max(0.0, min(1.0, new_difficulty))

        # Update stability based on rating and difficulty
        # Higher rating and lower difficulty => bigger stability gain
        quality_factor = 0.1 + 0.3 * (rating - 1) / 3.0   # 0.1..0.4
        difficulty_factor = 11.0 - new_difficulty * 10.0      # 1.0..11.0
        stability_gain = 1.0 + quality_factor * difficulty_factor
        new_stability = self.stability * stability_gain

        return MemoryStrength(
            difficulty=new_difficulty,
            stability=max(0.1, new_stability),
            retrievability=1.0,
        )


class SilentEngramEngine:
    """Manages memory strength decay and recovery.

    Memories decay via power-law but NEVER reach zero.  Forgotten facts
    become "silent engrams" that can be recovered via context similarity.
    Scientific basis: F5 (Ryan et al. 2015 Science -- forgetting != erasure).

    When emotion_modulated_decay_enabled is True, emotionally arousing
    memories decay more slowly: adjusted_half_life = half_life *
    (1 + emotion_decay_multiplier * |valence|).  With default multiplier
    of 2.0, high-emotion memories (valence ~0.6) decay up to 2.2x slower.
    Scientific basis: McGaugh 2004 -- amygdala modulates emotionally
    arousing memory consolidation.

    Thresholds:
        active:      strength > 0.5
        semi_active:  0.2 < strength <= 0.5
        silent:       0.05 < strength <= 0.2
        buried:       strength <= 0.05
    """

    ACTIVE = 0.5
    SEMI_ACTIVE = 0.2
    SILENT = 0.05

    def __init__(self, half_life_hours: float = 720.0,
                 emotion_modulated_decay_enabled: bool = False,
                 emotion_decay_multiplier: float = 2.0) -> None:
        self._half_life = half_life_hours
        self._emotion_modulated = emotion_modulated_decay_enabled
        self._emotion_multiplier = emotion_decay_multiplier

    def apply_decay(self, state: 'PipelineState', hours_elapsed: float = 1.0,
                    emotional_valence: float | None = None) -> int:
        """Apply power-law decay to all engram strengths. Returns affected rows.

        When emotional_valence is provided and emotion_modulated_decay is enabled,
        the half-life is adjusted: adjusted = half_life * (1 + multiplier * |valence|).
        High emotion means slower decay (up to 3x with default multiplier).
        Scientific basis: McGaugh 2004 -- amygdala modulates emotionally arousing
        memory consolidation.
        """
        if not state:
            return 0
        try:
            effective_half_life = self._half_life
            if (emotional_valence is not None
                    and self._emotion_modulated
                    and abs(emotional_valence) > 0.0):
                effective_half_life = self._half_life * (
                    1.0 + self._emotion_multiplier * abs(emotional_valence))
            decay_factor = 0.5 ** (hours_elapsed / effective_half_life)
            with state._lock:
                cursor = state._conn.execute(
                    "UPDATE engram_strengths SET "
                    "strength = MAX(0.001, strength * ?), "
                    "last_accessed = CURRENT_TIMESTAMP "
                    "WHERE strength > 0.001",
                    (decay_factor,),
                )
                state._conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.debug("Engram decay failed: %s", e)
            return 0

    def apply_decay_with_emotion(
        self, state: 'PipelineState', hours_elapsed: float,
        emotional_valences: dict[str, float],
    ) -> int:
        """Apply per-memory emotion-modulated decay.

        Each memory_ref in emotional_valences gets its own adjusted half-life
        based on its emotional valence.  Memories not in the dict use the
        base half-life.

        Args:
            state: PipelineState with engram_strengths table.
            hours_elapsed: Fractional hours since last decay.
            emotional_valences: {memory_ref: emotional_valence} mapping.
                Valence is in [0, 1] range (typically 0.0-0.6 from SalienceScorer).

        Returns:
            Total number of affected rows.
        """
        if not state:
            return 0
        total_affected = 0
        try:
            with state._lock:
                # Update half_life_hours for targeted memories
                for ref, valence in emotional_valences.items():
                    if self._emotion_modulated and abs(valence) > 0.0:
                        adjusted = self._half_life * (
                            1.0 + self._emotion_multiplier * abs(valence))
                    else:
                        adjusted = self._half_life
                    state._conn.execute(
                        "UPDATE engram_strengths SET decay_half_life_hours = ? "
                        "WHERE memory_ref = ?",
                        (adjusted, ref),
                    )

                # Apply decay using per-row half_life_hours
                cursor = state._conn.execute(
                    "UPDATE engram_strengths SET "
                    "strength = MAX(0.001, strength * "
                    "  POWER(0.5, ? / decay_half_life_hours)), "
                    "last_accessed = CURRENT_TIMESTAMP "
                    "WHERE strength > 0.001",
                    (hours_elapsed,),
                )
                total_affected = cursor.rowcount

                # Reset half_life_hours for non-targeted memories back to base
                refs = list(emotional_valences.keys())
                if refs:
                    placeholders = ",".join("?" for _ in refs)
                    state._conn.execute(
                        f"UPDATE engram_strengths SET decay_half_life_hours = ? "
                        f"WHERE memory_ref NOT IN ({placeholders}) "
                        f"AND decay_half_life_hours != ?",
                        [self._half_life] + refs + [self._half_life],
                    )
                else:
                    state._conn.execute(
                        "UPDATE engram_strengths SET decay_half_life_hours = ? "
                        "WHERE decay_half_life_hours != ?",
                        (self._half_life, self._half_life),
                    )

                state._conn.commit()
        except Exception as e:
            logger.debug("Engram emotion-modulated decay failed: %s", e)
        return total_affected

    def strengthen(self, state: 'PipelineState', memory_ref: str,
                   delta: float = 0.03,
                   emotional_valence: float = 0.0) -> float:
        """Strengthen an engram on retrieval (spacing effect). Returns new strength."""
        if not state:
            return 0.0
        try:
            with state._lock:
                row = state._conn.execute(
                    "SELECT strength FROM engram_strengths WHERE memory_ref = ?",
                    (memory_ref,),
                ).fetchone()
                _ff8 = FeatureFlags()
                if row:
                    new_str = min(1.0, row["strength"] + delta)
                    if _ff8.is_enabled('v2_emotion_decay'):
                        # FIX 8 (M2): Store per-memory emotional valence
                        state._conn.execute(
                            "UPDATE engram_strengths SET strength = ?, "
                            "last_accessed = CURRENT_TIMESTAMP, "
                            "access_count = access_count + 1, "
                            "emotional_valence = ? "
                            "WHERE memory_ref = ?",
                            (new_str, emotional_valence, memory_ref),
                        )
                    else:
                        state._conn.execute(
                            "UPDATE engram_strengths SET strength = ?, "
                            "last_accessed = CURRENT_TIMESTAMP, "
                            "access_count = access_count + 1 "
                            "WHERE memory_ref = ?",
                            (new_str, memory_ref),
                        )
                else:
                    _ff2 = FeatureFlags()
                    if _ff2.is_enabled('v2_engram') and delta > 0:
                        # FIX 2 (H3): New engrams start in fragile period (Ebbinghaus)
                        new_str = min(1.0, INITIAL_ENGRAM_STRENGTH + delta)
                    else:
                        new_str = min(1.0, 1.0 + delta)
                    state._conn.execute(
                        "INSERT INTO engram_strengths "
                        "(memory_ref, provider, strength, emotional_valence) "
                        "VALUES (?, 'unknown', ?, ?)",
                        (memory_ref, new_str, emotional_valence),
                    )
                state._conn.commit()
                return new_str
        except Exception as e:
            logger.debug("Engram strengthen failed: %s", e)
            return 0.0

    def classify(self, strength: float) -> str:
        """Classify strength into accessibility level."""
        if strength > self.ACTIVE:
            return "active"
        elif strength > self.SEMI_ACTIVE:
            return "semi_active"
        elif strength > self.SILENT:
            return "silent"
        return "buried"
