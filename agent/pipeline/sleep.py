"""SleepScheduler -- Layer 9 automatic sleep-driven consolidation.

Monitors per-message salience and idle gaps. When the system has been
quiet for idle_threshold_minutes AND accumulated salience exceeds
salience_threshold, triggers a two-phase sleep cycle:
  Phase 1 (SWS): ConsolidationEngine transfer
  Phase 2 (REM): DreamEngine selective replay

Contains:
- SleepScheduler class
- Cooperative shutdown via threading.Event
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class SleepScheduler:
    """Automatic sleep scheduler for memory consolidation.

    Monitors per-message salience and idle gaps.  When the system has been
    quiet for ``idle_threshold_minutes`` AND the accumulated salience since
    the last sleep cycle exceeds ``salience_threshold``, triggers a two-phase
    sleep cycle:

    Phase 1 (SWS):  Run ConsolidationEngine to transfer episodic facts into
                     semantic schemas.
    Phase 2 (REM):  Run DreamEngine for structured selective replay.

    Thread-safe: all mutable state protected by ``_lock``.
    """

    def __init__(self, idle_threshold_minutes: float = 5.0,
                 salience_threshold: float = 10.0) -> None:
        self._idle_threshold_s: float = idle_threshold_minutes * 60.0
        self._salience_threshold: float = salience_threshold
        self._accumulated_salience: float = 0.0
        self._last_activity: float = time.time()
        self._sleeping: bool = False
        self._sleep_count: int = 0
        self._last_sleep_duration_s: float = 0.0
        self._lock = threading.Lock()
        # Injected by MemoryPipeline during initialize()
        self._state: 'PipelineState | None' = None
        self._session_id: str = ""
        # Cooperative shutdown event
        self._shutdown_event = threading.Event()

    def on_message(self, salience_score: float) -> None:
        """Accumulate salience and update last-activity timestamp."""
        with self._lock:
            self._accumulated_salience += max(0.0, salience_score)
            self._last_activity = time.time()

    def should_sleep(self) -> bool:
        """True when idle AND accumulated salience >= threshold."""
        with self._lock:
            if self._sleeping:
                return False
            idle_seconds = time.time() - self._last_activity
            return (idle_seconds >= self._idle_threshold_s
                    and self._accumulated_salience >= self._salience_threshold)

    def sleep_cycle(self, consolidation_engine, dream_engine) -> dict:
        """Run Phase 1 (SWS) consolidation, Phase 2 (REM) dreaming, then
        reset accumulated salience.

        Returns a summary dict with keys ``sws``, ``rem``,
        ``salience_reset``, ``duration_s``.
        """
        with self._lock:
            if self._sleeping:
                return {"skipped": True, "reason": "already sleeping"}
            self._sleeping = True

        start = time.time()
        result: dict = {"sws": {}, "rem": {}}
        try:
            # --- Phase 1: SWS -- consolidation ---
            if consolidation_engine and self._state:
                try:
                    result["sws"] = consolidation_engine.consolidate(
                        self._state, facts=None)
                except Exception as e:
                    logger.debug("Sleep SWS consolidation failed: %s", e)
                    result["sws"] = {"error": str(e)}

            # --- Phase 2: REM -- dreaming ---
            if dream_engine:
                try:
                    dream_result = dream_engine.dream_cycle(
                        session_id=self._session_id)
                    result["rem"] = {
                        "mode": getattr(dream_result, "mode", "?"),
                        "facts_replayed": getattr(
                            dream_result, "facts_replayed", 0),
                    }
                except Exception as e:
                    logger.debug("Sleep REM dreaming failed: %s", e)
                    result["rem"] = {"error": str(e)}

            # --- Reset salience after full cycle ---
            duration = time.time() - start
            with self._lock:
                result["salience_reset"] = round(
                    self._accumulated_salience, 4)
                self._accumulated_salience = 0.0
                self._sleep_count += 1
                self._last_sleep_duration_s = duration
                self._sleeping = False

            result["duration_s"] = round(duration, 3)
            logger.debug(
                "Sleep cycle completed in %.1fs: sws=%s, rem=%s",
                duration, result["sws"], result["rem"])
        except Exception as e:
            logger.debug("Sleep cycle failed: %s", e)
            with self._lock:
                self._sleeping = False
            result["error"] = str(e)

        return result

    def get_status(self) -> dict:
        """Current state dict for health dashboard."""
        with self._lock:
            idle_seconds = time.time() - self._last_activity
            return {
                "sleeping": self._sleeping,
                "accumulated_salience": round(
                    self._accumulated_salience, 2),
                "salience_threshold": self._salience_threshold,
                "idle_seconds": round(idle_seconds, 1),
                "idle_threshold_minutes": round(
                    self._idle_threshold_s / 60.0, 1),
                "should_sleep": (
                    not self._sleeping
                    and idle_seconds >= self._idle_threshold_s
                    and self._accumulated_salience >= self._salience_threshold
                ),
                "sleep_count": self._sleep_count,
                "last_sleep_duration_s": round(
                    self._last_sleep_duration_s, 2),
            }

    def reset(self) -> None:
        """Reset accumulated salience and activity tracking."""
        with self._lock:
            self._accumulated_salience = 0.0
            self._last_activity = time.time()

    def request_shutdown(self) -> None:
        """Signal cooperative shutdown."""
        self._shutdown_event.set()

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_event.is_set()
