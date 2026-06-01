"""SalienceScorer -- Layer 1 sensory gate.

Pure rule-based multi-dimensional salience scorer. No LLM calls,
O(message_length) time. Thread-safe.

Contains:
- SalienceScorer class (with _lock)
- _RepetitionDetector class
- SalienceResult dataclass
- All pattern constants (_EMOTION_PATTERNS, _TRIVIAL_PATTERNS, etc.)
- All v2_salience fixes (decoupled novelty/rep_penalty)
"""

from __future__ import annotations

import math
import re
import threading
from collections import deque
from dataclasses import dataclass, field

from agent.pipeline.feature_flags import FeatureFlags


# ===========================================================================
# Pattern constants
# ===========================================================================

_EMOTION_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"[!！]{2,}"), 0.6),
    (re.compile(r"\b(urgent|critical|emergency|broken|crash|bug|fail)\b", re.I), 0.5),
    (re.compile(r"\b(down|outage|corrupt|overload|timeout|deadlock)\b", re.I), 0.45),
    (re.compile(r"\b(love|hate|amazing|terrible|awesome|awful)\b", re.I), 0.3),
    (re.compile(r"\b(worried|excited|frustrated|angry|happy|sad)\b", re.I), 0.35),
    (re.compile(r"\b(important|crucial|vital|essential|key)\b", re.I), 0.4),
]

_EMOTION_PATTERNS_ZH: list[tuple[re.Pattern, float]] = [
    (re.compile(r"[!！]{2,}"), 0.6),
    (re.compile(r"(宕机|崩溃|死锁|超时|断线|数据丢失|安全事故|生产事故)"), 0.8),
    (re.compile(r"(严重|紧急|危险|失败|出错|报错|异常|故障|漏洞)"), 0.65),
    (re.compile(r"(紧急|严重|崩溃|故障|坏了|挂了)"), 0.5),
    (re.compile(r"(喜欢|讨厌|太好了|太差了|棒极了|糟透了)"), 0.3),
    (re.compile(r"(担心|兴奋|沮丧|生气|开心|难过)"), 0.35),
    (re.compile(r"(重要|关键|必须|一定要|千万|别忘了)"), 0.4),
]

_IMPORTANCE_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\b(decided|decision|agreed|confirmed|final)\b", re.I), 0.7),
    (re.compile(r"\b(requirement|spec|specification|constraint)\b", re.I), 0.6),
    (re.compile(r"\b(deploy|release|production|launch)\b", re.I), 0.6),
    (re.compile(r"\b(architecture|design|refactor|migrat)\b", re.I), 0.5),
    (re.compile(r"\b(remember|note|important|don't forget)\b", re.I), 0.8),
    (re.compile(r"\b(prefer|always|never|usually)\b", re.I), 0.5),
    (re.compile(r"\b(bug|issue|error|problem)\b", re.I), 0.4),
]


_IMPORTANCE_PATTERNS_ZH: list[tuple[re.Pattern, float]] = [
    (re.compile(r"(决定|确认|最终|确定)"), 0.7),
    (re.compile(r"(需求|规格|约束|限制)"), 0.6),
    (re.compile(r"(部署|发布|上线|投产)"), 0.6),
    (re.compile(r"(记住|笔记|重要|别忘)"), 0.8),
    (re.compile(r"(喜欢|总是|从不|通常)"), 0.5),
]

_TRIVIAL_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"^(hi|hello|hey|thanks|ok|yes|no|sure)\s*[.!?]?\s*$", re.I), 0.9),
    (re.compile(r"^(good morning|good night|bye|see you)", re.I), 0.8),
    (re.compile(r"^(what time|what date|weather)", re.I), 0.5),
]

_TRIVIAL_PATTERNS_ZH: list[tuple[re.Pattern, float]] = [
    (re.compile(r"^(你好|嗨|谢谢|好的|是|不是|嗯)\s*[。！？]?\s*$"), 0.9),
    (re.compile(r"^(早上好|晚安|再见|拜拜)"), 0.8),
    (re.compile(r"^(几点|什么时间|天气)"), 0.5),
]

# Temporal recency patterns -- expressions indicating a recent or
# time-sensitive event.  Matches boost the novelty dimension of salience.
_RECENCY_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"(just now|right now|moments? ago|earlier today|this morning|this afternoon|this evening)", re.I), 0.6),
    (re.compile(r"(today|yesterday|last night)", re.I), 0.5),
    (re.compile(r"(this week|last week|recently|lately|just happened)", re.I), 0.4),
    (re.compile(r"(now|currently|at the moment|as we speak)", re.I), 0.45),
    (re.compile(r"(breaking|just in|update[:\s])", re.I), 0.55),
    (re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}"), 0.3),   # ISO date in text
]

_RECENCY_PATTERNS_ZH: list[tuple[re.Pattern, float]] = [
    (re.compile(r"(刚才|刚刚|此刻|现在|今天早上|今天下午|今天晚上)"), 0.6),
    (re.compile(r"(今天|昨天|前天|昨晚)"), 0.5),
    (re.compile(r"(本周|上周|最近|近期|近日)"), 0.4),
    (re.compile(r"(目前|当前|眼下|此时此刻)"), 0.45),
    (re.compile(r"(最新消息|突发|更新[：:])"), 0.55),
    (re.compile(r"\d{4}[-年/]\d{2}[-月/]\d{2}"), 0.3),   # CJK date in text
]


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class SalienceResult:
    """Multi-dimensional salience score for a message."""
    overall: float = 0.0
    emotion: float = 0.0
    novelty: float = 0.5
    importance: float = 0.0
    repetition_penalty: float = 1.0
    temporal_recency_boost: float = 0.0
    is_trivial: bool = False


@dataclass
class _RepetitionDetector:
    """Detects topic repetition using content hashing (F3 power-law penalty)."""
    window_size: int = 50
    _recent: deque = field(default_factory=lambda: deque(maxlen=50))
    _topic_counts: dict[str, int] = field(default_factory=dict)

    def _fuzzy_bucket(self, text: str) -> str:
        words = [w for w in re.sub(r"[^\w\s]", "", text.lower()).split() if len(w) > 2]
        return " ".join(words[:5])

    def observe(self, text: str) -> float:
        bucket = self._fuzzy_bucket(text)
        if not bucket:
            return 1.0
        self._topic_counts[bucket] = self._topic_counts.get(bucket, 0) + 1
        self._recent.append(bucket)
        if len(self._recent) == self._recent.maxlen or len(self._topic_counts) > self._recent.maxlen * 2:
            window_counts: dict[str, int] = {}
            for b in self._recent:
                window_counts[b] = window_counts.get(b, 0) + 1
            for topic in list(self._topic_counts):
                if topic not in window_counts:
                    del self._topic_counts[topic]
                else:
                    self._topic_counts[topic] = window_counts[topic]
        n = self._topic_counts.get(bucket, 1)
        return max(0.1, 1.0 / math.sqrt(n))

    def reset(self) -> None:
        self._recent.clear()
        self._topic_counts.clear()


# ===========================================================================
# SalienceScorer
# ===========================================================================

class SalienceScorer:
    """Multi-dimensional salience scorer -- the sensory gate.

    Pure rule-based -- no LLM calls, O(message_length) time.
    Scientific basis: F4 (CREB/excitability allocation).
    Thread-safe: all mutable state protected by _lock.
    """

    def __init__(self, novelty_window: int = 50) -> None:
        self._rep = _RepetitionDetector(window_size=novelty_window)
        self._lock = threading.Lock()

    def score(self, message: str) -> SalienceResult:
        if not message or not message.strip():
            return SalienceResult(overall=0.0, is_trivial=True)
        text = message.strip()
        with self._lock:
            trivial_penalty = 1.0
            for pattern, weight in _TRIVIAL_PATTERNS + _TRIVIAL_PATTERNS_ZH:
                if pattern.search(text):
                    trivial_penalty = min(trivial_penalty, 1.0 - weight)
            is_trivial = trivial_penalty < 0.3
            emotion = 0.0
            for pattern, weight in _EMOTION_PATTERNS + _EMOTION_PATTERNS_ZH:
                if pattern.search(text):
                    emotion = max(emotion, weight)
            if len(text) < 20:
                emotion *= 0.5
            importance = 0.0
            for pattern, weight in _IMPORTANCE_PATTERNS + _IMPORTANCE_PATTERNS_ZH:
                if pattern.search(text):
                    importance = max(importance, weight)
            if len(text) > 200:
                importance = min(1.0, importance + 0.1)
            freshness = self._rep.observe(text)
            # Bitemporal boost: recent-event expressions increase novelty
            recency_boost = 0.0
            for pattern, weight in _RECENCY_PATTERNS + _RECENCY_PATTERNS_ZH:
                if pattern.search(text):
                    recency_boost = max(recency_boost, weight)
            _ff1 = FeatureFlags()
            if _ff1.is_enabled('v2_salience'):
                # FIX 1 (H1): Decouple novelty from rep_penalty.
                # freshness IS the novelty signal; rep_penalty is separate.
                novelty = min(1.0, freshness + recency_boost)
                rep_factor = freshness
                raw = (0.25 * emotion + 0.30 * novelty + 0.30 * importance
                       + 0.15 * min(1.0, len(text) / 200))
                trivial_mult = (1.0 - (1.0 - trivial_penalty) * 0.8)
                adjusted = raw * (rep_factor if is_trivial else 1.0) * trivial_mult
            else:
                novelty = freshness
                rep_factor = freshness
                novelty = min(1.0, novelty + recency_boost)
                raw = (0.25 * emotion + 0.30 * novelty + 0.30 * importance
                       + 0.15 * min(1.0, len(text) / 200))
                adjusted = raw * rep_factor * (1.0 - (1.0 - trivial_penalty) * 0.8)
            overall = max(0.0, min(1.0, adjusted))
            return SalienceResult(
                overall=overall, emotion=emotion, novelty=novelty,
                importance=importance, repetition_penalty=rep_factor,
                temporal_recency_boost=recency_boost,
                is_trivial=is_trivial,
            )

    def reset(self) -> None:
        self._rep = _RepetitionDetector(window_size=self._rep.window_size)
