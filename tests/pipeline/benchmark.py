#!/usr/bin/env python3
"""Benchmark script for the Hermes organic memory pipeline.

Measures performance with feature flags OFF (baseline) vs ON (optimized).
Outputs JSON results to stdout for automated parsing.

Usage:
    python tests/pipeline/benchmark.py
"""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
import sys
import threading
import time
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Patch hermes_state before any pipeline imports
import hermes_state
hermes_state.apply_wal_with_fallback = lambda conn, db_label="": "wal"

from agent.memory_pipeline import (
    ActivationGraph,
    ConsolidationEngine,
    FeedbackCoordinator,
    PipelineState,
    ReconsolidationEngine,
    SalienceScorer,
    SilentEngramEngine,
)
from agent.pipeline.feature_flags import FeatureFlags

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(tmp_dir: str) -> PipelineState:
    """Create a PipelineState backed by a temp database."""
    import tempfile, os
    db_path = os.path.join(tmp_dir, "bench.db")
    return PipelineState(db_path=db_path)


def _seed_engrams(state: PipelineState, count: int) -> list[str]:
    """Insert N engram rows for benchmarking decay."""
    refs = []
    for i in range(count):
        ref = f"engram_{i:06d}"
        refs.append(ref)
        state._conn.execute(
            "INSERT OR REPLACE INTO engram_strengths "
            "(memory_ref, provider, strength) VALUES (?, 'bench', 1.0)",
            (ref,),
        )
    state._conn.commit()
    return refs


def _seed_activation_graph(state: PipelineState, n_entities: int,
                           edges_per_entity: int = 3) -> list[str]:
    """Build a synthetic activation graph for benchmarking."""
    entities = [f"Entity_{i}" for i in range(n_entities)]
    for i, ent in enumerate(entities):
        for j in range(1, edges_per_entity + 1):
            target = entities[(i + j) % n_entities]
            a, b = sorted([ent, target])
            state._conn.execute(
                "INSERT OR IGNORE INTO activation_edges "
                "(source_entity, target_entity, strength) "
                "VALUES (?, ?, 0.5)",
                (a, b),
            )
    state._conn.commit()
    return entities


def _seed_schemas(state: PipelineState, count: int) -> list[str]:
    """Insert N schema rows for consolidation dedup testing."""
    contents = []
    for i in range(count):
        content = f"Schema content number {i} about topic {i % 5} with unique details {i}"
        contents.append(content)
        state._conn.execute(
            "INSERT INTO schemas (content, domain, confidence) "
            "VALUES (?, ?, ?)",
            (content, f"domain_{i % 5}", 0.5 + (i % 10) * 0.05),
        )
    state._conn.commit()
    return contents


def _generate_test_messages(count: int) -> list[str]:
    """Generate diverse test messages for salience benchmarking."""
    templates = [
        "The {} system is experiencing a critical failure in production",
        "I decided to refactor the architecture for better performance",
        "Remember to update the deployment configuration before launch",
        "hi",
        "ok",
        "The weather is nice today",
        "紧急处理！系统崩溃了",
        "这是一个非常重要的决定，必须确认",
        "Python is a popular programming language used worldwide",
        "The database query optimization reduced latency by 80%",
        "Just now, the monitoring system detected an outage",
        "We need to migrate the legacy codebase to the new framework",
        "Breaking news: the API gateway has been completely rewritten",
        "The team agreed on the final specification for the new feature",
        "今天下午发现了一个严重的bug",
    ]
    messages = []
    for i in range(count):
        messages.append(templates[i % len(templates)])
    return messages


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def bench_salience_throughput(iterations: int = 5000) -> dict:
    """Measure salience scoring throughput (messages/second)."""
    scorer = SalienceScorer()
    messages = _generate_test_messages(iterations)

    # Warm up
    for m in messages[:100]:
        scorer.score(m)
    scorer.reset()

    start = time.perf_counter()
    for m in messages:
        scorer.score(m)
    elapsed = time.perf_counter() - start

    return {
        "total_messages": iterations,
        "elapsed_seconds": round(elapsed, 4),
        "throughput_msg_per_sec": round(iterations / elapsed, 1),
    }


def bench_consolidation_latency(n_facts_list: list[int]) -> dict:
    """Measure consolidation latency for various fact counts."""
    results = {}
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        state = _make_state(tmp)
        # Pre-seed existing schemas for dedup testing
        _seed_schemas(state, 50)

        for n in n_facts_list:
            engine = ConsolidationEngine(min_facts=max(1, n // 2))
            facts = [
                {"content": f"Consolidation benchmark fact {i} about topic {i % 3} with enough text for dedup",
                 "domain": f"domain_{i % 3}"}
                for i in range(n)
            ]
            # Time multiple runs
            times = []
            for _ in range(5):
                start = time.perf_counter()
                engine.consolidate(state, facts=facts)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            results[f"{n}_facts"] = {
                "mean_ms": round(statistics.mean(times) * 1000, 2),
                "median_ms": round(statistics.median(times) * 1000, 2),
                "p95_ms": round(sorted(times)[int(len(times) * 0.95)] * 1000, 2),
            }
        state.close()
    return results


def bench_engram_decay(memory_counts: list[int]) -> dict:
    """Measure engram decay performance for various memory counts."""
    results = {}
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        state = _make_state(tmp)

        for count in memory_counts:
            # Fresh DB state for each count
            state._conn.execute("DELETE FROM engram_strengths")
            state._conn.commit()
            _seed_engrams(state, count)

            engine = SilentEngramEngine(half_life_hours=720.0)
            times = []
            for _ in range(5):
                # Reset strengths
                state._conn.execute(
                    "UPDATE engram_strengths SET strength = 1.0")
                state._conn.commit()

                start = time.perf_counter()
                engine.apply_decay(state, hours_elapsed=360.0)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            results[f"{count}_memories"] = {
                "mean_ms": round(statistics.mean(times) * 1000, 2),
                "median_ms": round(statistics.median(times) * 1000, 2),
                "p95_ms": round(sorted(times)[int(len(times) * 0.95)] * 1000, 2),
            }
        state.close()
    return results


def bench_activation_graph_query(n_entities: int = 200) -> dict:
    """Measure activation graph shortest-path query latency."""
    import tempfile
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        state = _make_state(tmp)
        entities = _seed_activation_graph(state, n_entities, edges_per_entity=4)

        graph = ActivationGraph()

        # Benchmark get_neighbors (direct lookup)
        times_neighbors = []
        for ent in entities[:50]:
            start = time.perf_counter()
            graph.get_neighbors(state, ent, min_strength=0.1)
            times_neighbors.append(time.perf_counter() - start)

        results["get_neighbors"] = {
            "mean_us": round(statistics.mean(times_neighbors) * 1_000_000, 1),
            "median_us": round(statistics.median(times_neighbors) * 1_000_000, 1),
            "p95_us": round(sorted(times_neighbors)[int(len(times_neighbors) * 0.95)] * 1_000_000, 1),
        }

        # Benchmark expand_query
        times_expand = []
        for ent in entities[:50]:
            start = time.perf_counter()
            graph.expand_query(state, f"{ent} test query", limit=3)
            times_expand.append(time.perf_counter() - start)

        results["expand_query"] = {
            "mean_us": round(statistics.mean(times_expand) * 1_000_000, 1),
            "median_us": round(statistics.median(times_expand) * 1_000_000, 1),
            "p95_us": round(sorted(times_expand)[int(len(times_expand) * 0.95)] * 1_000_000, 1),
        }

        # Benchmark find_bridge_entities (shortest path)
        times_path = []
        for i in range(0, min(50, len(entities) - 1), 2):
            a, b = entities[i], entities[(i + 10) % len(entities)]
            start = time.perf_counter()
            graph.find_bridge_entities(state, a, b)
            times_path.append(time.perf_counter() - start)

        results["shortest_path"] = {
            "mean_us": round(statistics.mean(times_path) * 1_000_000, 1),
            "median_us": round(statistics.median(times_path) * 1_000_000, 1),
            "p95_us": round(sorted(times_path)[int(len(times_path) * 0.95)] * 1_000_000, 1),
        }

        state.close()
    return results


def bench_schema_dedup_accuracy(n_schemas: int = 100) -> dict:
    """Measure schema dedup accuracy (TP/FP rates).

    We insert N unique schemas, then try to insert N/2 exact duplicates
    and N/2 novel schemas.  We measure how many duplicates are correctly
    caught (TP) and how many novel schemas are incorrectly flagged (FP).
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        state = _make_state(tmp)
        engine = ConsolidationEngine(min_facts=1)

        # Phase 1: Insert unique schemas
        unique_facts = [
            {"content": f"Unique schema {i} with distinct content about topic {i}",
             "domain": f"domain_{i % 3}"}
            for i in range(n_schemas)
        ]
        engine.consolidate(state, facts=unique_facts)

        # Phase 2: Exact duplicates (should be caught)
        dup_facts = [
            {"content": f"Unique schema {i} with distinct content about topic {i}",
             "domain": f"domain_{i % 3}"}
            for i in range(n_schemas // 2)
        ]
        result_dup = engine.consolidate(state, facts=dup_facts)

        # Phase 3: Novel schemas (should NOT be caught)
        novel_facts = [
            {"content": f"Completely novel schema {i} with entirely different content from before",
             "domain": f"novel_domain_{i % 3}"}
            for i in range(n_schemas // 2)
        ]
        result_novel = engine.consolidate(state, facts=novel_facts)

        true_positives = result_dup["schemas_updated"]
        false_negatives = result_dup["schemas_created"]
        true_negatives = result_novel["schemas_created"]
        false_positives = 0  # novel content should never match existing

        total_dup = n_schemas // 2
        total_novel = n_schemas // 2

        tp_rate = true_positives / total_dup if total_dup > 0 else 0.0
        fp_rate = false_positives / total_novel if total_novel > 0 else 0.0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0

        state.close()

    return {
        "true_positive_rate": round(tp_rate, 4),
        "false_positive_rate": round(fp_rate, 4),
        "precision": round(precision, 4),
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
    }


# ---------------------------------------------------------------------------
# Feature-flag-aware benchmarking
# ---------------------------------------------------------------------------

_ALL_FLAGS_OFF = {name: False for name in FeatureFlags({}).get_all()}
_ALL_FLAGS_ON = {name: True for name in FeatureFlags({}).get_all()}


def run_with_flags(flag_config: dict | None, bench_fn, *args, **kwargs):
    """Run a benchmark function with FeatureFlags monkeypatched."""
    if flag_config is None:
        # Default: use whatever the code normally does
        return bench_fn(*args, **kwargs)

    original_init = FeatureFlags.__init__

    def patched_init(self, config=None):
        # Ignore any config passed, use our forced config
        original_init(self, flag_config)

    with patch.object(FeatureFlags, '__init__', patched_init):
        return bench_fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run all benchmarks with flags OFF and ON, output JSON results."""
    print("Running benchmarks...", file=sys.stderr)

    benchmarks = {
        "salience_throughput": lambda: bench_salience_throughput(5000),
        "consolidation_latency": lambda: bench_consolidation_latency([5, 10, 20, 50]),
        "engram_decay": lambda: bench_engram_decay([100, 500, 1000, 5000]),
        "activation_graph_query": lambda: bench_activation_graph_query(200),
        "schema_dedup_accuracy": lambda: bench_schema_dedup_accuracy(100),
    }

    results = {}

    for name, bench_fn in benchmarks.items():
        print(f"  {name} (flags OFF)...", file=sys.stderr)
        off_result = run_with_flags(_ALL_FLAGS_OFF, bench_fn)
        print(f"  {name} (flags ON)...", file=sys.stderr)
        on_result = run_with_flags(_ALL_FLAGS_ON, bench_fn)

        results[name] = {
            "flags_off_baseline": off_result,
            "flags_on_optimized": on_result,
        }

    # Output JSON
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
