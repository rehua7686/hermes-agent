# Merge Checklist: Memory Pipeline Optimization

Branch: `fix/memory-pipeline-optimization`
Date: 2026-06-01

## Pre-Merge Verification

- [ ] All tests pass (pytest green)
      **STATUS**: FAILING -- 1 of 28 tests fails (`test_lock_failure_raises_runtime_error`)
      due to closed-file-descriptor bug in `_file_lock` finally block.
- [ ] Feature flags default to OFF
      **STATUS**: FAILING -- `feature_flags.py` line 52 initializes all flags to `True`.
- [ ] No CRITICAL security issues
      **STATUS**: PASS -- No SQL injection, no destructive schema migrations.
- [ ] Backward compatible (flags OFF = original behavior)
      **STATUS**: FAILING -- Cannot verify because flags default to ON (see above).
- [ ] Rollback: set all flags to OFF to revert
      **STATUS**: FAILING -- Rollback path broken because default constructor enables all flags.
- [ ] Documentation: docs/memory-system-deep-analysis.md updated
      **STATUS**: PASS -- File exists (currently untracked).
- [ ] Code review: approved
      **STATUS**: PENDING.
- [ ] Benchmark: no performance regression with flags OFF
      **STATUS**: CANNOT VERIFY -- flags default to ON.

## Blockers (must resolve before merge)

1. **CRITICAL**: Change `feature_flags.py` line 52: `{name: True ...}` -> `{name: False ...}`
2. **HIGH**: Fix `_file_lock` finally block to handle already-closed file descriptor
3. **HIGH**: Stage and commit all changes (feature flags, tests, docs, modified files)
4. **LOW**: Remove temp files (`_b64_part1.txt`, `_edit_mp.py`, `*.bak`, `*.fixed`)
