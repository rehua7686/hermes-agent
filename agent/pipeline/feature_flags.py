"""Feature flags for incremental pipeline optimization rollout.

Every flag defaults to ``False`` so the production path is unchanged until
an operator explicitly enables a flag via the config dict.  This makes
roll-out safe and rollback instant.

Usage::

    from agent.pipeline.feature_flags import FeatureFlags

    flags = FeatureFlags({"v2_salience": True, "v2_dedup": True})
    if flags.is_enabled("v2_salience"):
        ...

    # Or get the full picture:
    for name, on in flags.get_all().items():
        ...
"""

from __future__ import annotations

from typing import Dict


# Canonical list of all v2 feature flags.  Every entry here appears in
# ``get_all()`` and can be queried with ``is_enabled()``.
_FLAG_NAMES: tuple[str, ...] = (
    "v2_concurrency",
    "v2_salience",
    "v2_engram",
    "v2_dedup",
    "v2_activation",
    "v2_confidence",
    "v2_error_handling",
    "v2_emotion_decay",
    "v2_predict",
    "v2_timestamp",
)


class FeatureFlags:
    """Boolean feature-flag gate backed by a plain dict.

    Parameters
    ----------
    config : dict[str, bool] | None
        Mapping of flag names to their desired state.  Any flag *not*
        present in *config* is treated as ``False`` (safe default).
    """

    def __init__(self, config: Dict[str, bool] | None = None) -> None:
        self._flags: dict[str, bool] = {name: False for name in _FLAG_NAMES}
        if config:
            for name, value in config.items():
                if name in self._flags:
                    self._flags[name] = bool(value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_enabled(self, flag_name: str) -> bool:
        """Return the current state of *flag_name*.

        Unknown flag names silently return ``False`` so callers don't need
        to guard against typos with a try/except.
        """
        return self._flags.get(flag_name, False)

    def get_all(self) -> Dict[str, bool]:
        """Return a snapshot dict of every flag and its current state."""
        return dict(self._flags)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        enabled = [k for k, v in self._flags.items() if v]
        return f"<FeatureFlags enabled={enabled!r}>"
