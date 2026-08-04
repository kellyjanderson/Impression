"""Compatibility imports for stale-result primitives owned by the workbench kit."""

from impression_workbench.async_core.staleness import (
    CancellationToken,
    CompletionDecision,
    LatestRequestTracker,
)

__all__ = ["CancellationToken", "CompletionDecision", "LatestRequestTracker"]
