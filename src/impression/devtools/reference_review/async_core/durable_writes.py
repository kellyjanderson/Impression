"""Compatibility imports for durable-write primitives owned by the workbench kit."""

from impression_workbench.async_core.durable_writes import (
    DurableWriteLane,
    DurableWriteRequest,
    DurableWriteResult,
)

__all__ = ["DurableWriteLane", "DurableWriteRequest", "DurableWriteResult"]
