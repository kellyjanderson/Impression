"""Serialized durable write lane for notes and promotions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from typing import Callable


@dataclass(frozen=True)
class DurableWriteRequest:
    """A bounded write request rooted under an approved directory."""

    name: str
    root: Path
    lock_name: str
    timeout_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @property
    def lock_path(self) -> Path:
        return self.root / f"{self.lock_name}.lock"


@dataclass(frozen=True)
class DurableWriteResult:
    accepted: bool
    diagnostic: str | None = None


class DurableWriteLane:
    """Serialize write callbacks with in-process and lock-file guards."""

    def __init__(self) -> None:
        self._lock = Lock()

    def run(self, request: DurableWriteRequest, write: Callable[[], None]) -> DurableWriteResult:
        root = request.root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        lock_path = request.lock_path.resolve()
        if root != lock_path.parent:
            return DurableWriteResult(False, "lock_path_outside_root")

        start = monotonic()
        with self._lock:
            fd: int | None = None
            while fd is None:
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    if monotonic() - start >= request.timeout_seconds:
                        return DurableWriteResult(False, "lock_timeout")
                    sleep(0.01)
            try:
                write()
            except Exception as exc:
                return DurableWriteResult(False, str(exc))
            finally:
                if fd is not None:
                    os.close(fd)
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
        return DurableWriteResult(True)

