"""Allowlisted Python-to-QML bridge registration contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


ALLOWED_BRIDGE_NAMES = frozenset(
    {
        "queueBridge",
        "selectionBridge",
        "codexBridge",
        "notesBridge",
        "artifactsBridge",
    }
)
FORBIDDEN_AUTHORITY_WORDS = ("filesystem", "fileSystem", "promotion", "promote", "shell", "git")


@dataclass(frozen=True)
class BridgeRecord:
    name: str
    bridge: Any
    authority: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("bridge name must not be empty")


@dataclass(frozen=True)
class BridgeAvailabilityDiagnostic:
    code: str
    message: str
    bridge_name: str | None = None


@dataclass(frozen=True)
class BridgeRegistry:
    records: Mapping[str, BridgeRecord] = field(default_factory=dict)

    def register(self, record: BridgeRecord) -> "BridgeRegistry":
        diagnostics = self.validate_record(record)
        if diagnostics:
            raise ValueError(diagnostics[0].code)
        next_records = dict(self.records)
        next_records[record.name] = record
        return BridgeRegistry(next_records)

    def validate_record(self, record: BridgeRecord) -> tuple[BridgeAvailabilityDiagnostic, ...]:
        diagnostics: list[BridgeAvailabilityDiagnostic] = []
        if record.name not in ALLOWED_BRIDGE_NAMES:
            diagnostics.append(
                BridgeAvailabilityDiagnostic(
                    "bridge-not-allowlisted",
                    "bridge object is not in the QML allowlist",
                    record.name,
                )
            )
        for authority in record.authority:
            if any(word == authority for word in FORBIDDEN_AUTHORITY_WORDS):
                diagnostics.append(
                    BridgeAvailabilityDiagnostic(
                        "bridge-authority-forbidden",
                        "bridge object exposes forbidden project authority",
                        record.name,
                    )
                )
        return tuple(diagnostics)

    def diagnostics(self, required: tuple[str, ...] = ()) -> tuple[BridgeAvailabilityDiagnostic, ...]:
        missing = [
            BridgeAvailabilityDiagnostic(
                "bridge-missing",
                "required bridge object was not registered",
                name,
            )
            for name in required
            if name not in self.records
        ]
        return tuple(missing)

