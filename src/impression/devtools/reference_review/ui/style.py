"""Style token and shared component contracts for the QML workbench."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StyleTokenRecord:
    name: str
    value: str


@dataclass(frozen=True)
class ComponentContractRecord:
    name: str
    role: str
    overflow_policy: str
    stable_size: bool = True


@dataclass(frozen=True)
class StyleLoadDiagnostic:
    code: str
    message: str


def load_style_tokens() -> tuple[StyleTokenRecord, ...]:
    return (
        StyleTokenRecord("surface", "#f7f7f2"),
        StyleTokenRecord("panel", "#ffffff"),
        StyleTokenRecord("border", "#c9c8be"),
        StyleTokenRecord("text", "#242622"),
        StyleTokenRecord("accent", "#2f6f73"),
        StyleTokenRecord("warning", "#9b5d17"),
    )


def component_contracts() -> tuple[ComponentContractRecord, ...]:
    return (
        ComponentContractRecord("IconButton", "compact command", "fixed-size elided tooltip"),
        ComponentContractRecord("TextField", "bounded text entry", "single-line elide"),
        ComponentContractRecord("StatusBadge", "state summary", "fixed-height elide"),
        ComponentContractRecord("SplitPane", "primary layout", "fill available space"),
    )

