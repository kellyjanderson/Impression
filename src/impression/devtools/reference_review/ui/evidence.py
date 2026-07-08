"""Component gallery, screenshot, and accessibility evidence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .packaging import qml_resource_root
from .shell import launch_workbench

REQUIRED_STATE_COVERAGE = frozenset(
    {"hover", "focus", "disabled", "loading", "error", "empty", "overflow"}
)


@dataclass(frozen=True)
class ComponentStateScenario:
    component: str
    state: str
    fixture_label: str

    def safe(self) -> bool:
        text = f"{self.component} {self.state} {self.fixture_label}".lower()
        return "/" not in text and all(word not in text for word in ("secret", "token", "password"))


@dataclass(frozen=True)
class ComponentGallery:
    scenarios: tuple[ComponentStateScenario, ...]

    @property
    def states(self) -> frozenset[str]:
        return frozenset(scenario.state for scenario in self.scenarios)

    @property
    def valid(self) -> bool:
        return all(scenario.safe() for scenario in self.scenarios)

    def missing_states(self) -> tuple[str, ...]:
        return tuple(sorted(REQUIRED_STATE_COVERAGE - self.states))


def default_component_gallery() -> ComponentGallery:
    return ComponentGallery(
        (
            ComponentStateScenario("StatusBadge", "hover", "Synthetic ready fixture"),
            ComponentStateScenario("StatusBadge", "focus", "Synthetic focused fixture"),
            ComponentStateScenario("IconButton", "disabled", "Synthetic disabled action"),
            ComponentStateScenario("ArtifactPanel", "loading", "Synthetic thumbnail loading"),
            ComponentStateScenario("NotesPanel", "error", "Synthetic save failure"),
            ComponentStateScenario("QueuePanel", "empty", "Synthetic empty queue"),
            ComponentStateScenario("MarkdownPanel", "overflow", "Synthetic long context"),
        )
    )


@dataclass(frozen=True)
class StateMatrixRecord:
    state: str
    covered: bool
    evidence: str | None = None


@dataclass(frozen=True)
class AccessibilityOverflowMatrix:
    records: tuple[StateMatrixRecord, ...]

    @property
    def valid(self) -> bool:
        return all(record.covered for record in self.records)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(record.state for record in self.records if not record.covered)


def build_accessibility_overflow_matrix(
    scenarios: Iterable[ComponentStateScenario],
) -> AccessibilityOverflowMatrix:
    states = {scenario.state: scenario.component for scenario in scenarios}
    return AccessibilityOverflowMatrix(
        tuple(
            StateMatrixRecord(
                state=state,
                covered=state in states,
                evidence=states.get(state),
            )
            for state in sorted(REQUIRED_STATE_COVERAGE)
        )
    )


@dataclass(frozen=True)
class ScreenshotScenario:
    name: str
    width: int = 1180
    height: int = 760

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError("scenario name must be a safe file stem")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("scenario dimensions must be positive")


@dataclass(frozen=True)
class ScreenshotArtifact:
    scenario: ScreenshotScenario
    path: Path


@dataclass(frozen=True)
class ScreenshotRunReport:
    artifacts: tuple[ScreenshotArtifact, ...]
    missing_states: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.diagnostics and not self.missing_states and bool(self.artifacts)


class ScreenshotScenarioRunner:
    def __init__(self, *, max_scenarios: int = 8) -> None:
        self.max_scenarios = max_scenarios

    def run(
        self,
        *,
        output_root: Path,
        scenarios: tuple[ScreenshotScenario, ...],
        gallery: ComponentGallery | None = None,
    ) -> ScreenshotRunReport:
        if len(scenarios) > self.max_scenarios:
            return ScreenshotRunReport((), diagnostics=("too_many_scenarios",))
        output_root.mkdir(parents=True, exist_ok=True)
        gallery = gallery or default_component_gallery()
        launch = launch_workbench(
            qml_path=qml_resource_root() / "ComponentGallery.qml",
            offscreen=True,
        )
        if not launch.launched or launch.engine is None:
            return ScreenshotRunReport((), gallery.missing_states(), launch.diagnostics)
        root = launch.engine.rootObjects()[0]
        artifacts: list[ScreenshotArtifact] = []
        for scenario in scenarios:
            root.setWidth(scenario.width)
            root.setHeight(scenario.height)
            image = root.grabWindow()
            path = output_root / f"{scenario.name}.png"
            if image.isNull() or not image.save(str(path)):
                return ScreenshotRunReport(
                    tuple(artifacts),
                    gallery.missing_states(),
                    (f"screenshot_failed:{scenario.name}",),
                )
            artifacts.append(ScreenshotArtifact(scenario, path))
        return ScreenshotRunReport(tuple(artifacts), gallery.missing_states())

