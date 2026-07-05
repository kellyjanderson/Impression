from __future__ import annotations

from pathlib import Path

from impression.devtools.reference_review.ui import (
    ComponentGallery,
    ComponentStateScenario,
    ScreenshotScenario,
    ScreenshotScenarioRunner,
    build_accessibility_overflow_matrix,
    default_component_gallery,
    verify_qml_resource_layout,
)


def test_component_gallery_uses_synthetic_safe_fixture_data() -> None:
    gallery = default_component_gallery()

    assert gallery.valid
    assert gallery.missing_states() == ()
    assert {scenario.state for scenario in gallery.scenarios} >= {
        "hover",
        "focus",
        "disabled",
        "loading",
        "error",
        "empty",
        "overflow",
    }


def test_component_gallery_rejects_secret_or_local_path_fixture_data() -> None:
    gallery = ComponentGallery(
        (ComponentStateScenario("NotesPanel", "error", "/tmp/secret-token"),)
    )

    assert not gallery.valid


def test_accessibility_overflow_matrix_reports_missing_and_covered_states() -> None:
    full = build_accessibility_overflow_matrix(default_component_gallery().scenarios)
    partial = build_accessibility_overflow_matrix(
        (ComponentStateScenario("StatusBadge", "focus", "Synthetic focused fixture"),)
    )

    assert full.valid
    assert full.missing == ()
    assert not partial.valid
    assert "overflow" in partial.missing


def test_qml_resource_layout_includes_component_gallery() -> None:
    result = verify_qml_resource_layout()

    assert result.valid


def test_screenshot_runner_writes_bounded_png_evidence(tmp_path: Path) -> None:
    runner = ScreenshotScenarioRunner(max_scenarios=2)

    report = runner.run(
        output_root=tmp_path / "screenshots",
        scenarios=(ScreenshotScenario("gallery-default", width=640, height=420),),
    )

    assert report.valid
    assert report.artifacts[0].path.exists()
    assert report.artifacts[0].path.suffix == ".png"


def test_screenshot_runner_reports_unbounded_scenario_sets(tmp_path: Path) -> None:
    runner = ScreenshotScenarioRunner(max_scenarios=1)

    report = runner.run(
        output_root=tmp_path / "screenshots",
        scenarios=(
            ScreenshotScenario("one"),
            ScreenshotScenario("two"),
        ),
    )

    assert not report.valid
    assert report.diagnostics == ("too_many_scenarios",)
