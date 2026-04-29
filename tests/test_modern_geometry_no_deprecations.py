from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np

from impression.modeling import (
    make_box,
)
from impression.modeling.loft import loft
from impression.modeling.topology import Loop, Region, Section


_FORBIDDEN_DEPRECATION_FILES = (
    Path("src/impression/modeling/surface.py"),
    Path("src/impression/modeling/primitives.py"),
    Path("src/impression/modeling/loft.py"),
    Path("src/impression/modeling/threading.py"),
    Path("src/impression/modeling/hinges.py"),
)


def _simple_section() -> Section:
    return Section(
        regions=(
            Region(
                outer=Loop(
                    np.asarray(
                        [
                            [0.0, 0.0],
                            [1.0, 0.0],
                            [1.0, 1.0],
                            [0.0, 1.0],
                        ],
                        dtype=float,
                    )
                ),
                holes=(),
            ),
        )
    )


def _deprecation_messages(callable_obj) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        callable_obj()
    return [item for item in caught if issubclass(item.category, DeprecationWarning)]


def test_finished_surface_first_modules_do_not_import_or_call_legacy_deprecation_helpers(project_root) -> None:
    for relpath in _FORBIDDEN_DEPRECATION_FILES:
        text = (project_root / relpath).read_text()
        assert "warn_mesh_primary_api" not in text
        assert "warn_mesh_primary_backend" not in text
        assert "_legacy_mesh_deprecation" not in text


def test_finished_surface_first_geometry_paths_do_not_emit_deprecation_warnings() -> None:
    section = _simple_section()

    primitive_messages = _deprecation_messages(lambda: make_box(size=(1.0, 1.0, 1.0), backend="mesh"))
    loft_messages = _deprecation_messages(lambda: loft([section, section]))

    assert primitive_messages == []
    assert loft_messages == []
