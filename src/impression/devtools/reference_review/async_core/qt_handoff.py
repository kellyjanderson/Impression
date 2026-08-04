"""Compatibility imports for UI handoff primitives owned by the workbench kit."""

from impression_workbench.async_core.qt_handoff import (
    DiagnosticSanitizerOptions,
    SanitizedDiagnostic,
    UICompletionBridge,
    WorkbenchUiHandoff,
    bound_diagnostic_stream_excerpt,
    sanitize_diagnostic_text,
    sanitize_error_text,
)

__all__ = [
    "DiagnosticSanitizerOptions",
    "SanitizedDiagnostic",
    "UICompletionBridge",
    "WorkbenchUiHandoff",
    "bound_diagnostic_stream_excerpt",
    "sanitize_diagnostic_text",
    "sanitize_error_text",
]
