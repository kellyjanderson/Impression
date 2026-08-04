"""Reference Review compatibility adapter for shared markdown rendering."""

from impression_workbench.ui.markdown_context import (
    BlockedLinkDiagnostic,
    MarkdownContextRenderer as _SharedMarkdownContextRenderer,
    RenderedMarkdownContext,
)


class MarkdownContextRenderer(_SharedMarkdownContextRenderer):
    """Preserve the historical fixture-named call while delegating to the kit."""

    def render(
        self,
        *,
        fixture_id: str,
        source_digest: str,
        text: str,
    ) -> RenderedMarkdownContext:
        return super().render(
            context_id=fixture_id,
            source_digest=source_digest,
            text=text,
        )


__all__ = [
    "BlockedLinkDiagnostic",
    "MarkdownContextRenderer",
    "RenderedMarkdownContext",
]
