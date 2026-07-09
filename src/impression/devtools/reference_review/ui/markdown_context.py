"""Markdown context rendering and link policy for QML panels."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import markdown


_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


@dataclass(frozen=True)
class BlockedLinkDiagnostic:
    label: str
    target: str
    reason: str


@dataclass(frozen=True)
class RenderedMarkdownContext:
    cache_key: str
    html: str
    blocked_links: tuple[BlockedLinkDiagnostic, ...] = ()


class MarkdownContextRenderer:
    def __init__(self) -> None:
        self._cache: dict[str, RenderedMarkdownContext] = {}

    def render(self, *, fixture_id: str, source_digest: str, text: str) -> RenderedMarkdownContext:
        cache_key = hashlib.sha256(f"{fixture_id}:{source_digest}:{text}".encode("utf-8")).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        blocked: list[BlockedLinkDiagnostic] = []

        def replace(match: re.Match[str]) -> str:
            label, target = match.group(1), match.group(2)
            if target.startswith(("http://", "https://")):
                blocked.append(BlockedLinkDiagnostic(label, target, "external-link-blocked"))
                return label
            return match.group(0)

        sanitized = _LINK_PATTERN.sub(replace, text)
        rendered = RenderedMarkdownContext(
            cache_key=cache_key,
            html=markdown.markdown(sanitized),
            blocked_links=tuple(blocked),
        )
        self._cache[cache_key] = rendered
        return rendered

