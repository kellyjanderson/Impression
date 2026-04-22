from __future__ import annotations

from pathlib import Path

from fontTools.ttLib import TTFont, TTLibFileIsCollectionError


TEXT_GLYPH_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Supplemental/Helvetica.ttc"),
)


def font_supports_text(font_path: Path, content: str) -> bool:
    try:
        font = TTFont(str(font_path))
    except TTLibFileIsCollectionError:
        font = TTFont(str(font_path), fontNumber=0)
    cmap = font.getBestCmap() or {}
    return all(character == " " or ord(character) in cmap for character in content)


def require_glyph_capable_font(content: str = "SURFACE") -> Path:
    for candidate in TEXT_GLYPH_FONT_CANDIDATES:
        if candidate.exists() and font_supports_text(candidate, content):
            return candidate
    raise FileNotFoundError(f"No glyph-capable test font found for {content!r}.")
