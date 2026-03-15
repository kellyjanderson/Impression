## GitHub Release Policy

When generating a release entry:

1. Use the git tag as the version.
2. The title must include both the version and a short milestone theme.

Format:
{TAG} — {Milestone Theme}

Release body must contain the following sections:

## Overview
Explain what milestone this release represents.

## Major Work
List the primary capabilities or systems introduced.

## Technical Changes
Describe significant implementation changes.

## Stability
Explain the maturity and expected reliability of the release.

## Known Limitations
List important unsupported cases or constraints.

## Next Direction
Briefly describe the next development focus.

Rules:
- Focus on capabilities, not commit messages.
- Do not fabricate features.
- Be concise and technical.
- Alpha releases should clearly state that APIs may change.