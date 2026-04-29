# Release 0.0.3a2

## Intent

`0.0.3a2` is the corrected prerelease that publishes the rearchitecture work
from `main` and repairs the release docs bundle so downloaded docs match the
actual current project state.

## Why This Release Exists

`0.0.3a0` and `0.0.3a1` were cut from the rearchitecture branch before that
branch had been merged into `main`.

This release corrects that by:

- merging the rearchitecture branch into `main`
- publishing a new prerelease from the merged mainline commit
- fixing docs asset packaging so downloaded docs bundles are complete

## Delivered Outcomes

- the surface-first modeling and testing-architecture tranche is now released
  from `main`, not only from a feature branch
- docs bundles produced for release now include real docs content again
- the distributed docs now include:
  - `docs/agents/`
  - `docs/skills/impression/`
  - the updated surface-first loft guidance

## Key Corrections

- fixed [package_docs_zip.py](../../scripts/release/package_docs_zip.py) so it
  actually emits docs files into the release zip
- shipped the folder-based agent guide:
  - [docs/agents/index.md](../../docs/agents/index.md)
  - [docs/agents/loft.md](../../docs/agents/loft.md)
- shipped the installable Impression skill:
  - [docs/skills/index.md](../../docs/skills/index.md)
  - [docs/skills/impression/SKILL.md](../../docs/skills/impression/SKILL.md)

## Inherited Scope

This release inherits the full modeling/testing tranche already documented in:

- [Release 0.0.3a1](../release-0.0.3a1/README.md)

That includes the surfaced modeling stack, bounded surfaced CSG, top-level
testing architecture work, and the broader documentation/project-structure
rebuild.

## Verification

Targeted verification for the rerelease path:

- `./.venv/bin/pytest tests/test_documentation_rules.py tests/test_surface_csg_docs.py -q`
- `python3 scripts/release/package_docs_zip.py --ref test-docs`
- verified the resulting zip contains:
  - `docs/skills/impression/SKILL.md`
  - `docs/skills/impression/references/feature-map.md`
  - `docs/skills/impression/references/loft.md`
  - `docs/agents/index.md`
  - `docs/modeling/loft.md`

## Notes

- this is still an alpha release
- the purpose of `0.0.3a2` is not a new modeling tranche; it is the corrected
  publication of the already-landed rearchitecture work plus the completed docs
  distribution path
