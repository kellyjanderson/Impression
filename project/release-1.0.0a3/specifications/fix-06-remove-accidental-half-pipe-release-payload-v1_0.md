# Fix 06: Remove Accidental Half-Pipe Release Payload (v1.0)

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Extract the accidental half-pipe payload from Impression

- Input: the current half-pipe example, CAD adapter, dependency, and live references.
- Work: remove that payload from the product tree and metadata while preserving Git history.
- Output: source and distributions with no experiment files or `build123d` requirement.
- Complete when: source, wheel, sdist, and clean install are clear of the payload
  and approved imports/examples still pass.

## Problem And Outcome

The experimental half-pipe branch was merged into `main` even though experimental
branches were not approved for release. The example, build123d adapter, and
`build123d` dependency must be absent from Impression while their history remains
recoverable in Git.

## Scope

- Remove `examples/half_pipe.py` and `src/impression/cad.py`.
- Remove `build123d` from runtime dependencies when no approved code uses it.
- Remove or update only references that exist solely for that experiment.

Not in scope: deleting Git history, publishing the experiment elsewhere, or
removing unrelated CAD functionality.

## Implementation Routing

- `examples/half_pipe.py`, `src/impression/cad.py`, `pyproject.toml`.
- Package-content and dependency regression tests.

## Contract

Input is the current release tree. Output is a tree and built distribution with
no half-pipe files, import surface, or `build123d` runtime requirement. Git commit
history remains the recovery mechanism. No unresolved product decision remains:
the experiment is excluded from a3.

## Acceptance Criteria

- Repository and built wheel/sdist exclude both experimental modules.
- Clean installation does not install `build123d` through Impression metadata.
- Package imports and approved examples still pass.
- Release notes identify the extraction as payload correction, not a feature loss.

## Verification

[Paired test specification](../test-specifications/fix-06-remove-accidental-half-pipe-release-payload-v1_0.md)
