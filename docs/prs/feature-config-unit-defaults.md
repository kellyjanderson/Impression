# PR: Add configurable unit defaults

- **Branch**: `feature/config-unit-defaults`
- **Status**: Open (PR #6)
- **Owner/Reviewer**: @kellyjanderson

## Summary
1. Document the config-driven units feature in the feature pipeline.
2. Create `~/.impression/impression.cfg` automatically with a default `units` entry and guidance on valid values.
3. Surface the configured units inside the CLI (preview/export) and label preview axes accordingly so users see which units are active end-to-end.

## Testing
- Manual: Not run (CLI + preview interactions require GUI runtime).
- Automated: Not run (no unit-aware tests yet).
