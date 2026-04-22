# Testing Rules

## Coverage Discipline

When an implementation pass finishes, the agent must run coverage for at least
the code it just changed.

Coverage is not optional follow-up work. It is part of finishing the change.

At minimum:

- run a focused coverage command for the touched subsystem when one exists
- refresh XML and HTML artifacts under `project/coverage/`

When the agent checks the final remaining box in
`project/planning/progression.md`, it must also run a full repository coverage
pass and refresh:

- `project/coverage/coverage.xml`
- `project/coverage/html/`

## Model Output Verification

Any capability that outputs a model must have a durable reference-artifact
test before the work is considered complete.

By default, that means:

- a rendered image reference
- an exported STL reference

First execution of a new named fixture should bootstrap dirty references.
Later executions should compare against the clean reference when present,
otherwise against the dirty reference.

If the output diff is not clean under the fixture's comparison contract, the
test must fail.

Agents should not treat model-outputting work as done while reference-artifact
coverage is still missing.

## Modern Surface-First Modules

The finished modern geometry domains should not carry legacy deprecation wiring.

For this project, that means there should be no legacy deprecation helper usage
in:

- `src/impression/modeling/surface.py`
- `src/impression/modeling/primitives.py`
- `src/impression/modeling/loft.py`
- `src/impression/modeling/threading.py`
- `src/impression/modeling/hinges.py`

If deprecation markers or helper calls appear there, one of two things is true:

1. a still-mesh-centric capability was missed and needs proper migration work
2. the deprecation marker is stale and should be removed

Agents should prefer adding an automated guard test for this rule so it remains
durable.

## Where Deprecation Still Belongs

Legacy mesh deprecation is still valid in mesh-centric geometry generators that
have not yet completed their surface-first migration.

Mesh-centric behavior also remains valid in downstream infrastructure such as:

- rendering pipelines
- export paths
- watertightness and repair tools
- compatibility bridges
- mesh-focused analysis helpers

The rule is not "no mesh anywhere." The rule is:

- no stale geometry deprecation markers in finished surface-first domains
- deprecation only where a geometry capability is still genuinely mesh-primary
