# Agent Repo Path And Delivery Rules

- Date: 2026-04-24
- Status: delivered through repo workflow
- Path: ad hoc

## Summary

This ad hoc work adds a lightweight planning path alongside feature/spec work
and tightens the repository delivery rules so agents follow the same workflow
the project expects.

## Scope

This work covers:

- adding `project/adhoc/` as a durable planning space
- defining the required choice between feature-path work and ad-hoc-path work
- clarifying branch, commit, push, PR, merge, and delivery behavior in the
  agent docs

This work does not create a new feature architecture branch or a new feature
specification tree.

## Implementation Notes

The rule updates are carried in:

- `agents/git-and-github.md`
- `agents/workflow.md`
- `agents/index.md`
- `agents/documentation.md`
- `project/agents/index.md`
- `project/README.md`
- `project/adhoc/README.md`

The intended operating model is:

- active implementation happens on a feature branch
- the planning anchor is either a specification or an ad hoc work document
- the user must choose feature path or ad hoc path before implementation begins
- if the user does not choose, the agent should ask

## Verification Notes

Validation for this doc/rule change is the repository documentation rules test:

```bash
./.venv/bin/pytest tests/test_documentation_rules.py -q
```

## Related Documents

- `agents/git-and-github.md`
- `agents/workflow.md`
- `project/adhoc/README.md`
