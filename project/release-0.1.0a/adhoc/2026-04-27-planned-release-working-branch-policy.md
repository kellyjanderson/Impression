# Planned Release Working Branch Policy

- Date: 2026-04-27
- Status: in progress
- Path: ad hoc

## Summary

Adopt a planned-release workflow where version planning is merged first, then a
release working branch is created from `main`, and all feature work for that
release integrates through the working branch until the release is complete.

## Scope

This work covers:

- codifying release working-branch rules in the active agent workflow overlays
- documenting the same policy in the project-facing planning docs
- using that policy to establish the first release working branch for `0.1.0.a`

This work does not define the feature contents of `0.1.0.a` itself.

## Policy

For planned releases:

- release planning may happen on a feature branch
- once planning is stable, merge that planning branch into `main`
- create a release working branch from updated `main`
- name the working branch `working/<release>` when practical
- merge all feature branches for that release into the working branch
- merge the working branch into `main` when the planned release is done

## Rationale

This keeps:

- `main` as completed integrated history
- feature branches as isolated units of work
- the working branch as the visible release-integration surface

## Verification Notes

Expected verification:

- policy text exists in the active workflow overlays
- project-facing planning docs reflect the same rule
- the `0.1.0.a` planning branch is merged
- the `working/0.1.0.a` branch exists from updated `main`
