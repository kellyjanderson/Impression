# Agent Directives

This folder is a local directive workspace for agent-facing process documents
that should not automatically be treated as part of the published `agents/`
skill set.

## Folders

- [potentials/](potentials/index.md)
  These are processes that are in the process of being developed and tested.
  Treat them like any other agent directive, but they should never be added to
  any skill set, published directive set, or otherwise be considered canonical
  directives until they are explicitly moved out of this folder and into a
  folder without this restriction. For the workspace containing this
  `.agents/` folder, agents should treat directives under `potentials/` as
  local workspace directives only, never as canonical directives.
