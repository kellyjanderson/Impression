---
name: planning
description: Create and maintain progression-style planning documents that sequence final leaf specifications and paired test specifications in dependency order.
---

# Planning

Planning documents define the implementation and verification sequence for specifications.

## Purpose

Planning answers:

* what must be completed before other work can proceed
* what order executable specification leaves should be implemented in
* what has been completed

Planning does not define version intent. That belongs to `release-definitions`.

## Core Rule

Only these belong in progression-style planning:

* final leaf specifications
* paired feature test specifications

Parent or umbrella specifications must not appear there.

## Structure

Planning documents should:

* group work into implementation lanes when useful
* order items by dependency
* use checkboxes for completion state

Common lanes include:

* core functionality
* obligate specifications
* polish specifications

## Completion Tracking

Keep implementation and verification visible separately.

For feature leaves:

* check the feature leaf when implementation is complete
* leave the paired test-spec item unchecked until verification is complete

## Relationship To Other Documents

* architecture defines the system
* specifications define implementation work
* release definitions define holistic version scope
* planning defines execution order
