---
name: modern-surface-first-guardrails
description: Protect finished Impression surface-first geometry modules from stale legacy deprecation wiring while allowing deprecation only in genuinely mesh-primary areas.
---

# Modern Surface-First Guardrails

Finished modern geometry domains should not carry stale legacy deprecation wiring.

## Guardrail

There should be no legacy deprecation helper usage in finished surface-first domains such as:

* `src/impression/modeling/surface.py`
* `src/impression/modeling/primitives.py`
* `src/impression/modeling/loft.py`
* `src/impression/modeling/threading.py`
* `src/impression/modeling/hinges.py`

If deprecation markers appear there, either:

* a still-mesh-centric capability was missed and needs proper migration work
* or the deprecation marker is stale and should be removed

## Scoped Exception

Legacy mesh deprecation may still belong in:

* mesh-centric geometry generators that are not yet surface-first
* rendering pipelines
* export paths
* repair tools
* compatibility bridges
* mesh-focused analysis helpers
