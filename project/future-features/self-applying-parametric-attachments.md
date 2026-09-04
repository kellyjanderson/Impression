# Self-Applying Parametric Attachments

## Idea

Capture a reusable parametric-modeling pattern for components that are intended
to be added to an existing model rather than modeled as isolated parts.

A component definition should contain enough geometry and constraints to apply
itself to a host object at a requested attachment location.

The pattern does not require any Impression-specific feature. It is a modeling
and API pattern that can be implemented with ordinary solid modeling,
transformations, unions, and differences.

## Example: USB-C Module Mount

Consider a USB-C module mount that must be added to an enclosure.

The reusable component definition would generate several coordinated pieces of
geometry:

- the positive mounting tray or bracket that remains in the final model
- a positive model of the USB-C connector body, positioned relative to that
  tray
- a through-wall cutter representing the connector opening; this should extend
  far enough along the surface normal to pass through any plausible enclosure
  wall
- a local recess / clearance cutter that removes material from the inside of
  the enclosure wall without cutting through the outside surface

The component also carries physical placement constraints. In particular, it
knows the maximum allowable distance between the connector receptacle and the
outside enclosure surface for a USB-C plug to fully seat.

That constraint determines how deeply the local recess must cut into the wall
and/or how the mounting tray must be positioned relative to the wall.

## Application Interface

Conceptually, the operation takes:

- a host enclosure or solid body
- a point on the enclosure wall where the component should attach
- component-specific parameters, if any

The attachment operation then:

1. determines the local wall orientation and surface normal at the requested
   point
2. determines which side of the wall is inside versus outside
3. orients the component so the connector faces outward and the mount remains
   inward
4. places the mounting geometry and connector reference geometry from the local
   surface frame
5. calculates the required connector setback and any wall recess needed to
   satisfy the plug-seating constraint
6. differences the through-wall opening and local clearance/recess geometry from
   the enclosure
7. unions the mounting tray or other retained attachment geometry into the
   enclosure
8. returns the resulting enclosure

The important point is that the caller supplies *where* the component belongs,
but the component definition owns the knowledge of *how* it must modify the host
body to work correctly.

## Component Contract

A reusable attachment can be thought of as a small parametric assembly with an
integration contract.

It may define:

- **retained geometry** — solids to union with the host
- **through cutters** — solids intended to create openings through the host
- **local cutters** — solids intended to create recesses, pockets, or clearance
  without necessarily penetrating the host
- **reference geometry** — non-final geometry representing the attached device,
  mating connector, keep-out volume, or other spatial constraint
- **attachment frame** — the component-local origin and orientation used when
  placed against a host surface
- **placement constraints** — allowable setback, insertion depth, wall
  thickness, clearance, minimum support material, or similar limits

This separates the geometry used to *describe the attached thing* from the
boolean roles needed to *integrate it into the host*.

## Generalization

The same pattern applies to many enclosure and fixture features:

- USB, HDMI, audio, power, or network connector mounts
- switches and buttons
- displays and indicator windows
- fans and vents
- cable glands and strain reliefs
- PCB trays and board-edge connectors
- threaded inserts and captive nuts
- hinges, latches, feet, handles, and brackets
- sensor windows and probe mounts

A well-defined attachment can therefore become a reusable parametric building
block rather than a collection of dimensions that must be manually recreated in
each enclosure.

## Functional vs. Object-Oriented Semantics

The pattern is independent of mutation semantics.

A functional API would conceptually behave like:

```text
modified_enclosure = attach_usb_c(enclosure, wall_point, ...)
```

and return a modified copy of the enclosure while leaving the original
unchanged.

An object-oriented API could instead behave like:

```text
enclosure.attach(usb_c_mount, wall_point, ...)
```

and modify the original enclosure object.

The geometric contract is the same in either case. The significant abstraction
is that the attachment encapsulates its integration geometry and physical
placement rules.

## Design Value

This pattern moves enclosure modeling away from ad-hoc boolean construction and
toward reusable, physically informed components.

The useful unit of reuse is no longer merely a solid model of a USB-C mount. It
is a **host-transforming parametric feature** that knows:

- what geometry must be added
- what geometry must be removed
- which side of the host is inside and outside
- how it must be oriented
- how close the real device must be to the finished exterior surface
- what clearances are necessary for the real mating part to function

That makes the component reusable across enclosures with different wall
thicknesses, orientations, shapes, and construction styles while preserving the
real-world constraints of the mounted hardware.

## Open Questions

- whether attachment points should be represented only by a surface point and
  inferred local frame, or optionally by an explicit frame
- how robust inside/outside detection should work for open shells or
  non-watertight host geometry
- whether placement constraints should be expressed procedurally or as a
  declarative constraint set
- whether retained geometry and cutter geometry should remain separate until the
  final boolean application step
- how to report impossible placements, such as a wall that is too thick or
  nearby geometry that prevents the required recess
- whether a generic attachment protocol should expose intermediate geometry for
  visualization, collision checking, or manual override before application
