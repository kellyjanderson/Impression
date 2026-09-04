# Parametric Design Patterns

This guide collects reusable programming and modeling patterns for parametric
CAD. These patterns are not Impression features; they are ways to structure
parametric models so that geometry, constraints, and host interactions remain
reusable and composable.

## Self-Applying Attachments

### Intent

Model an attachment not merely as a standalone solid, but as a parametric
operation that knows how to integrate itself into another model.

The caller should specify *where* the attachment belongs. The attachment should
own the knowledge of *how* it must modify the host to work correctly.

A self-applying attachment can therefore encapsulate:

- **retained geometry** — solids that become part of the host
- **through cutters** — geometry used to create openings through the host
- **local cutters** — pockets, recesses, or clearances that remove material
  without necessarily passing through the host
- **reference geometry** — representations of the real component, mating parts,
  insertion envelopes, or keep-out volumes
- **attachment frame** — the component-local origin and orientation used when it
  is placed on a host surface
- **physical constraints** — setback, insertion depth, clearance, minimum wall
  thickness, support requirements, and other real-world limits

This makes the unit of reuse a **host-transforming parametric component** rather
than simply a reusable solid.

### Example: USB-C Module Mount

Suppose a USB-C module must be mounted against an enclosure wall.

The component definition can generate several coordinated pieces of geometry:

- a positive mounting tray or bracket that remains in the final model
- a positive model of the USB-C receptacle positioned relative to that tray
- a connector-opening cutter long enough to pass completely through any
  plausible enclosure wall
- a local recess cutter that can remove material from the inside of the wall
  without cutting through the outside surface

The definition also knows the maximum distance the USB-C receptacle may sit
behind the finished exterior surface while still allowing a plug to seat fully.
That physical constraint determines the allowed setback and, when necessary,
the depth of the internal wall recess.

Conceptually, application looks like this:

1. accept the host enclosure and a point on the enclosure wall
2. determine the local wall orientation and surface normal
3. determine which side of the wall is inside and which is outside
4. orient the USB-C assembly so the receptacle faces outward and the mount stays
   inward
5. position the mounting tray and connector reference geometry from the local
   surface frame
6. calculate the required setback and any recess needed to keep the receptacle
   within its plug-seating limit
7. difference the connector opening and clearance geometry from the enclosure
8. union the retained mounting geometry into the enclosure
9. return or retain the modified enclosure according to the program's object
   semantics

The caller does not need to know the mount dimensions, required connector
setback, cutter lengths, or clearance rules. Those belong to the reusable
component definition.

### Functional Form

In a functional design, applying the attachment returns a new host model:

```text
modified_enclosure = attach_usb_c(enclosure, wall_point, ...)
```

The original enclosure remains unchanged.

### Object-Oriented Form

In an object-oriented design, the operation may modify the host directly:

```text
enclosure.attach(usb_c_mount, wall_point, ...)
```

The modeling pattern is identical in either form. Mutation semantics are an API
choice, not part of the geometric pattern.

### Why the Pattern Is Useful

Without this pattern, adding a connector to several enclosures often means
repeating the same sequence manually:

- position the connector
- calculate its required surface setback
- create the port opening
- remove enough interior wall material for the connector or plug
- create the bracket
- place and union the bracket

That approach duplicates both dimensions and design knowledge.

A self-applying attachment instead preserves the relationship between the real
hardware and every piece of geometry needed to integrate it. The same component
can then adapt to hosts with different wall thicknesses, orientations, shapes,
and construction styles.

### Other Uses

The same pattern applies naturally to:

- USB, HDMI, audio, power, and network connector mounts
- switches and buttons
- displays and indicator windows
- fans and vents
- cable glands and strain reliefs
- PCB trays and board-edge connectors
- threaded inserts and captive nuts
- hinges, latches, feet, handles, and brackets
- sensor windows and probe mounts

### Design Questions

Implementations should decide explicitly:

- whether an attachment location is expressed as a surface point with an
  inferred local frame, an explicit frame, or both
- how inside/outside detection behaves for open shells or non-watertight hosts
- whether constraints are procedural calculations or declarative data
- whether retained geometry and cutter geometry remain separate until the final
  application step
- how impossible placements are reported, such as walls that are too thick or
  neighboring geometry that blocks the required recess
- whether intermediate geometry is exposed for visualization, collision checks,
  debugging, or manual overrides before the boolean operations are committed
