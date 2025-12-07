# Groups

Groups let you manipulate several meshes together without baking them into a single boolean result. This keeps the source geometry editable until you explicitly fuse or export.

## API
- `group(meshes)` → `MeshGroup`
- `MeshGroup.add(mesh)` — append another mesh
- `MeshGroup.translate(offset)` — move all children
- `MeshGroup.rotate(axis, angle_deg, origin=(0,0,0))` — rotate all children about an axis
- `MeshGroup.scale((sx, sy, sz))` — scale all children
- `MeshGroup.to_multiblock()` — returns a `pv.MultiBlock` (keeps child colors, ideal for preview)
- `MeshGroup.to_polydata()` — merges and triangulates into one mesh (for export/CSG)

All transforms are cumulative and non-destructive to the original meshes.

## Example
```python
from impression.modeling import make_box, make_cylinder, group

def build():
    box = make_box(size=(2.0, 1.0, 0.5), center=(0.0, 0.0, 0.25))
    cyl = make_cylinder(radius=0.4, height=1.0, center=(0.0, 0.0, 0.5))
    grp = group([box, cyl])
    grp.translate((0.5, 0.0, 0.2))
    grp.rotate(axis=(0, 0, 1), angle_deg=30)
    return grp.to_multiblock()
```
Preview it:
```bash
impression preview docs/examples/groups/group_example.py
```
If you need a single mesh (e.g., for STL), call `grp.to_polydata()` and save it or pass it into your CSG helpers.
