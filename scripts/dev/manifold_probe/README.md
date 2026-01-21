# Manifold Build Probe

This folder is a tiny harness to test building `manifold3d` from source without
installing it into the project environment.

## Why

On macOS + newer Python versions, `manifold3d` may try to build in parallel and
fail if Intel TBB is missing. The probe shows which flags are required and
captures a build wheel for inspection.

## Usage

```bash
# build in serial (no TBB required)
scripts/dev/manifold_probe/probe.sh serial

# attempt default (parallel) build
scripts/dev/manifold_probe/probe.sh parallel
```

The wheel is written to `scripts/dev/manifold_probe/dist/`.

## Notes

- The probe uses `.venv/bin/python` if it exists, otherwise falls back to
  system Python.
- The serial build sets `CMAKE_ARGS="-DMANIFOLD_PARALLEL=OFF"`.
