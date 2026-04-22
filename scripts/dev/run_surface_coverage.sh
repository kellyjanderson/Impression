#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

"$ROOT_DIR/.venv/bin/pytest" \
  --cov=src/impression/modeling \
  --cov-report=term-missing:skip-covered \
  --cov-report=xml:"$ROOT_DIR/project/coverage/surface/coverage.xml" \
  --cov-report=html:"$ROOT_DIR/project/coverage/surface/html" \
  "$ROOT_DIR/tests/test_surface.py" \
  "$ROOT_DIR/tests/test_surface_kernel.py" \
  "$@"
