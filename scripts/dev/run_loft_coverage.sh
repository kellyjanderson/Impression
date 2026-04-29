#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

"$ROOT_DIR/.venv/bin/pytest" \
  --cov=src/impression/modeling \
  --cov-report=term-missing:skip-covered \
  --cov-report=xml:"$ROOT_DIR/project/coverage/loft/coverage.xml" \
  --cov-report=html:"$ROOT_DIR/project/coverage/loft/html" \
  "$ROOT_DIR/tests/test_loft.py" \
  "$ROOT_DIR/tests/test_loft_api.py" \
  "$ROOT_DIR/tests/test_loft_correspondence.py" \
  "$ROOT_DIR/tests/test_loft_kernel.py" \
  "$ROOT_DIR/tests/test_loft_showcase.py" \
  "$ROOT_DIR/tests/test_loft_suite.py" \
  "$@"
