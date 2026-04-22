#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

"$ROOT_DIR/.venv/bin/pytest" \
  --cov=src/impression \
  --cov-report=term-missing:skip-covered \
  --cov-report=xml:"$ROOT_DIR/project/coverage/coverage.xml" \
  --cov-report=html:"$ROOT_DIR/project/coverage/html" \
  "$@"
