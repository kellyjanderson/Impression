#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
./.venv/bin/pytest -m reference_image "$@"
