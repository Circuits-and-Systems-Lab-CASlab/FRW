#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLUTION="${1:-200}"
TOLERANCE="${2:-1e-12}"
MAX_ITERATIONS="${3:-20000}"

BIN_PATH="$ROOT_DIR/data/cube_gft_n${RESOLUTION}.bin"
NPZ_PATH="$ROOT_DIR/data/cube_gft_n${RESOLUTION}.npz"

make -C "$ROOT_DIR"
"$ROOT_DIR/bin/ggft_homogeneous" \
    "$BIN_PATH" \
    "$RESOLUTION" \
    "$TOLERANCE" \
    "$MAX_ITERATIONS"

python3 "$ROOT_DIR/extract_binary.py" "$BIN_PATH" "$NPZ_PATH"
