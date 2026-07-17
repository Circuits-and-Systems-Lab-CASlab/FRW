#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLUTION="${1:-23}"
TOLERANCE="${2:-1e-10}"
MAX_ITERATIONS="${3:-20000}"
BLOCK_WIDTH="${4:-1}"

BIN_PATH="$ROOT_DIR/data/cube_tables_n${RESOLUTION}.bin"
NPZ_PATH="$ROOT_DIR/data/cube_tables_n${RESOLUTION}.npz"

echo "-------------compiling GGFT...-------------"
make -C "$ROOT_DIR"
echo "-------------generating GGFT tables...-------------"
"$ROOT_DIR/bin/ggft" \
    --output "$BIN_PATH" \
    --resolution "$RESOLUTION" \
    --block-width "$BLOCK_WIDTH" \
    --tolerance "$TOLERANCE" \
    --max-iterations "$MAX_ITERATIONS"

# generate .npz
echo "-------------generating .npz from .bin...-------------"
python3 "$ROOT_DIR/extract_binary.py" "$BIN_PATH" "$NPZ_PATH"
