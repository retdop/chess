#!/usr/bin/env bash
# Run once on a fresh GPU instance to download and decompress the puzzle database.
set -euo pipefail

mkdir -p data checkpoints

echo "==> Downloading Lichess puzzle database (~250 MB compressed)..."
wget -c -O data/lichess_puzzles.csv.zst \
    https://database.lichess.org/lichess_db_puzzle.csv.zst

echo "==> Decompressing..."
zstd -d data/lichess_puzzles.csv.zst -o data/lichess_puzzles.csv

echo "==> Done."
echo ""
echo "Next steps:"
echo "  uv sync"
echo "  uv run python src/train.py"
