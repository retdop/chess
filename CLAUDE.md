# CLAUDE.md

## Project overview

Chess Puzzle Rating Predictor — a deep learning model that predicts Lichess puzzle difficulty ratings from board positions. Uses a transformer encoder over per-square piece embeddings of FEN strings to regress a single Elo rating value.

## Tech stack

- **Language:** Python 3.13
- **ML framework:** PyTorch (transformer encoder)
- **Chess library:** python-chess (FEN parsing)
- **Data:** pandas, numpy
- **Visualization:** matplotlib
- **Cloud training:** Modal (serverless GPU)
- **Package manager:** uv

## Repository layout

```
src/model.py       - ChessPuzzleTransformer architecture (pre-norm transformer)
src/dataset.py     - PuzzleDataset, FEN-to-tensor encoding
src/train.py       - Local training loop with CLI args
src/evaluate.py    - Evaluation metrics and scatter plots
src/train_modal.py - Modal wrapper for cloud GPU training
tests/test_smoke.py - CPU-only smoke tests (no data files needed)
setup.sh           - Downloads Lichess puzzle CSV (~800MB)
```

## Common commands

```bash
# Install dependencies
uv sync --extra dev

# Run linter
uv run ruff check src tests

# Run type checker
uv run ty check src tests

# Run tests
uv run pytest tests/ -v

# Train locally
uv run python src/train.py

# Train on Modal (GPU)
modal run src/train_modal.py

# Download puzzle data (one-time setup)
bash setup.sh
```

## CI

GitHub Actions runs on PRs to `main` (`.github/workflows/ci.yml`):
1. `ruff check src tests` — lint
2. `ty check src tests` — type check
3. `pytest tests/ -v` — smoke tests

All three must pass before merge.

## Code style

- **Formatter/linter:** ruff, line length 100, rules: E, F, I (errors, pyflakes, isort)
- **Type checker:** ty (python-version 3.13)
- snake_case for functions/variables, PascalCase for classes
- Type hints on function signatures
- CLI args use `argparse` with `parse_args()` pattern

## Testing

Smoke tests in `tests/` run on CPU with synthetic data — no puzzle CSV required. Tests cover tensor shapes, piece encoding, model forward pass, and parameter count scaling.

## Architecture notes

- Input: FEN string -> 64 piece tokens (0=empty, 1-6=friendly, 7-12=opponent) + side-to-move
- Embeddings: piece embedding + learned positional embedding over 8x8 board
- CLS token pooled through LayerNorm -> Linear -> GELU -> Linear -> scalar rating
- Training: MSELoss, AdamW, LR warmup + cosine annealing, gradient clipping
- Data: Lichess puzzles filtered to RatingDeviation < 75, ratings normalized by mean/std
