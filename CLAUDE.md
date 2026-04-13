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
src/model.py        - ChessPuzzleTransformer architecture (pre-norm transformer)
src/dataset.py      - PuzzleDataset, FEN-to-tensor encoding
src/train.py        - Local training loop with CLI args
src/evaluate.py     - Evaluation metrics and scatter plots
src/train_modal.py  - Modal wrapper for cloud GPU training
tests/test_smoke.py - CPU-only smoke tests (no data files needed)
experiments/        - One subfolder per experiment (config + results)
IDEAS.md            - Backlog of feature and architecture ideas
setup.sh            - Downloads Lichess puzzle CSV (~800MB)
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

## Running experiments

Experiments are the primary way to train models and collect results. Each experiment lives in its own folder under `experiments/` with a config file and (after training) result artifacts.

### Experiment layout

```
experiments/
  baseline/
    config.json       - Hyperparameter config (required)
    PENDING           - Marker file that triggers training (create to run)
    results.json      - Test metrics: RMSE, MAE, Pearson r (auto-generated)
    stats.json        - Rating normalization stats (auto-generated)
    history.json      - Per-step and per-epoch training metrics (auto-generated)
    predictions.csv   - Per-puzzle true vs predicted ratings (auto-generated)
```

Scatter plots (`scatter.png`) are uploaded as GitHub Actions artifacts (not committed).

### How to start an experiment

1. Create the experiment folder and config:
   ```bash
   mkdir -p experiments/<experiment-name>
   ```
2. Write `experiments/<experiment-name>/config.json` with hyperparameters. Any key not specified falls back to defaults in `src/train.py`:
   ```json
   {
     "epochs": 20,
     "batch_size": 512,
     "d_model": 256,
     "nhead": 8,
     "num_layers": 6,
     "dim_feedforward": 1024,
     "dropout": 0.1,
     "lr": 1e-3,
     "warmup_frac": 0.1,
     "val_frac": 0.05,
     "seed": 42,
     "max_samples": null
   }
   ```
3. Create the PENDING marker file:
   ```bash
   touch experiments/<experiment-name>/PENDING
   ```
4. Commit and push. The `dispatch.yml` workflow detects PENDING files and triggers `train.yml` for each experiment automatically.

Multiple experiments can be launched in parallel by creating several `PENDING` files in one push.

### How to read results

After training completes, GitHub Actions commits results back to the branch and removes the PENDING file.

- **`results.json`** — key metrics to check first:
  - `test_rmse_elo`: Root mean square error in Elo (lower is better)
  - `test_mae_elo`: Mean absolute error in Elo (lower is better)
  - `pearson_r`: Correlation between predicted and true ratings (higher is better; ~0 means model predicts a constant)
- **`history.json`** — training dynamics:
  - `steps[].train_loss`: per-batch loss (should decrease over training)
  - `epochs[].val_rmse_elo`: validation RMSE after each epoch (should decrease)
  - `epochs[].lr`: learning rate at each epoch (verify schedule looks right)
- **`predictions.csv`** — raw predictions for offline analysis or custom plots
- **Scatter plot artifact** — download from the GitHub Actions run; a diagonal trend means the model discriminates difficulty, a horizontal band means it predicts a constant

### Config reference

| Key | Default | Description |
|-----|---------|-------------|
| `epochs` | 20 | Number of training epochs |
| `batch_size` | 512 | Training batch size |
| `d_model` | 256 | Transformer hidden dimension |
| `nhead` | 8 | Number of attention heads |
| `num_layers` | 6 | Transformer encoder depth |
| `dim_feedforward` | 1024 | FFN intermediate dimension |
| `dropout` | 0.1 | Dropout rate |
| `lr` | 1e-3 | Peak learning rate (AdamW) |
| `warmup_frac` | 0.1 | Fraction of total steps for LR warmup |
| `val_frac` | 0.05 | Fraction of data for validation |
| `seed` | 42 | Random seed for reproducibility |
| `max_samples` | null | Limit dataset size (null = use all data, set to small number for overfit tests) |
| `pool` | `"cls"` | Pooling strategy: `"cls"` (learnable CLS token) or `"mean"` (average all 64 squares) |
| `pos_enc` | `"flat"` | Positional encoding: `"flat"` (one embedding per square) or `"2d"` (separate rank + file embeddings) |
| `encoding` | `"piece_index"` | Input encoding: `"piece_index"` (single int 0-12 per square) or `"bitboard"` (12 binary 8x8 planes) |

### Diagnosing training issues

- **Horizontal scatter (pearson_r ~ 0):** Model is predicting a near-constant value. Check if train_loss is decreasing at all in `history.json`. Try smaller model, higher LR, or an overfit test on a tiny dataset.
- **Train loss flat from the start:** Learning rate may be too low, or gradients may be vanishing. Check LR schedule in history.
- **Train loss drops but val doesn't improve:** Overfitting. Try more dropout, fewer layers, or weight decay.
- **Val RMSE plateaus early:** Model capacity may be saturated for the features available. Consider richer input encoding (castling rights, en passant, etc.)
