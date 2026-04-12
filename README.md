# Chess Puzzle Rating Estimator

Predict Lichess puzzle difficulty ratings from board position using a Vision Transformer-style architecture.

## Setup (on GPU instance)

```bash
# Install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
bash setup.sh   # downloads Lichess puzzle CSV (~800 MB)
uv sync         # installs all dependencies into .venv
```

## Train

```bash
uv run python src/train.py
```

Key flags:
- `--epochs 20` — number of training epochs
- `--batch_size 512` — batch size
- `--d_model 256` — transformer hidden dim
- `--num_layers 6` — transformer depth

Alternatively, trigger training via the **Train** GitHub Actions workflow (requires a self-hosted GPU runner).

## Evaluate

```bash
uv run python src/evaluate.py
```

Prints test RMSE, MAE, and Pearson r; saves `checkpoints/scatter.png`.

## Architecture

- Input: FEN string → 64 square tokens (piece type index per square)
- Model: piece embedding + positional embedding → Transformer encoder (pre-norm) → CLS token → regression head
- Output: normalized puzzle rating (denormalized for metrics)
- Loss: MSE on normalized ratings

## Data

[Lichess puzzle database](https://database.lichess.org/#puzzles) (~4M puzzles).
Filtered to `RatingDeviation < 75` for well-calibrated labels.
