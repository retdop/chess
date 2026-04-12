"""
Evaluate a trained model on a held-out test split and save a scatter plot.

Usage:
    python src/evaluate.py --data_path data/lichess_puzzles.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset import PuzzleDataset, load_puzzles
from model import ChessPuzzleTransformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/lichess_puzzles.csv")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--test_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(args.checkpoint_dir)
    with open(ckpt_dir / "stats.json") as f:
        stats = json.load(f)
    rating_mean = stats["rating_mean"]
    rating_std  = stats["rating_std"]

    df = load_puzzles(args.data_path)
    dataset = PuzzleDataset(df, rating_mean, rating_std)

    n_test  = int(len(dataset) * args.test_frac)
    n_val   = int(len(dataset) * args.val_frac)
    n_train = len(dataset) - n_val - n_test
    _, _, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=4)

    config_path = ckpt_dir / "config.json"
    model_kwargs = {}
    if config_path.exists():
        with open(config_path) as f:
            model_kwargs = json.load(f)
    model = ChessPuzzleTransformer(**model_kwargs).to(device)
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.extend(model(x.to(device)).cpu().numpy())
            targets.extend(y.numpy())

    preds   = np.array(preds)   * rating_std + rating_mean
    targets = np.array(targets) * rating_std + rating_mean

    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae  = float(np.mean(np.abs(preds - targets)))
    corr = float(np.corrcoef(preds, targets)[0, 1])

    print(f"Test RMSE : {rmse:.1f} Elo")
    print(f"Test MAE  : {mae:.1f} Elo")
    print(f"Pearson r : {corr:.4f}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets, preds, alpha=0.05, s=1, color="steelblue")
    lo, hi = targets.min(), targets.max()
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("True Rating")
    ax.set_ylabel("Predicted Rating")
    ax.set_title(f"Puzzle Rating Prediction\nRMSE={rmse:.0f} Elo   r={corr:.3f}")
    out = ckpt_dir / "scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Scatter plot saved to {out}")


if __name__ == "__main__":
    main()
