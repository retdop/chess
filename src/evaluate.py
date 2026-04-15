"""
Evaluate a trained model on a held-out test split and save a scatter plot.

Usage:
    python src/evaluate.py --data_path data/lichess_puzzles.csv
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset import PuzzleDataset, load_puzzles, puzzle_collate_fn
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


def evaluate_on(
    model: torch.nn.Module,
    test_loader: DataLoader,
    rating_mean: float,
    rating_std: float,
    use_move_count: bool,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run model on a test loader and return (preds_elo, targets_elo, metrics)."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            board = batch["board"].to(device)
            extra = None
            if use_move_count:
                extra = batch["num_moves"].to(device).unsqueeze(-1)
            seq_lens = batch.get("seq_lens")
            preds.extend(model(board, extra_features=extra, seq_lens=seq_lens).cpu().numpy())
            targets.extend(batch["rating"].numpy())

    preds_elo = np.array(preds) * rating_std + rating_mean
    targets_elo = np.array(targets) * rating_std + rating_mean

    rmse = float(np.sqrt(np.mean((preds_elo - targets_elo) ** 2)))
    mae = float(np.mean(np.abs(preds_elo - targets_elo)))
    corr = float(np.corrcoef(preds_elo, targets_elo)[0, 1])

    metrics = {
        "test_rmse_elo": round(rmse, 1),
        "test_mae_elo": round(mae, 1),
        "pearson_r": round(corr, 4),
    }
    return preds_elo, targets_elo, metrics


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(args.checkpoint_dir)
    with open(ckpt_dir / "stats.json") as f:
        stats = json.load(f)
    rating_mean = stats["rating_mean"]
    rating_std  = stats["rating_std"]

    config_path = ckpt_dir / "config.json"
    model_kwargs: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            model_kwargs = json.load(f)
    use_move_count = model_kwargs.pop("use_move_count", False)
    use_solution_seq = model_kwargs.get("use_solution_seq", False)
    max_rd = model_kwargs.pop("max_rating_deviation", 75.0)

    encoding = model_kwargs.get("encoding", "piece_index")
    collate_fn = puzzle_collate_fn if use_solution_seq else None

    model = ChessPuzzleTransformer(**model_kwargs).to(device)
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    model.eval()

    # ── Primary evaluation (same data distribution as training) ───────────
    df = load_puzzles(args.data_path, max_rating_deviation=max_rd)
    dataset = PuzzleDataset(
        df, rating_mean, rating_std, encoding=encoding,
        use_solution_seq=use_solution_seq,
    )

    n_test  = int(len(dataset) * args.test_frac)
    n_val   = int(len(dataset) * args.val_frac)
    n_train = len(dataset) - n_val - n_test
    _, _, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn,
    )

    preds, targets, results = evaluate_on(
        model, test_loader, rating_mean, rating_std, use_move_count, device,
    )

    print(f"Test RMSE : {results['test_rmse_elo']:.1f} Elo")
    print(f"Test MAE  : {results['test_mae_elo']:.1f} Elo")
    print(f"Pearson r : {results['pearson_r']:.4f}")

    with open(ckpt_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {ckpt_dir / 'results.json'}")

    # Per-puzzle predictions (allows recreating plots offline)
    pred_path = ckpt_dir / "predictions.csv"
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_rating", "predicted_rating"])
        for t, p in zip(targets, preds):
            writer.writerow([round(float(t), 1), round(float(p), 1)])
    print(f"Per-puzzle predictions saved to {pred_path}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets, preds, alpha=0.05, s=1, color="steelblue")
    lo, hi = targets.min(), targets.max()
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("True Rating")
    ax.set_ylabel("Predicted Rating")
    ax.set_title(f"Puzzle Rating Prediction\nRMSE={results['test_rmse_elo']:.0f} Elo"
                 f"   r={results['pearson_r']:.3f}")
    out = ckpt_dir / "scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Scatter plot saved to {out}")

    # ── Standardized evaluation on RD < 75 (comparable across experiments) ─
    STANDARD_RD = 75.0
    if max_rd != STANDARD_RD:
        print(f"\n{'='*60}")
        print(f"Standardized evaluation (RD < {STANDARD_RD})")
        print(f"{'='*60}")
        df_std = load_puzzles(args.data_path, max_rating_deviation=STANDARD_RD)
        dataset_std = PuzzleDataset(
            df_std, rating_mean, rating_std, encoding=encoding,
            use_solution_seq=use_solution_seq,
        )
        n_test_std = int(len(dataset_std) * args.test_frac)
        n_val_std = int(len(dataset_std) * args.val_frac)
        n_train_std = len(dataset_std) - n_val_std - n_test_std
        _, _, test_ds_std = random_split(
            dataset_std, [n_train_std, n_val_std, n_test_std],
            generator=torch.Generator().manual_seed(args.seed),
        )
        test_loader_std = DataLoader(
            test_ds_std, batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn,
        )
        _, _, results_std = evaluate_on(
            model, test_loader_std, rating_mean, rating_std, use_move_count, device,
        )
        print(f"Test RMSE : {results_std['test_rmse_elo']:.1f} Elo")
        print(f"Test MAE  : {results_std['test_mae_elo']:.1f} Elo")
        print(f"Pearson r : {results_std['pearson_r']:.4f}")

        with open(ckpt_dir / "results_std75.json", "w") as f:
            json.dump(results_std, f, indent=2)
        print(f"Standardized results saved to {ckpt_dir / 'results_std75.json'}")
    else:
        # Training already used RD < 75 — primary results are the standard
        print("\nTraining used RD < 75; primary results are the standardized benchmark.")


if __name__ == "__main__":
    main()
