import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import PuzzleDataset, load_puzzles
from model import ChessPuzzleTransformer


def parse_args():
    p = argparse.ArgumentParser(description="Train chess puzzle rating predictor")
    p.add_argument("--data_path", default="data/lichess_puzzles.csv")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--dim_feedforward", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    df = load_puzzles(args.data_path)

    rating_mean = float(df["Rating"].mean())
    rating_std  = float(df["Rating"].std())
    print(f"Rating  mean={rating_mean:.0f}  std={rating_std:.0f}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    with open(ckpt_dir / "stats.json", "w") as f:
        json.dump({"rating_mean": rating_mean, "rating_std": rating_std}, f)

    dataset = PuzzleDataset(df, rating_mean, rating_std)
    n_val   = int(len(dataset) * args.val_frac)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Split  train={n_train:,}  val={n_val:,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2,
        num_workers=4, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ChessPuzzleTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_rmse = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = []

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss.append(loss.item())

            if (step + 1) % 200 == 0:
                print(f"  epoch {epoch}  step {step+1}/{len(train_loader)}"
                      f"  loss={np.mean(running_loss[-200:]):.4f}")

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_losses.append(criterion(model(x), y).item())

        val_rmse_norm = float(np.mean(val_losses)) ** 0.5
        val_rmse_elo  = val_rmse_norm * rating_std
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}/{args.epochs} | val RMSE={val_rmse_elo:.1f} Elo | lr={lr_now:.2e}")

        scheduler.step()

        if val_rmse_elo < best_val_rmse:
            best_val_rmse = val_rmse_elo
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  -> saved best (RMSE={best_val_rmse:.1f})")

    print(f"\nDone. Best val RMSE: {best_val_rmse:.1f} Elo")


if __name__ == "__main__":
    main()
