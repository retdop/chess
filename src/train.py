import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import PuzzleDataset, load_puzzles
from model import ChessPuzzleTransformer

DEFAULTS: dict = {
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
}


def parse_args():
    p = argparse.ArgumentParser(description="Train chess puzzle rating predictor")
    p.add_argument("--config", required=True, help="Path to experiment config JSON")
    p.add_argument("--data_path", default="data/lichess_puzzles.csv")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = {**DEFAULTS, **json.load(f)}

    torch.manual_seed(cfg["seed"])
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
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump({
            "d_model": cfg["d_model"],
            "nhead": cfg["nhead"],
            "num_layers": cfg["num_layers"],
            "dim_feedforward": cfg["dim_feedforward"],
            "dropout": cfg["dropout"],
        }, f)

    dataset = PuzzleDataset(df, rating_mean, rating_std)
    n_val   = int(len(dataset) * cfg["val_frac"])
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    print(f"Split  train={n_train:,}  val={n_val:,}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2,
        num_workers=4, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ChessPuzzleTransformer(
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)

    total_steps  = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_frac"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_rmse = float("inf")
    history: dict[str, list[dict]] = {"steps": [], "epochs": []}

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss = []

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss.append(loss.item())

            if (step + 1) % 200 == 0:
                avg_loss = float(np.mean(running_loss[-200:]))
                print(f"  epoch {epoch}  step {step+1}/{len(train_loader)}"
                      f"  loss={avg_loss:.4f}")
                history["steps"].append({
                    "epoch": epoch,
                    "step": step + 1,
                    "train_loss": round(avg_loss, 6),
                })

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
        print(f"Epoch {epoch}/{cfg['epochs']} | val RMSE={val_rmse_elo:.1f} Elo | lr={lr_now:.2e}")

        history["epochs"].append({
            "epoch": epoch,
            "train_loss": round(float(np.mean(running_loss)), 6),
            "val_rmse_elo": round(val_rmse_elo, 1),
            "lr": lr_now,
        })

        if val_rmse_elo < best_val_rmse:
            best_val_rmse = val_rmse_elo
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  -> saved best (RMSE={best_val_rmse:.1f})")

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {ckpt_dir / 'history.json'}")
    print(f"\nDone. Best val RMSE: {best_val_rmse:.1f} Elo")


if __name__ == "__main__":
    main()
