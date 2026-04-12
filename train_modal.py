"""
Run training on a Modal GPU.

Local usage:
    modal run train_modal.py
    modal run train_modal.py --epochs 10 --d-model 128

From CI (MODAL_TOKEN_ID / MODAL_TOKEN_SECRET set as secrets):
    modal run train_modal.py --epochs 20
"""
import sys
from pathlib import Path

import modal

app = modal.App("chess-puzzle-rating")

# Persistent volumes — data survives between runs so the puzzle CSV is only downloaded once.
data_vol = modal.Volume.from_name("chess-data", create_if_missing=True)
ckpt_vol  = modal.Volume.from_name("chess-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("zstd", "wget")
    .pip_install(
        "torch>=2.1.0",
        "python-chess>=1.999",
        "pandas>=2.0",
        "numpy>=1.24",
        "matplotlib>=3.7",
    )
    # Mount source code at runtime (copy=False) so the heavy image layer is
    # cached and reused even when source files change.
    .add_local_dir(Path(__file__).parent / "src", remote_path="/app/src")
)


@app.function(
    gpu="T4",
    timeout=14400,  # 4 hours
    image=image,
    volumes={"/data": data_vol, "/checkpoints": ckpt_vol},
)
def train_and_eval(
    epochs: int = 20,
    batch_size: int = 512,
    d_model: int = 256,
    num_layers: int = 6,
    nhead: int = 8,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    lr: float = 1e-3,
    warmup_frac: float = 0.1,
    val_frac: float = 0.05,
    seed: int = 42,
):
    import subprocess

    data_path = "/data/lichess_puzzles.csv"

    # Download puzzle database once; the volume caches it across runs.
    if not Path(data_path).exists():
        print("Downloading Lichess puzzle database (~250 MB compressed)...")
        subprocess.run(
            ["wget", "-q", "-O", "/data/puzzles.csv.zst",
             "https://database.lichess.org/lichess_db_puzzle.csv.zst"],
            check=True,
        )
        subprocess.run(
            ["zstd", "-d", "/data/puzzles.csv.zst", "-o", data_path],
            check=True,
        )
        Path("/data/puzzles.csv.zst").unlink()
        data_vol.commit()
        print("Data ready.")

    def run(script: str, *extra_args: str):
        subprocess.run(
            [sys.executable, script, *extra_args],
            cwd="/app/src",
            check=True,
        )

    run(
        "train.py",
        "--data_path",      data_path,
        "--checkpoint_dir", "/checkpoints",
        "--epochs",         str(epochs),
        "--batch_size",     str(batch_size),
        "--d_model",        str(d_model),
        "--num_layers",     str(num_layers),
        "--nhead",          str(nhead),
        "--dim_feedforward", str(dim_feedforward),
        "--dropout",        str(dropout),
        "--lr",             str(lr),
        "--warmup_frac",    str(warmup_frac),
        "--val_frac",       str(val_frac),
        "--seed",           str(seed),
    )

    run(
        "evaluate.py",
        "--data_path",      data_path,
        "--checkpoint_dir", "/checkpoints",
    )

    ckpt_vol.commit()
    print("Checkpoints committed to Modal volume 'chess-checkpoints'.")


@app.local_entrypoint()
def main(
    epochs: int = 20,
    batch_size: int = 512,
    d_model: int = 256,
    num_layers: int = 6,
    nhead: int = 8,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    lr: float = 1e-3,
    warmup_frac: float = 0.1,
    val_frac: float = 0.05,
    seed: int = 42,
):
    train_and_eval.remote(
        epochs=epochs,
        batch_size=batch_size,
        d_model=d_model,
        num_layers=num_layers,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        lr=lr,
        warmup_frac=warmup_frac,
        val_frac=val_frac,
        seed=seed,
    )
