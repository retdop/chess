"""
Run training on a Modal GPU.

Local usage:
    modal run src/train_modal.py --config experiments/baseline/config.json

From CI (MODAL_TOKEN_ID / MODAL_TOKEN_SECRET set as secrets):
    modal run src/train_modal.py --config experiments/baseline/config.json
"""
import json
import sys
from pathlib import Path

import modal

app = modal.App("chess-puzzle-rating")

# Persistent volumes — data survives between runs so the puzzle CSV is only downloaded once.
data_vol = modal.Volume.from_name("chess-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("chess-checkpoints", create_if_missing=True)

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
    .add_local_dir(Path(__file__).parent, remote_path="/app/src")
)


@app.function(
    gpu="T4",
    timeout=14400,  # 4 hours
    image=image,
    volumes={"/data": data_vol, "/checkpoints": ckpt_vol},
)
def train_and_eval(config: dict):
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

    # Write config to a temp file for train.py to read
    config_path = "/tmp/config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    def run(script: str, *extra_args: str):
        subprocess.run(
            [sys.executable, script, *extra_args],
            cwd="/app/src",
            check=True,
        )

    run(
        "train.py",
        "--config",         config_path,
        "--data_path",      data_path,
        "--checkpoint_dir", "/checkpoints",
    )

    run(
        "evaluate.py",
        "--data_path",      data_path,
        "--checkpoint_dir", "/checkpoints",
        "--val_frac",       str(config.get("val_frac", 0.05)),
        "--seed",           str(config.get("seed", 42)),
    )

    ckpt_vol.commit()
    print("Checkpoints committed to Modal volume 'chess-checkpoints'.")


@app.local_entrypoint()
def main(config: str = "experiments/baseline/config.json"):
    with open(config) as f:
        cfg = json.load(f)
    train_and_eval.remote(config=cfg)
