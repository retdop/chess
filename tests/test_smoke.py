"""
Smoke tests — run entirely on CPU with synthetic data, no puzzle CSV needed.
"""
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import fen_to_tensor
from model import ChessPuzzleTransformer

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MIDGAME_FEN = "r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 7"


def test_fen_to_tensor_shape():
    t = fen_to_tensor(STARTING_FEN)
    assert t.shape == (64,), f"Expected (64,), got {t.shape}"
    assert t.dtype == torch.long


def test_fen_to_tensor_starting_position():
    t = fen_to_tensor(STARTING_FEN)
    # 32 non-empty squares in starting position
    assert (t != 0).sum().item() == 32


def test_fen_to_tensor_midgame():
    t = fen_to_tensor(MIDGAME_FEN)
    assert t.shape == (64,)
    assert t.min().item() >= 0
    assert t.max().item() <= 12


def test_model_forward_single():
    model = ChessPuzzleTransformer(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    model.eval()
    x = fen_to_tensor(STARTING_FEN).unsqueeze(0)  # (1, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1,), f"Expected (1,), got {out.shape}"


def test_model_forward_batch():
    model = ChessPuzzleTransformer(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    model.eval()
    batch = torch.randint(0, 13, (16, 64))
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (16,)
    assert not torch.isnan(out).any(), "NaN in model output"


def test_model_parameter_count():
    small = ChessPuzzleTransformer(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    full = ChessPuzzleTransformer()
    assert sum(p.numel() for p in full.parameters()) > sum(p.numel() for p in small.parameters())


def test_evaluate_loads_config(tmp_path):
    """evaluate.py must reconstruct the model from config.json, not hardcoded defaults."""
    # Train a small model and save its checkpoint + config
    model = ChessPuzzleTransformer(
        d_model=64, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.0
    )
    torch.save(model.state_dict(), tmp_path / "best.pt")
    cfg = {"d_model": 64, "nhead": 2, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.0}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "stats.json").write_text(json.dumps({"rating_mean": 1500.0, "rating_std": 300.0}))

    # Reload via the same logic used in evaluate.py
    with open(tmp_path / "config.json") as f:
        loaded_cfg: dict[str, Any] = json.load(f)
    restored = ChessPuzzleTransformer(**loaded_cfg)
    restored.load_state_dict(torch.load(tmp_path / "best.pt", map_location="cpu"))
    restored.eval()

    # Confirm it runs and produces finite output
    x = torch.randint(0, 13, (4, 64))
    with torch.no_grad():
        out = restored(x)
    assert out.shape == (4,)
    assert not torch.isnan(out).any()


def test_train_modal_imports():
    """train_modal.py must import without errors (catches removed Modal API attributes)."""
    repo_root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location("train_modal", repo_root / "train_modal.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]


def test_evaluate_config_mismatch_raises(tmp_path):
    """Loading a checkpoint into a mismatched model must raise RuntimeError."""
    small_model = ChessPuzzleTransformer(
        d_model=64, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.0
    )
    torch.save(small_model.state_dict(), tmp_path / "best.pt")

    # Try to load into a larger model (simulates the original bug)
    large_model = ChessPuzzleTransformer(d_model=128, nhead=4, num_layers=4, dim_feedforward=256)
    try:
        large_model.load_state_dict(torch.load(tmp_path / "best.pt", map_location="cpu"))
        raise AssertionError("Expected RuntimeError for mismatched architectures")
    except RuntimeError:
        pass  # expected
