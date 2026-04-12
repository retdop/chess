"""
Smoke tests — run entirely on CPU with synthetic data, no puzzle CSV needed.
"""
import sys
from pathlib import Path

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
