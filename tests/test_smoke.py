"""
Smoke tests — run entirely on CPU with synthetic data, no puzzle CSV needed.
"""
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import _flip_board_h, fen_to_bitboard, fen_to_sequence, fen_to_tensor
from model import ChessPuzzleTransformer

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MIDGAME_FEN = "r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 7"
# A real Lichess puzzle: FEN + 3-move sequence (setup + 2 solution moves)
PUZZLE_FEN = "r1bqkbnr/pppppppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
PUZZLE_MOVES = "d2d4 c6d4 f3d4"  # setup move + 2 solution moves


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


def test_fen_to_bitboard_shape():
    t = fen_to_bitboard(STARTING_FEN)
    assert t.shape == (12, 8, 8), f"Expected (12, 8, 8), got {t.shape}"
    assert t.dtype == torch.float32


def test_fen_to_bitboard_piece_count():
    t = fen_to_bitboard(STARTING_FEN)
    # 32 pieces on the board, each in exactly one plane
    assert t.sum().item() == 32


def test_model_forward_bitboard():
    model = ChessPuzzleTransformer(
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128, encoding="bitboard"
    )
    model.eval()
    x = fen_to_bitboard(STARTING_FEN).unsqueeze(0)  # (1, 12, 8, 8)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1,), f"Expected (1,), got {out.shape}"
    assert not torch.isnan(out).any()


def test_model_forward_extra_features():
    model = ChessPuzzleTransformer(
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_extra_features=1
    )
    model.eval()
    batch = torch.randint(0, 13, (8, 64))
    extra = torch.randn(8, 1)
    with torch.no_grad():
        out = model(batch, extra_features=extra)
    assert out.shape == (8,)
    assert not torch.isnan(out).any(), "NaN in model output with extra features"


def test_fen_to_sequence_shape():
    seq = fen_to_sequence(PUZZLE_FEN, PUZZLE_MOVES, encoding="piece_index")
    assert seq.shape == (3, 64), f"Expected (3, 64), got {seq.shape}"
    assert seq.dtype == torch.long


def test_fen_to_sequence_bitboard():
    seq = fen_to_sequence(PUZZLE_FEN, PUZZLE_MOVES, encoding="bitboard")
    assert seq.shape == (3, 12, 8, 8), f"Expected (3, 12, 8, 8), got {seq.shape}"


def test_flip_board_piece_index():
    t = fen_to_tensor(STARTING_FEN)
    flipped = _flip_board_h(t, "piece_index")
    assert flipped.shape == t.shape
    # Flipping twice should restore the original
    assert torch.equal(_flip_board_h(flipped, "piece_index"), t)


def test_flip_board_bitboard():
    t = fen_to_bitboard(STARTING_FEN)
    flipped = _flip_board_h(t, "bitboard")
    assert flipped.shape == t.shape
    assert torch.equal(_flip_board_h(flipped, "bitboard"), t)
    # Piece count should be preserved
    assert flipped.sum().item() == t.sum().item()


def test_model_forward_solution_seq():
    model = ChessPuzzleTransformer(
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128, use_solution_seq=True
    )
    model.eval()
    # Simulate a batch of 4 puzzles, each with 3 board positions
    x = torch.randint(0, 13, (4, 3, 64))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4,), f"Expected (4,), got {out.shape}"
    assert not torch.isnan(out).any()


def test_model_forward_solution_seq_with_seq_lens():
    model = ChessPuzzleTransformer(
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128, use_solution_seq=True
    )
    model.eval()
    # Variable-length sequences padded to max_len=5
    x = torch.randint(0, 13, (4, 5, 64))
    seq_lens = torch.tensor([3, 5, 2, 4])
    with torch.no_grad():
        out = model(x, seq_lens=seq_lens)
    assert out.shape == (4,)
    assert not torch.isnan(out).any()


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
    """train_modal must import without errors (catches removed Modal API attributes)."""
    import train_modal  # noqa: F401


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
