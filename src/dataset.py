import chess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 0 = empty, 1-6 = white pieces, 7-12 = black pieces
PIECE_TO_IDX = {
    (chess.PAWN,   chess.WHITE): 1,
    (chess.KNIGHT, chess.WHITE): 2,
    (chess.BISHOP, chess.WHITE): 3,
    (chess.ROOK,   chess.WHITE): 4,
    (chess.QUEEN,  chess.WHITE): 5,
    (chess.KING,   chess.WHITE): 6,
    (chess.PAWN,   chess.BLACK): 7,
    (chess.KNIGHT, chess.BLACK): 8,
    (chess.BISHOP, chess.BLACK): 9,
    (chess.ROOK,   chess.BLACK): 10,
    (chess.QUEEN,  chess.BLACK): 11,
    (chess.KING,   chess.BLACK): 12,
}


def fen_to_tensor(fen: str) -> torch.Tensor:
    """Convert FEN string to (64,) tensor of piece indices (0=empty, 1-12=pieces)."""
    board = chess.Board(fen)
    squares = []
    for sq in chess.SQUARES:  # a1..h8
        piece = board.piece_at(sq)
        if piece is None:
            squares.append(0)
        else:
            squares.append(PIECE_TO_IDX[(piece.piece_type, piece.color)])
    return torch.tensor(squares, dtype=torch.long)


class PuzzleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, rating_mean: float, rating_std: float):
        self.fens = df["FEN"].values
        self.ratings = ((df["Rating"].values - rating_mean) / rating_std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, idx: int):
        board = fen_to_tensor(self.fens[idx])
        rating = torch.tensor(self.ratings[idx], dtype=torch.float32)
        return board, rating


def load_puzzles(csv_path: str, max_rating_deviation: float = 75.0) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["RatingDeviation"] < max_rating_deviation].reset_index(drop=True)
    print(f"Loaded {len(df):,} puzzles (RatingDeviation < {max_rating_deviation})")
    return df
