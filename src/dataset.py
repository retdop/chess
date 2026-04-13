import chess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 0 = empty, 1-6 = side-to-move pieces, 7-12 = opponent pieces
# Mapping is resolved at runtime based on board.turn.
_PIECE_IDX_FRIENDLY = {
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
    chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6,
}
_PIECE_IDX_OPPONENT = {
    chess.PAWN: 7, chess.KNIGHT: 8, chess.BISHOP: 9,
    chess.ROOK: 10, chess.QUEEN: 11, chess.KING: 12,
}

# Bitboard plane indices: 0-5 = friendly PNBRQK, 6-11 = opponent PNBRQK
_PLANE_FRIENDLY = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}
_PLANE_OPPONENT = {
    chess.PAWN: 6, chess.KNIGHT: 7, chess.BISHOP: 8,
    chess.ROOK: 9, chess.QUEEN: 10, chess.KING: 11,
}


def fen_to_tensor(fen: str, first_move: str | None = None) -> torch.Tensor:
    """Convert FEN string to (64,) tensor of piece indices (0=empty, 1-12=pieces).

    Pieces are encoded relative to the side to move: 1-6 = friendly pieces,
    7-12 = opponent pieces.  This makes the representation colour-invariant so
    the model doesn't have to learn symmetric patterns for white and black.

    If *first_move* is given (UCI string), it is applied to the board first.
    In the Lichess puzzle CSV the FEN is the position **before** the opponent's
    setup move, so applying the first move yields the actual puzzle position.
    """
    board = chess.Board(fen)
    if first_move is not None:
        board.push_uci(first_move)
    stm = board.turn  # side to move
    squares = []
    for sq in chess.SQUARES:  # a1..h8
        piece = board.piece_at(sq)
        if piece is None:
            squares.append(0)
        elif piece.color == stm:
            squares.append(_PIECE_IDX_FRIENDLY[piece.piece_type])
        else:
            squares.append(_PIECE_IDX_OPPONENT[piece.piece_type])
    return torch.tensor(squares, dtype=torch.long)


def fen_to_bitboard(fen: str, first_move: str | None = None) -> torch.Tensor:
    """Convert FEN string to (12, 8, 8) binary tensor of piece planes.

    12 planes: 6 friendly piece types (PNBRQK) + 6 opponent piece types.
    Pieces are relative to side-to-move (colour-invariant), same as
    ``fen_to_tensor``.  Board layout: planes[channel][rank][file] where
    rank 0 = rank 1 (a1-h1) and file 0 = file a.
    """
    board = chess.Board(fen)
    if first_move is not None:
        board.push_uci(first_move)
    stm = board.turn
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        if piece.color == stm:
            planes[_PLANE_FRIENDLY[piece.piece_type], rank, file] = 1.0
        else:
            planes[_PLANE_OPPONENT[piece.piece_type], rank, file] = 1.0
    return torch.from_numpy(planes)


class PuzzleDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        df: pd.DataFrame,
        rating_mean: float,
        rating_std: float,
        encoding: str = "piece_index",
    ):
        if encoding not in ("piece_index", "bitboard"):
            raise ValueError(f"encoding must be 'piece_index' or 'bitboard', got {encoding!r}")
        self.encoding = encoding
        self.fens = df["FEN"].values
        self.first_moves = (
            df["Moves"].str.split().str[0].values
            if "Moves" in df.columns
            else np.array([None] * len(df))
        )
        self.ratings = ((df["Rating"].values - rating_mean) / rating_std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # ty: ignore[invalid-method-override]
        if self.encoding == "bitboard":
            board = fen_to_bitboard(self.fens[idx], self.first_moves[idx])
        else:
            board = fen_to_tensor(self.fens[idx], self.first_moves[idx])
        rating = torch.tensor(self.ratings[idx], dtype=torch.float32)
        return board, rating


def load_puzzles(csv_path: str, max_rating_deviation: float = 75.0) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["RatingDeviation"] < max_rating_deviation].reset_index(drop=True)
    print(f"Loaded {len(df):,} puzzles (RatingDeviation < {max_rating_deviation})")
    return df
