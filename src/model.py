import torch
import torch.nn as nn


class ChessPuzzleTransformer(nn.Module):
    """
    Treats a chess position as a sequence of 64 square tokens.

    Input encoding is configurable via *encoding*:
      - ``"piece_index"`` (default): input is (B, 64) integer tensor with
        piece indices 0-12.  Each index is looked up in an embedding table.
      - ``"bitboard"``: input is (B, 12, 8, 8) float tensor with 12 binary
        piece-type planes.  Reshaped to (B, 64, 12) and linearly projected
        into d_model.

    Positional encoding is configurable via *pos_enc*:
      - ``"flat"`` (default): one learned embedding per square (64 total).
      - ``"2d"``: separate rank (row) and file (column) embeddings (8+8=16),
        summed to give each square a 2D-aware position signal.

    Pooling strategy is configurable via *pool*:
      - ``"cls"`` (default): a learnable [CLS] token is prepended and its
        final hidden state drives the rating head.
      - ``"mean"``: the 64 square outputs are averaged to form the input
        to the rating head (no CLS token).
    """

    def __init__(
        self,
        num_piece_types: int = 13,   # 0=empty, 1-12=pieces
        num_planes: int = 12,        # bitboard planes (6 friendly + 6 opponent)
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pool: str = "cls",
        pos_enc: str = "flat",
        encoding: str = "piece_index",
        num_extra_features: int = 0,
    ):
        super().__init__()
        if pool not in ("cls", "mean"):
            raise ValueError(f"pool must be 'cls' or 'mean', got {pool!r}")
        if pos_enc not in ("flat", "2d"):
            raise ValueError(f"pos_enc must be 'flat' or '2d', got {pos_enc!r}")
        if encoding not in ("piece_index", "bitboard"):
            raise ValueError(f"encoding must be 'piece_index' or 'bitboard', got {encoding!r}")
        self.pool = pool
        self.pos_enc = pos_enc
        self.encoding = encoding
        self.num_extra_features = num_extra_features

        # Input projection: embedding lookup for piece_index, linear for bitboard
        if encoding == "piece_index":
            self.piece_embedding = nn.Embedding(num_piece_types, d_model)
        else:
            self.piece_projection = nn.Linear(num_planes, d_model)

        if pos_enc == "flat":
            self.pos_embedding = nn.Embedding(64, d_model)
        else:
            self.rank_embedding = nn.Embedding(8, d_model)
            self.file_embedding = nn.Embedding(8, d_model)
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        head_input_dim = d_model + num_extra_features
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        if self.encoding == "piece_index":
            nn.init.trunc_normal_(self.piece_embedding.weight, std=0.1)
        else:
            nn.init.trunc_normal_(self.piece_projection.weight, std=0.1)
            nn.init.zeros_(self.piece_projection.bias)
        if self.pos_enc == "flat":
            nn.init.trunc_normal_(self.pos_embedding.weight, std=0.1)
        else:
            nn.init.trunc_normal_(self.rank_embedding.weight, std=0.1)
            nn.init.trunc_normal_(self.file_embedding.weight, std=0.1)
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, extra_features: torch.Tensor | None = None) -> torch.Tensor:
        B = x.shape[0]

        # Piece features → (B, 64, d_model)
        if self.encoding == "bitboard":
            # x: (B, 12, 8, 8) → (B, 64, 12)
            piece_feats = x.reshape(B, x.shape[1], 64).permute(0, 2, 1)
            piece_emb = self.piece_projection(piece_feats)
        else:
            # x: (B, 64)  piece indices
            piece_emb = self.piece_embedding(x)

        # Positional encoding → (B, 64, d_model)
        if self.pos_enc == "flat":
            positions = torch.arange(64, device=x.device).unsqueeze(0).expand(B, -1)
            pos_emb = self.pos_embedding(positions)
        else:
            sq = torch.arange(64, device=x.device)
            files = (sq % 8).unsqueeze(0).expand(B, -1)
            ranks = (sq // 8).unsqueeze(0).expand(B, -1)
            pos_emb = self.file_embedding(files) + self.rank_embedding(ranks)

        tokens = piece_emb + pos_emb                                 # (B, 64, d_model)

        if self.pool == "cls":
            cls = self.cls_token.expand(B, -1, -1)               # (B,  1, d_model)
            tokens = torch.cat([cls, tokens], dim=1)              # (B, 65, d_model)
            out = self.transformer(tokens)                        # (B, 65, d_model)
            pooled = out[:, 0]                                    # CLS token
        else:
            out = self.transformer(tokens)                        # (B, 64, d_model)
            pooled = out.mean(dim=1)                              # mean over squares

        if extra_features is not None and self.num_extra_features > 0:
            pooled = torch.cat([pooled, extra_features], dim=-1)

        return self.head(pooled).squeeze(-1)  # (B,)
