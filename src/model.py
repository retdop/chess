import torch
import torch.nn as nn


class ChessPuzzleTransformer(nn.Module):
    """
    Treats a chess position as a sequence of 64 square tokens.
    Each token = piece_embedding(piece_idx) + pos_embedding(square_idx).

    Pooling strategy is configurable via *pool*:
      - ``"cls"`` (default): a learnable [CLS] token is prepended and its
        final hidden state drives the rating head.
      - ``"mean"``: the 64 square outputs are averaged to form the input
        to the rating head (no CLS token).
    """

    def __init__(
        self,
        num_piece_types: int = 13,   # 0=empty, 1-12=pieces
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pool: str = "cls",
    ):
        super().__init__()
        if pool not in ("cls", "mean"):
            raise ValueError(f"pool must be 'cls' or 'mean', got {pool!r}")
        self.pool = pool

        self.piece_embedding = nn.Embedding(num_piece_types, d_model)
        self.pos_embedding = nn.Embedding(64, d_model)
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

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.piece_embedding.weight, std=0.1)
        nn.init.trunc_normal_(self.pos_embedding.weight, std=0.1)
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 64)  piece indices per square
        B = x.shape[0]
        positions = torch.arange(64, device=x.device).unsqueeze(0).expand(B, -1)

        tokens = self.piece_embedding(x) + self.pos_embedding(positions)  # (B, 64, d_model)

        if self.pool == "cls":
            cls = self.cls_token.expand(B, -1, -1)               # (B,  1, d_model)
            tokens = torch.cat([cls, tokens], dim=1)              # (B, 65, d_model)
            out = self.transformer(tokens)                        # (B, 65, d_model)
            pooled = out[:, 0]                                    # CLS token
        else:
            out = self.transformer(tokens)                        # (B, 64, d_model)
            pooled = out.mean(dim=1)                              # mean over squares

        return self.head(pooled).squeeze(-1)  # (B,)
