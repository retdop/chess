# bitboard

## Purpose

Switch from piece-index encoding (single integer 0-12 per square) to bitboard
planes (12 binary 8x8 planes), matching the AlphaZero/Leela Chess Zero input
paradigm. Each plane represents one piece type, giving the model explicit
per-piece-type spatial features instead of a compressed integer.

## Config

Same as small-model-long (2 layers, 64 dim, 60 epochs) with encoding="bitboard"
and pos_enc="2d".

## Results

| Metric | Value |
|--------|-------|
| Test RMSE | 243.2 Elo |
| Pearson r | 0.22 |
| Train loss | 0.992 → 0.938 |
| Val RMSE | 247.1 → 243.1 (best 243.1 at epoch 50) |

## Outcome

Same r=0.22 ceiling. However, train loss (0.938) is higher than piece_index
models (0.918-0.923) at the same val RMSE, suggesting the small 2L/64d model
is capacity-limited with the richer input. This motivated the bitboard-medium
experiment with a larger model.
