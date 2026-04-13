# pos2d

## Purpose

Test whether 2D positional encoding (separate rank + file embeddings) helps the
model understand spatial relationships on the board. The flat encoding treats the
board as an arbitrary 64-token sequence; 2D encoding bakes in the grid structure.

## Config

Same as small-model-long (2 layers, 64 dim, 60 epochs) with pos_enc="2d".

## Results

| Metric | Value |
|--------|-------|
| Test RMSE | 243.2 Elo |
| Pearson r | 0.22 |
| Train loss | 0.989 → 0.923 |
| Val RMSE | 246.8 → 244.1 (best 243.1 at epoch 30) |

## Outcome

No improvement over flat positional encoding (r=0.22 vs 0.22). The model
was already learning spatial relationships through attention — giving it
explicit rank/file structure didn't help. The bottleneck is not spatial
awareness but the input encoding itself (piece indices carry limited info).
