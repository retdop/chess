# bitboard-medium

## Purpose

The bitboard experiment showed higher train loss than piece_index models at the
same val RMSE, suggesting the 2L/64d model was capacity-limited. This experiment
scales up to 4 layers / 128 dim with all best settings (bitboard + 2D pos + mean
pooling) to test whether more capacity can break the r=0.22 ceiling.

## Config

4 layers, 128 dim, 4 heads, 512 FFN dim, dropout=0.1, lr=1e-3, 60 epochs,
encoding=bitboard, pos_enc=2d, pool=mean.

## Results (partial — training stopped at epoch 29/60 due to Modal credits)

| Metric | Value |
|--------|-------|
| Train loss (epoch 29) | 0.954 |
| Val RMSE (epoch 29) | 243.6 Elo |

For comparison, the small bitboard model at epoch 30 had train loss 0.954 and
val RMSE 243.7 — essentially identical.

## Outcome

**The 4x larger model performed identically to the small model at the same
epoch count.** More capacity did not break through the r≈0.22 ceiling. This
confirms that the limitation is fundamental: static board positions (piece
placement alone) contain only ~5% of the information needed to predict puzzle
difficulty. Breaking past this ceiling would require fundamentally different
input features — move sequences, puzzle themes, or game context.
