# mean-pool

## Purpose

Test whether the CLS token is a bottleneck. Mean pooling averages all 64 square
embeddings instead of relying on a single learnable CLS token to attend to the
board. Hypothesis: the CLS token in a shallow 2-layer model might not attend
broadly enough.

## Config

Same as small-model-long (2 layers, 64 dim, 60 epochs) but with pool="mean"
instead of "cls".

## Results

| Metric | Value |
|--------|-------|
| Test RMSE | 243.3 Elo |
| Pearson r | 0.22 |
| Train loss | 0.986 → 0.918 |
| Val RMSE | 246.5 → 244.4 (best 243.2 at epoch 30) |

## Outcome

Essentially identical to CLS pooling (r=0.22 vs 0.22). The CLS token is
**not** the bottleneck. Both pooling strategies hit the same ceiling, confirming
the limitation is in the input representation, not the aggregation method.
