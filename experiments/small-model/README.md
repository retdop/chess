# small-model

## Purpose

Test whether a simpler model (fewer params, shallower) learns where the
baseline 6-layer model couldn't. Hypothesis: the deep model gets stuck in
a mean-prediction local minimum.

## Config

2 layers, 64 dim, 4 heads, lr=3e-3, 20 epochs.

## Results

| Metric | Value |
|--------|-------|
| Test RMSE | 245.9 Elo |
| Pearson r | 0.16 |
| Train loss | 0.986 → 0.969 |
| Val RMSE | 247.1 → 245.8 |

## Outcome

The small model learned more than the deep baseline (r=0.16 vs ~0), confirming
that a shallower model optimizes more easily. Train loss was still declining at
epoch 20, suggesting more training would help. This led to the small-model-long
experiment.
