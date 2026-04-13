# high-lr

## Purpose

Test whether the baseline 6-layer/256-dim model just needs a higher learning
rate to escape the mean-prediction local minimum.

## Config

Baseline architecture (6 layers, 256 dim) with lr=3e-3 (3x default), 20 epochs.

## Results

| Metric | Value |
|--------|-------|
| Test RMSE | 247.2 Elo |
| Pearson r | 0.13 |
| Train loss | 0.986 → 0.984 |
| Val RMSE | 247.0 → 247.3 |

## Outcome

Higher LR on the deep model barely helped (r=0.13), and train loss was nearly
flat. The 6-layer model is fundamentally harder to optimize for this task —
the problem is model depth, not learning rate. The small model with default LR
outperformed it.
