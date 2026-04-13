# small-model-long

## Purpose

The small-model experiment showed train loss still declining at epoch 20.
This experiment trains 3x longer to see how far the small model can go.

## Config

Same as small-model (2 layers, 64 dim, 4 heads, lr=3e-3) but 60 epochs
and dropout reduced to 0.05.

## Results

| Metric | Value |
|--------|-------|
| Test RMSE | 243.5 Elo |
| Pearson r | 0.22 |
| Train loss | 0.985 → 0.923 |
| Val RMSE | 246.5 → 244.2 (best 243.3 at epoch 27) |

## Outcome

Pearson r jumped from 0.16 to 0.22 with longer training. However, val RMSE
plateaued around epoch 27 while train loss kept dropping — the model is
**underfitting** (not overfitting). It explains only ~8% of variance even on
training data. The train-val gap is tiny, meaning the bottleneck is input
representation, not model capacity. This became the baseline config for all
subsequent experiments.
