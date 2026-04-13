# overfit-tiny

## Purpose

Sanity check: can the model memorize a tiny dataset? If train loss doesn't drop,
the architecture or optimization is fundamentally broken.

## Config

Full-size model (6 layers, 256 dim), but only 500 samples, 100 epochs, no dropout.

## Results

| Metric | Value |
|--------|-------|
| Test RMSE | 251.4 Elo |
| Pearson r | 0.05 |
| Train loss | 1.096 → 0.167 |
| Val RMSE | 321.4 → 417.0 |

## Outcome

**The model CAN memorize** — train loss dropped from 1.09 to 0.17. But val RMSE
got *worse* (321 → 417), confirming pure memorization with zero generalization.
This ruled out a fundamentally broken architecture and pointed to optimization
or model size as the issue on the full dataset.
