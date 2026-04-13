# long-warmup

## Purpose

Test whether keeping the learning rate high for longer helps the baseline
6-layer model. A 30% warmup fraction means the LR stays near peak for more
of training before cosine decay kicks in.

## Config

Baseline architecture (6 layers, 256 dim) with warmup_frac=0.3, 20 epochs.

## Results

| Metric | Value |
|--------|-------|
| Test RMSE | 244.8 Elo |
| Pearson r | 0.19 |
| Train loss | 0.989 → 0.961 |
| Val RMSE | 246.3 → 244.7 |

## Outcome

Best result for the deep model (r=0.19), confirming that LR schedule matters.
But still worse than the small model trained longer (r=0.22), reinforcing that
simpler + longer beats deeper + shorter for this task.
