# Future Ideas

## Features
- Add side-to-move as an explicit input token (extra 65th square or a learned bias added to all tokens)
- Encode castling rights (4 bits: KQkq) as additional input features
- Encode en-passant square availability
- ~~Include the puzzle solution length (number of moves) as a feature or auxiliary target~~ ✅ `use_move_count`
- Include puzzle themes (fork, pin, mate-in-N, etc.) as auxiliary classification targets for multi-task learning
- Feed Stockfish centipawn evaluation of each position as an extra feature (engine difficulty proxy)
- Encode material balance (piece count difference) as explicit features instead of letting the model learn it

## Loss & Training
- ~~Try `SmoothL1Loss` (Huber) if rating outliers dominate the gradient~~ ✅ `loss_fn: "huber"`
- Experiment with auxiliary losses (e.g. predict piece count, material balance) to enrich representations
- Try longer training runs (40-60 epochs) now that the model can actually learn
- Sweep learning rate (5e-4 to 3e-3) and warmup fraction (0.05 to 0.2)
- Exclude weight decay from embeddings and LayerNorm parameters
- ~~Inverse-RD loss weighting to down-weight uncertain puzzle labels~~ ✅ `rd_weighted`
- ~~Stochastic target sampling from N(rating, RD²) for label smoothing~~ ✅ `stochastic_targets`
- Try Schedule-Free optimizer (referenced by GlickFormer team) instead of AdamW + cosine
- Post-hoc distribution calibration: fit a simple nonlinear mapping (isotonic regression or polynomial) from predicted to true ratings on validation set — top teams report 30%+ MSE improvement
- Gradient accumulation to simulate larger batch sizes without OOM on T4

## Architecture
- ~~Try a smaller model (4 layers, 128 dim) to check if the current model is over-parameterised for board-only features~~ ✅ tested
- Add a learned global pooling (attention pooling over all 64 squares) instead of relying solely on the CLS token
- ~~Try relative position embeddings or 2D-aware positional encoding (rank + file) instead of flat 1-64~~ ✅ `pos_enc: "2d"`
- ~~Encode full solution move sequence with GRU temporal aggregation~~ ✅ `use_solution_seq`
- Try BiLSTM instead of GRU for temporal aggregation (bidirectional may help)
- Scale up model now that solution sequences work: try 4-layer 128d or 6-layer 256d with the sequence encoder
- Add move prediction auxiliary head: predict the correct solution move from each position embedding, feed prediction uncertainty (cross-entropy) back as an input feature to the rating head (Schütt's key insight)
- Add a CNN encoder path (4 conv layers + batch norm) as alternative to the transformer for board encoding — simpler, faster, may work as well for the small model regime
- Try a temporal transformer (cross-attention over the position sequence) instead of GRU

## Data
- ~~Augment by mirroring boards horizontally (flip files a-h) to double the dataset~~ ✅ `augment_flip`
- Stratified sampling so the model sees equal numbers of easy / medium / hard puzzles per batch
- Pre-compute and cache tensors to disk to avoid repeated FEN parsing during training
- ~~Raise max_rating_deviation to use more training data with rd_weighted handling noise~~ ✅ `max_rating_deviation` config
- Use full dataset (4M+ puzzles) with gradient checkpointing or smaller batch to fit on T4
- Curriculum learning: train first on high-confidence (low RD) puzzles, then gradually include noisier ones

## Pretrained Models
- Fine-tune Maia-2 (current SOTA, MSE ~52k): use ResNet backbone for position embeddings + transformer, fine-tune end-to-end with rating MSE loss
- Use Leela Chess Zero (lc0) embeddings as frozen position features — extract the penultimate layer of lc0's value head as board representation
- Distillation: train the small transformer to match Stockfish evaluation on each position, then fine-tune on puzzle ratings
