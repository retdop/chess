# Future Ideas

## Features
- Add side-to-move as an explicit input token (extra 65th square or a learned bias added to all tokens)
- Encode castling rights (4 bits: KQkq) as additional input features
- Encode en-passant square availability
- Include the puzzle solution length (number of moves) as a feature or auxiliary target
- Include puzzle themes (fork, pin, mate-in-N, etc.) as auxiliary classification targets for multi-task learning

## Loss & Training
- Try `SmoothL1Loss` (Huber) if rating outliers dominate the gradient
- Experiment with auxiliary losses (e.g. predict piece count, material balance) to enrich representations
- Try longer training runs (40-60 epochs) now that the model can actually learn
- Sweep learning rate (5e-4 to 3e-3) and warmup fraction (0.05 to 0.2)
- Exclude weight decay from embeddings and LayerNorm parameters

## Architecture
- Try a smaller model (4 layers, 128 dim) to check if the current model is over-parameterised for board-only features
- Add a learned global pooling (attention pooling over all 64 squares) instead of relying solely on the CLS token
- Try relative position embeddings or 2D-aware positional encoding (rank + file) instead of flat 1-64

## Data
- Augment by mirroring boards horizontally (flip files a-h) to double the dataset
- Stratified sampling so the model sees equal numbers of easy / medium / hard puzzles per batch
- Pre-compute and cache tensors to disk to avoid repeated FEN parsing during training
