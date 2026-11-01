# Data and methodology

## Input schema

Each sample contains `sample_id`, `company_id`, `as_of`, `recency`, and `popularity`. Training samples additionally contain `label`: 0 = negative, 1 = neutral, 2 = positive. Each stream is a list of opinions with `opinion_id`, `text`, `published_at`, and `available_at`. Popularity opinions also require nonnegative `engagement` and an `engagement_observed_at` timestamp.

Timestamp inputs normalize to UTC. Eligible publications and their availability must be at or before assessment time. Popularity counts must also be observed by then; future engagement cannot rank past opinions. Timely opinions sort newest first and popularity opinions sort by engagement, then publication time. Each stream keeps at most five opinions by default. No eligible opinions in a stream produces an explicit input error. Repeated opinion IDs within a stream are deduplicated.

## Synthetic experiment

The generator creates 600 companies with two assessment windows each, 1-5 valid opinions per stream, plus an additional future opinion that must be filtered out. Text is generated from controlled financial phrases. Latent recency and popularity polarity combine with weights 0.7 and 0.3, plus Gaussian noise, to generate three-class labels. Individual opinions have polarity noise. This defines a transparent simulation, not an empirical market-sentiment finding.

Companies are split 60/20/20 into training, validation and test. Both assessment windows stay together. The split is not chronological, and this experiment does not estimate future-market performance. There is no pretraining or original dataset in the published learned-encoder run.

## Encoders and dimensions

Default opinion representations average learned non-padding token embeddings, then apply a learned linear projection and ReLU. This is not contextual BERT. Its vocabulary is learned on training opinions only, with explicit padding and unknown-token IDs. The network uses dimension 64, four heads, factor count four, dropout 0.1, and up to five opinions per stream.

The optional local BERT path uses each opinion's final CLS representation, projected to the network dimension. It freezes the pretrained backbone and does not fine-tune BERT. Only valid opinion rows enter BERT; padded rows are masked. No model files are downloaded automatically.

## Cross-attention and fusion

Both streams receive separate learned CLS vectors and shared positional embeddings. Stage 1 uses trending queries with timely keys/values. Stage 2 uses the attended trending-indexed sequence as queries and the original trending sequence as keys/values. Residual connections and layer normalization stabilize both stages. This follows the supplied equations: the refined sequence is indexed by trending opinions, not a separately reconstructed timely-length sequence.

Two independent Transformer encoder layers refine the attended and original trending paths. Their CLS vectors enter MFB: two bias-free projections produce dimension-by-factor features; elementwise products are summed over factors, then signed-square-root and L2 normalization produce the fused vector. A linear classifier yields three logits. Normalization and residual details are explicit implementation choices because the paper does not specify them fully.

Key-padding masks exclude padded opinion columns. Padded query rows cannot influence valid outputs because downstream self-attention also masks those positions. Tests verify padded-opinion invariance, per-head attention normalization, batch/individual consistency, and gradients reaching both attention stages and MFB.

## Training and measurement

An initial 15-epoch run was extended to a common 50-epoch budget because the full model validation curve was still improving; the initial report is preserved. The holdout was observed during development and is not claimed as an untouched final benchmark.

The five neural variants train independently using the same company split, seed, class weights computed from training labels, Adam optimizer, gradient clipping at 1.0, and validation macro-F1 checkpoint selection. Defaults: learning rate 0.002, batch size 32, up to 50 epochs, early-stopping patience eight. These settings target CPU training of the smaller learned encoder and differ from the paper's BERT training recipe.

The primary full architecture is declared in advance; model comparisons are not used to choose a new architecture from the holdout. Variants have different parameter counts, which are reported. The no-cross variant still uses Transformer processing and MFB; no-MFB replaces bilinear pooling with a learned concatenation projection. Single-stream outputs do not use the other stream's content.

Report accuracy, weighted/macro F1, weighted precision, weighted recall and a three-class confusion matrix. For complete single-label classification, weighted recall equals accuracy. Batch inference timing includes tokenization, batching and neural forward passes, but excludes training, checkpoint loading and file I/O; it is not a deployment-latency benchmark.

## Limits

This is two textual channels, not audio/image fusion. Learned synthetic phrase representations cannot establish contextual language understanding or Chinese-BERT performance. Company-disjoint evaluation does not prevent every potential form of real-world leakage, and requires representative data permissions, timestamp/engagement availability checks, entity handling, temporal evaluation and robustness analysis before real financial use. Attention plots are qualitative routing examples, not explanation or fairness guarantees.
