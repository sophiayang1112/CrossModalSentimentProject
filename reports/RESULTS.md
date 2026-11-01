# Executed model comparison

Experiment: 600 generated companies, 1,200 opinion-pair samples, seed 31. Opinion encoder: learned token embeddings; no pretrained BERT run is included. Company overlap across partitions: 0.

| Variant | Accuracy | Weighted F1 | Macro F1 | Weighted precision | Weighted recall |
| :--- | ---: | ---: | ---: | ---: | ---: |
| majority_baseline | 0.350 | 0.181 | 0.173 | 0.122 | 0.350 |
| full | 0.758 | 0.759 | 0.760 | 0.761 | 0.758 |
| no_cross | 0.754 | 0.754 | 0.755 | 0.756 | 0.754 |
| no_mfb | 0.729 | 0.722 | 0.725 | 0.731 | 0.729 |
| recency_only | 0.762 | 0.763 | 0.764 | 0.763 | 0.762 |
| popularity_only | 0.404 | 0.395 | 0.398 | 0.416 | 0.404 |

## Findings

The full network reaches **75.8%** accuracy. No-cross accuracy is 75.4%, no-MFB accuracy is 72.9%, and recency-only accuracy is 76.2%. These single-seed differences do not establish statistical significance or reproduction of the paper's gains.

The recency-only stream remains competitive because the synthetic target mechanism gives recency information greater weight. The full model does not lead every variant. Representative market text, contextual representations, and repeated independent evaluations are needed to assess real cross-stream value.

## Training budget and evaluation provenance

The initial 15-epoch full-model run had accuracy 53.8%, with a validation curve still improving at the final epoch. The common training budget was extended to 50 epochs with validation early stopping, without changing the architecture or selecting checkpoints by test scores. [Initial experiment record](initial_15_epoch_metrics.json) is retained. Since this holdout was viewed during development, treat it as a development evaluation rather than an untouched final benchmark.

Sample counts: {'train': 720, 'validation': 240, 'test': 240}. Each variant uses its own validation macro-F1 checkpoint. The full architecture remains the default saved model, rather than selecting an architecture by the largest holdout score. Weighted recall equals accuracy for this single-label task.

## Verification

- Twelve automated tests passed: future-publication filtering, future-engagement exclusion, company grouping, vocabulary handling, metric consistency, attention normalization, padding invariance, batch consistency, gradient propagation, MFB factor pooling, single-stream independence and state-dict persistence.
- Editable installation and the installed scoring command succeeded.
- Eight checkpoint predictions have finite nonnegative class probabilities summing to one.
- Saved attention matrices have normalized rows and cite the actual selected opinion IDs.
- Figures and metrics were generated from the executed neural training run.
- Local BERT integration is implemented but not run; these numbers are not labeled as BERT results.
