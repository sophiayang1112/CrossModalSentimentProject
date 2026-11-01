# Research source and implementation mapping

Source: *Multi-Modal Opinion Integration for Financial Sentiment Analysis using Cross-Modal Attention*, supplied manuscript by **Yujing Liu and Chen Yang**. The PDF did not provide a verified publication identifier for this implementation. Author contact details and the source PDF are not redistributed.

This repository independently implements the described methods and attributes the architecture to the source authors. It does not claim authorship of the paper or reproduction of the paper's 837-company dataset, 83.5% accuracy, or reported BERT results. Code, synthetic records, tests, and result figures are newly created.

| Paper component | Implementation | Published validation |
| :--- | :--- | :--- |
| Recency and popularity opinion streams | Separate variable-length textual channels with as-of selection | Executed on generated records |
| Chinese-wwm-ext BERT | Optional local Hugging Face BERT encoder with frozen backbone | Adapter implemented; pretrained execution not run |
| BERT 768-to-128 projection | Local encoder hidden size to configurable model dimension | Not claimed in the learned-encoder benchmark |
| Financial multi-head cross-attention | Trending-to-timely, then attended-to-trending attention | Trained with head/padding/gradient tests |
| Transformer refinement | Separate encoder blocks on refined and trending paths | Trained |
| MFB | Factorized elementwise products, factor summation and normalization | Trained and tested |
| Negative/neutral/positive output | Three-class neural classifier | Executed |
| Baselines and ablations | Majority, two single streams, no-cross and no-MFB | Executed; not labeled as BERT baselines |
| Original 837 companies and manual annotations | 600 generated companies, two windows each | Separate synthetic experiment |
| Original GPU recipe | CPU, 64 dimensions, four heads and factors, early stopping | Actual settings recorded in metrics.json |

## Equation interpretation

The two cross-attention equations return a sequence indexed by trending queries in both stages. This implementation preserves that indexing and uses its CLS output. It does not imply that stage 2 reconstructs a sequence with the original timely opinion count. Each document is encoded independently before opinion-level attention.

The paper omits some tensor layout, masking and normalization details. The repository makes them explicit: batch-first tensors, per-opinion token masking, stream-level CLS and positions, two residual normalization steps, and signed-square-root/L2 MFB normalization. These choices enable a working implementation and are documented differences, not assertions about undisclosed source code.

## General references

- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [PyTorch TransformerEncoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)
- [Hugging Face local model loading](https://huggingface.co/docs/transformers/installation#offline-mode)
- [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)
