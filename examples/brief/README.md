# Brief

Brief is a ~0.6B language model for real-time systems, trained from scratch.

- **Recipe: Puro** (arXiv:2608.27370): MuonHyperball with AdamW routing, effective LR as the
  primary hyperparameter, the open-ended power schedule in phase 1 and linear decay in phase 2,
  the two-phase Puro corpus, Qwen3-family architecture and tokenizer. Blockwise FP8 was
  evaluated and not adopted (+1.1% at this scale on H100).
- **Method: Marin's Hero Run** (marin-community/marin#8435): a scaling ladder at fixed tokens
  per parameter, preregistered loss predictions checked every 5% of training, gradient-norm
  monitoring against the small rungs, z-loss, and data provenance with decontamination.

`run_brief_0p6b.sh` is the recipe. Cluster job scripts live in `alliance/`, the corpus
pipeline in `data/build_brief_data.py`. `../puro/run_puro_2b.sh` is the upstream Puro-2B recipe
this one is derived from. The budget and phased plan are on the Brief Frontier Budget page.
