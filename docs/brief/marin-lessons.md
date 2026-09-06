# What Brief takes from Marin's Hero Run code

Source: `marin-community/marin`, `experiments/grug/moe_hero_ep/` and the modules it imports,
read 2026-09-05. Marin's hero is a 535B-A23B MoE on GB200s; the pieces below are the ones that
transfer to a dense or hybrid 0.6B trained with the Puro recipe.

## 1. A fitted hyperparameter law instead of per-rung sweeps (`heuristic.py`)

Marin fits learning rate to the token budget, width and batch (Aug hero refit, R^2 = 0.978):

    adam_lr  = 0.087571 * tokens^-0.3461 * hidden^-0.3448 * sqrt(tokens_per_batch)   (cap 0.05)
    muonh_lr = (13/3) * adam_lr                                                        (cap 0.05)
    beta2    = clip(0.999^(tokens_per_batch / 131072), 0.95, 0.9999)
    epsilon  = 9.676e-18 * sqrt(tokens / tokens_per_batch)

`muonh_lr` is the hyperball relative step, the same quantity Puro calls the effective LR
(`--lr` x `--muon-hyperball-lr-mult`). For Brief (hidden 1024, GBS 512 x 4096):

| tokens | MuonH effective LR | Adam LR | Puro reference |
|---:|---:|---:|---|
| 12B   | 0.0164 | 3.8e-3 | Puro F.2 sweep optimum 0.012 at this horizon |
| 439B  | 0.0047 | 1.1e-3 | Phase 1 |
| 1.38T | 0.0032 | 7.3e-4 | full Puro corpus |
| 4T    | 0.0022 | 5.1e-4 | |

The two recipes agree at the one point where Puro measured (0.012 vs 0.016, adjacent grid cells),
and the law answers the question Puro's sweep could not: how the optimum falls with horizon
(exponent -0.346). It also gives every ladder rung its own LR without a sweep
(188M: 0.0178, 365M: 0.0123, 596M: 0.0094 at 100 tokens per parameter). `run_brief_0p6b.sh`
computes this with `MARIN_LR=1`. The fit was made on Marin's MoE at sequence length 8192, so the
first LR ablation is Puro's 0.012 against the law's 0.016 at 12B tokens, not a five-point grid.

## 2. MuonH mechanics (`optimizer.py`, `grugmuon_hero.py`, `adamh.py`)

| | Marin | Puro / Puro-Megatron |
|---|---|---|
| momentum / Nesterov | 0.95 / on | 0.9 default / off unless `--muon-nesterov` |
| Newton-Schulz | 5 steps, quintic coefficients | 5 steps, quintic (same table via emerging_optimizers) |
| post-NS scale | sqrt(max(1, fan_out/fan_in)) | `--muon-scale-mode spectral` (same rule) |
| hyperball step | `p - lr*u*|p|/|u|`, renormalised to `|p|` | same, per logical matrix with `separate` modes |
| LM head | **AdamH** (Adam with the hyperball projection) | AdamW |
| embeddings, norms, 1-D | Adam, no decay | AdamW, decay 0.1 |
| gradient clipping | **none** ("1pct-noclip") | `--clip-grad 1.0` |
| weight decay | only on attention gate and router | 0.1 on the AdamW route |
| LR groups | muonh / adamh / adam | muon / scalar |

`MARIN_OPT=1` in the recipe switches momentum 0.95, Nesterov on, clipping off and the beta2 rule.
AdamH for the head has no Puro-Megatron equivalent yet; it is an ablation to implement, not a flag.

## 3. Training loop (`train.py`)

- **z-loss** `1e-4` on the final-logit logsumexp is on by default. Not in Puro-Megatron; the
  TE fused cross-entropy path (`--cross-entropy-fusion-impl te`) will need a separate logsumexp
  term or the unfused path. Implementation task before the long run.
- **EMA of weights** (`ema_beta`) evaluated alongside live weights. Not in Puro-Megatron.
- **Watch statistics every step**: grad and parameter norms, per parameter, split per layer;
  histograms optional; logged every 10 steps. Puro-Megatron has `--log-params-norm` and the
  grad norm in the training log; per-parameter norms are the gap.
- **Eval cadence** every 5% of the run (`steps_per_eval = num_steps / 20`), on live and EMA
  weights, in **bits per byte** so tokenizers compare.
- `stop_after_steps` vs `num_train_steps`: train the head of a longer schedule. Puro's
  open-ended power schedule serves the same purpose.
- Warmup 1% of steps; linear decay to 5% of peak.

## 4. Model tricks (`model.py`, `heuristic.py`)

- init std `0.5 / sqrt(hidden)` (0.0156 at hidden 1024; Qwen3 uses 0.02). Under a hyperball
  optimizer the init sets the weight norm for the whole run, so this is a first-class knob.
- `qk_mult = 1.3` on queries; short conv (`sconv`, kernel 4) at the K, attention-output and MLP
  sites; sliding window 2048 with full attention every 4th layer and a smaller KV head count
  on the global layers. All three are cheap architecture ablations for the dense Brief shape.

## 5. Scaling ladder (`launch_scaling_ladder.py`, issue #8435)

Same recipe at five widths, tokens fixed at 791 per active parameter, batch fixed, eval every
5%. Fit `L = E + A * C^-alpha` per 5% of training, extrapolate to the hero, and validate the
method on each rung by predicting its 100% loss from its own 60-80% window (they landed within
0.003-0.004). Cost about 1% of hero compute. Brief's ladder: 188M / 365M / 596M at 100 tokens
per parameter, same fit, same self-check.

## 6. Data (`harrier_mix_*.py`, `launch_datakit_moe_mix.py`, `token-counts`)

- Provenance table with token counts and licenses per source; MinHash dedup; n-gram
  decontamination against the evaluation suites before training.
- Documents bucketed into 40 topic clusters x 5 quality levels = 200 cells; **two phases, the
  second (cooldown, 20% of tokens) shifting weight to higher-quality cells**; an eight-epoch cap
  per cell; phase boundary at 80% of steps rounded to the mixture block (49,152 sequences).
- **Simulated epoching**: a short run declares `target_budget` (the hero's) and its own
  `experiment_budget`, and the mixture is scaled so the rung sees the same phase boundary and
  per-cell exposure the hero will. Brief's ladder rungs should cross the Puro phase-1 to
  phase-2 boundary at the same fraction as the long run for the same reason.
- Validation sets ride along as zero-weight mixture components so they are tagged evals.

## 7. Neutral evaluation (`experiments/datasets/uncheatable.py`, `paloma.py`)

- **Uncheatable Eval** (GitHub `ziqing-huang/uncheatable_eval`, public): dated dumps of
  Wikipedia, BBC News, arXiv physics and CS, GitHub Python and C++, AO3. Latest English slices
  are **2025-09-01 to 2025-09-14**: after Qwen3-0.6B's training but not after Qwen3.5-0.8B's.
  For a clean comparison against Qwen3.5, Brief needs a fresher dump (Marin's downloader is
  generic; a September-2026 crawl of the same sources is a CPU job).
- **Paloma** (`allenai/paloma`): c4_en, c4_100_domains, mC4, Dolma programming languages and
  subreddits, M2D2 S2ORC and Wikipedia, RedPajama, PTB, WikiText-103, plus 4chan/Gab/manosphere.
- Score everything in bits per byte.

## Ablation list this adds for Brief

1. Effective peak LR: Puro 0.012 vs law 0.016 (12B tokens); then the law's horizon scaling vs a
   fixed peak on the 60B rung.
2. `MARIN_OPT`: momentum 0.95 + Nesterov + no clipping + beta2 0.984 vs Puro's 0.9 / off / 1.0 / 0.95.
3. z-loss 1e-4 on/off (needs implementation).
4. init std 0.5/sqrt(h) vs 0.02 under the hyperball.
5. qk_mult 1.3; sconv; local/global attention (dense shape only).
6. AdamH on the LM head (needs implementation).
