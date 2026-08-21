# Source and commit audit

## Fork point

Puro-Megatron forks NVIDIA Megatron-LM at tag `core_v0.16.0`, exact commit:

```text
3bec9aa97dda898d16ff5a89bac0ed2b6682b172
```

The internal branch lineage was `effective_lr` ->
`numeric_efflr_wd_experiment` -> `scaling_ladder`, whose inspected tips were
`57dd9dc23`, `8b9344a67`, and `5df860dc7`. The release does not fork from any
of those tips. It replays only the selected changes onto the clean NVIDIA tag.

## Public patch series

The authoritative list is `git log --reverse core_v0.16.0..main`. The commits
are grouped below by capability. Curated commits preserve the primary original
author and record all folded source SHAs in the commit message.

| Public commit | Source commit(s) | Capability |
| --- | --- | --- |
| `b7ff60000` | `dd8b4e614`, `ee94062b3`, `aeed231b2`, `5d100adbe`, `4bcd2f63f`, `324b68046`, `af4ccff8f`, `907c4e669`, `e89eb65d8`, `bb90af489`, `d70af3702` | PROM NPY data, packed THD batches, stable resume indexing |
| `a2486d4e7` | `fddde3891` | Route GPT batches through the custom TP loader |
| `af2d1bf2b`, `14d350d3d` | `a071673f1`, `dbbebb670` | Transformer Engine FP8 extra-state compatibility |
| `1390f5847` | `0ff1b4fd2` | Restore runtime LR bounds after checkpoint override |
| `59b6561bd` | `9239c1e90` | Reset scheduler/data progress for phase resume |
| `92f4f44c7` | `12e40b001` | Non-persistent checkpoint retention |
| `68c18d86d` | `34a35709f` | Honor explicit persistent checkpoint step |
| `93bafa203` | `6e4a9a6d4`, `3e76e52e2`, `87df41c57`, `6a9ae0869` | MuonHyperball, AdamW routing, effective-LR diagnostics |
| `d18236839` | `1e0a5aab3`, `9d0cc6753`, `edc7f437a`, `75d74a7f4` | Memory-balanced layer-wise distributed optimizer |
| `469b91917` | `57dd9dc23` | Strict effective-LR control |
| `bfc0fb953` | `2f7d55872` | Open-ended power LR schedule |
| `fb89d3fa4` | `1aa8d4cc3` | Custom dummy-batch consumption during rerun skip |
| `636cc5a40`, `1b7d9c26a` | same SHAs | Corrupt-data skip and cross-rank request synchronization |
| `55666e70d`, `38f66d6e5` | same SHAs | Startup theoretical FLOPs report and TP/SP correction |
| `21426d021` | release-only | Align stale FLOPs formatting tests; no formula change |

The short public SHAs above identify this prepared local history. Use the full
SHAs from Git directly when scripting because rebasing for publication would
change them.

## Deliberately excluded

- `numeric_efflr_wd_experiment`: two-stage and numeric WD sweeps, instant
  checkpoint plans, TE runtime monkey patches, and experimental logging.
- `scaling_ladder`: SFT/curriculum work, evaluation suites, long-context
  experiments, gated attention/norm/activation variants, and data ablations.
- hardware-specific DCU ports, profiler output, cached datasets, checkpoints,
  and run artifacts.

These exclusions keep the public repository reviewable while retaining the
code paths exercised by the release regression runs.
