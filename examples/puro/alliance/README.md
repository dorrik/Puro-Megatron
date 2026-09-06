# Running Puro-0.6B on Digital Research Alliance of Canada clusters

Scripts for training [`run_puro_0p6b.sh`](../run_puro_0p6b.sh) on Alliance HPC
(Fir, Nibi, Rorqual, Trillium, Killarney) using the NGC PyTorch container under
Apptainer.

## Why a container

`pyproject.toml` declares

```toml
override-dependencies = ["torch; sys_platform == 'never'", "torchvision; ...", "triton; ..."]
```

so `uv pip install torch` silently does nothing: the project is meant to be
installed *into* a PyTorch base image. `nvcr.io/nvidia/pytorch:26.01-py3` is the
matching tag — torch 2.10.0a0, TransformerEngine 2.11, CUDA 13.1 — which
satisfies the `>=2.9.0a0,<2.12.0` TE pin and skips a source build of TE.

## Order of operations

```bash
# 1. Build the SIF on a compute node's local NVMe (minutes, vs hours on Lustre)
sbatch pull_container.sh

# 2. Overlay venv: image torch/TE + the few extras Megatron needs
./setup_container_env.sh

# 3. Build the dataset (CPU allocation, no GPU hours)
python ../data/build_puro_data.py plan --out-dir $SCRATCH/puro/data_p1 \
    --phase phase1 --target-tokens 439e9
sbatch --array=0-7 --export=ALL,NUM_TASKS=8 data_array.sh
python ../data/build_puro_data.py finalize --out-dir $SCRATCH/puro/data_p1

# 4. Validate on ONE GPU before committing a long job
python ../data/make_smoke_data.py --out-dir $SCRATCH/puro/data_smoke
srun -A def-dorrik_gpu -N1 --gpus-per-node=h100:1 -c 12 --mem=100G -t 1:00:00 \
    ./smoke.sh 1 20

# 5. Measure throughput to size the campaign
./sweep.sh

# 6. Train. -N sets the node count; GPUs per node is auto-detected.
sbatch -N2 train.sh          # 8 GPUs on a 4-GPU-per-node cluster
```

## Cluster gotchas these scripts encode

Every item below was a real failure, most of them appearing only at run time.

**Container / environment**

- **Build the SIF on node-local disk.** Unpacking an NGC image writes hundreds of
  thousands of small files. On Lustre `$SCRATCH` this ran at ~1 GB per 40 minutes;
  with `APPTAINER_TMPDIR=$SLURM_TMPDIR` on NVMe it finished in ~4 minutes.
- **`PYTHONNOUSERSITE=1` is required.** `/home` is bind-mounted, so a pip-installed
  torch in `~/.local/lib/python3.12/site-packages` shadows the image's and fails as
  `ValueError: libcublas.so.*[0-9] not found in the system path` — which reads like a
  broken image rather than path shadowing.
- **`~/.local/bin` shadows image console scripts.** `torchrun` resolves to a host
  build and dies with `cannot execute: required file not found`. `PYTHONNOUSERSITE`
  fixes module imports but *not* executable lookup, so launch distributed jobs as
  `python -m torch.distributed.run` and strip `$HOME/.local/bin` from `PATH`.
- **Triton cannot find `libcuda.so`.** `torch.compile` fails minutes in, at the first
  compiled kernel, with `InductorError: AssertionError: libcuda.so cannot found!`.
  Triton locates the driver by grepping `ldconfig -p`, whose cache was baked at image
  build time and points at `/usr/local/cuda/compat/lib` — empty at run time (the
  compat libs are in `lib.real`). Because that lookup returns a non-empty list,
  Triton never reaches its `LD_LIBRARY_PATH` fallback. `--nv` injects the real driver
  at `/.singularity.d/libs`, so `ct.sh` binds a directory of symlinks over the stale
  path.
- **Host CA bundle paths do not exist in the image.** Alliance exports
  `CURL_CA_BUNDLE`/`SSL_CERT_FILE` as `/etc/pki/tls/certs/ca-bundle.crt` (RHEL); the
  Ubuntu-based image needs `/etc/ssl/certs/ca-certificates.crt`, or pip inside the
  container dies with "Could not find a suitable TLS CA certificate bundle".
- **No apostrophes** in the payload passed to `ct.sh`: it runs inside a single-quoted
  `bash -c '...'`, so one `'` in a comment truncates the script. `bash -n` before
  deploying.

**Scheduler**

- Jobs inherit the submitting shell's environment. Alliance sets `PYTHONPATH` and
  `PIP_CONFIG_FILE` at login and both shadow a venv or container; `unset` them.
- Slurm buffers job output in blocks and drops the tail if a job is cancelled, so a
  silent log is not evidence of a hang. Use `PYTHONUNBUFFERED=1`.
- Memory suffixes are binary: `--mem=64G` means 65536M.
- Do not pass `--partition`; let the scheduler choose. (Exception: tamIA has no
  default partition. `trillium-gpu` rejects any job without `--gpus-per-node`.)
- Interactive `salloc` jobs of **<= 3 h** run on dedicated test nodes and start almost
  immediately; longer ones queue with normal priority for hours or days.
- `uv` is not installed on Fir or Trillium; bootstrap it from `astral.sh`.

**Storage and connectivity** (measured 2026-09)

| Cluster | /scratch | GPUs/node | Compute-node internet |
|---|---:|---:|---|
| Nibi | 1.0 TiB | 8x H100 | yes |
| Killarney | 500 GiB | 8x H100 | yes |
| Fir | 19 TiB | 4x H100 | **yes** |
| Rorqual | 20 TB | 4x H100 | no |
| Trillium | 25 TiB | 4x H100 | no |

Only **Fir and Nibi** reach the internet from compute nodes, so on the others the
corpus download has to run on a login node. Fir is the only cluster with both a
large scratch and compute-node internet, which is why it is the default target
here: the tokenized Phase-1 set is ~1.8 TB and both phases are ~5.5 TB.

## Measured

One H100, Puro-0.6B, seq 4096, MBS 4, BF16:

```
Total number of parameters in billions: 0.60
throughput per GPU (TFLOP/s/GPU): 398.6      # 40.3% MFU of 989.4 dense BF16
elapsed time per iteration (ms): 819.7
max allocated: 35408.95 MB                   # of 81559 MB
```

Note H100 SXM BF16 dense peak is **989.4** TFLOP/s (1979 is the sparsity figure) and
FP8 dense is 1979; using the sparsity number halves apparent MFU and doubles every
schedule estimate.
