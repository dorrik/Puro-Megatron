#!/bin/bash
# Run a command inside the NGC PyTorch container with the Puro env wired up.
#   ct.sh <command...>
# Sourced by every other script so the container invocation exists in one place.
set -euo pipefail

PURO_ROOT=${PURO_ROOT:-$SCRATCH/puro}
SIF=${SIF:-$PURO_ROOT/pytorch_26.01.sif}
REPO=${REPO:-$PURO_ROOT/Puro-Megatron}
CTVENV=${CTVENV:-$PURO_ROOT/venv_ct}

module load apptainer/1.4.5 2>/dev/null || true

# Alliance injects its own python stack via PYTHONPATH/PIP_CONFIG_FILE; inside
# the container that would shadow the image's torch. APPTAINERENV_* sets the
# variable *inside* the container.
export APPTAINERENV_PYTHONPATH="$REPO"
export APPTAINERENV_PIP_CONFIG_FILE=""
export APPTAINERENV_PYTHONNOUSERSITE=1
# The host exports CURL_CA_BUNDLE/SSL_CERT_FILE as RHEL paths
# (/etc/pki/tls/certs/ca-bundle.crt) that do not exist in the Ubuntu-based NGC
# image, so pip inside the container dies with "Could not find a suitable TLS
# CA certificate bundle". Point them at the image's own bundle.
export APPTAINERENV_CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export APPTAINERENV_SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export APPTAINERENV_REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
# /home is not reliably writable on compute nodes and torch.compile/triton
# write cache there; keep every cache on scratch.
export APPTAINERENV_TRITON_CACHE_DIR="$PURO_ROOT/.cache/triton"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$PURO_ROOT/.cache/inductor"
export APPTAINERENV_XDG_CACHE_HOME="$PURO_ROOT/.cache/xdg"
export APPTAINERENV_HF_HOME="$SCRATCH/.hf"
export APPTAINERENV_CTVENV="$CTVENV"
export APPTAINERENV_REPO="$REPO"
export APPTAINERENV_PURO_ROOT="$PURO_ROOT"
export APPTAINERENV_SCRATCH="$SCRATCH"
mkdir -p "$PURO_ROOT"/.cache/{triton,inductor,xdg}

# Triton resolves libcuda by grepping `ldconfig -p`, whose cache was baked at
# image build time and points at /usr/local/cuda/compat/lib -- a directory that
# is empty at run time (the compat libs are in lib.real). Because that lookup
# returns a non-empty list, Triton never reaches its LD_LIBRARY_PATH fallback
# and dies with "libcuda.so cannot found!". Apptainer --nv injects the real host
# driver at /.singularity.d/libs, so bind a directory of symlinks to it over the
# stale path. The symlinks are dangling on the host and resolve inside.
COMPAT="$PURO_ROOT/.cudacompat"
mkdir -p "$COMPAT"
[ -L "$COMPAT/libcuda.so.1" ] || ln -sf /.singularity.d/libs/libcuda.so.1 "$COMPAT/libcuda.so.1"
[ -L "$COMPAT/libcuda.so" ]   || ln -sf /.singularity.d/libs/libcuda.so   "$COMPAT/libcuda.so"

exec apptainer exec --nv \
  -B /scratch:/scratch \
  -B /home/dorrik:/home/dorrik \
  -B "$COMPAT":/usr/local/cuda/compat/lib \
  "$SIF" \
  bash -c '
    # Prefer the overlay venv (extra deps) but keep the image site-packages,
    # which is where torch / transformer_engine / apex live.
    # /home is bind-mounted, so ~/.local/bin from the login environment lands
    # on PATH inside the container and shadows image console scripts: torchrun
    # resolves to a host build that cannot execute here. NOTE: no apostrophes
    # in this block, it is inside a single-quoted bash -c string.
    export PATH="$(echo "$PATH" | tr ":" "\n" | grep -v "^$HOME/.local/bin$" | paste -sd: -)"
    if [ -d "$CTVENV" ]; then
      export PATH="$CTVENV/bin:$PATH"
      export PYTHONPATH="$REPO:$(ls -d $CTVENV/lib/python*/site-packages)"
    fi
    '"$*"'
  '
