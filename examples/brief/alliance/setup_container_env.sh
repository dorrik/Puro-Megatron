#!/bin/bash
# Create the overlay venv inside the NGC container: keeps the image's
# torch / transformer_engine / apex and adds only what Megatron/Puro needs
# on top. Run once, on the login node (no GPU required).
set -euo pipefail
PURO_ROOT=${PURO_ROOT:-$SCRATCH/puro}
cd "$PURO_ROOT"

"$PURO_ROOT/ct.sh" '
set -euo pipefail
echo "=== image baseline ==="
python -c "
import torch, sys
print(\"python\", sys.version.split()[0])
print(\"torch \", torch.__version__, \"cuda\", torch.version.cuda)
import transformer_engine as te; print(\"TE    \", te.__version__)
"

if [ ! -d "$CTVENV" ]; then
  echo "=== creating overlay venv ==="
  python -m venv --system-site-packages "$CTVENV"
fi
"$CTVENV/bin/pip" install -q --no-cache-dir --upgrade pip
echo "=== installing Puro/Megatron extras ==="
# Deliberately NOT installing torch/torchvision/triton: the image provides them
# and the repo pyproject overrides them to never-install for exactly this reason.
"$CTVENV/bin/pip" install -q --no-cache-dir \
  "git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0" \
  tensorstore nvtx multi-storage-client "opentelemetry-api~=1.33.1" \
  sentencepiece tiktoken wandb flask-restful wget onnxscript
echo "=== versions ==="
"$CTVENV/bin/python" - <<PY
import importlib
for m in ("torch","transformer_engine","emerging_optimizers","tensorstore","einops","transformers","numpy"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:22s} {getattr(mod,\"__version__\",\"?\")}")
    except Exception as e:
        print(f"  {m:22s} MISSING ({type(e).__name__}: {e})")
PY
'
