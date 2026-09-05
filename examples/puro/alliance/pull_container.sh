#!/bin/bash
#SBATCH --account=def-dorrik_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --tmp=120G
#SBATCH --time=2:00:00
#SBATCH --job-name=puro-pull
#SBATCH --output=/scratch/dorrik/puro/logs/pull_%j.out

set -euo pipefail
module purge 2>/dev/null || true
module load apptainer/1.4.5
unset PYTHONPATH PIP_CONFIG_FILE

# Unpacking an NGC image writes hundreds of thousands of small files. On Lustre
# that ran at ~1 GB per 40 min; node-local NVMe finishes it in minutes. Only the
# finished single-file SIF goes back to /scratch.
export APPTAINER_TMPDIR="$SLURM_TMPDIR/atmp"
export APPTAINER_CACHEDIR="$SLURM_TMPDIR/acache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

echo "host=$(hostname) tmpdir=$SLURM_TMPDIR start=$(date -Is)"
df -h "$SLURM_TMPDIR" | tail -1

cd "$SLURM_TMPDIR"
time apptainer pull --force pytorch_26.01.sif docker://nvcr.io/nvidia/pytorch:26.01-py3

ls -la "$SLURM_TMPDIR/pytorch_26.01.sif"
echo "=== copying SIF to scratch ==="
time cp "$SLURM_TMPDIR/pytorch_26.01.sif" /scratch/dorrik/puro/pytorch_26.01.sif.partial
mv /scratch/dorrik/puro/pytorch_26.01.sif.partial /scratch/dorrik/puro/pytorch_26.01.sif
ls -la /scratch/dorrik/puro/pytorch_26.01.sif

echo "=== sanity: image versions ==="
apptainer exec /scratch/dorrik/puro/pytorch_26.01.sif python -c "
import torch, transformer_engine as te
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('TE   ', te.__version__)
"
echo "PULL_DONE $(date -Is)"
