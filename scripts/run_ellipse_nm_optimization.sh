#!/bin/bash
#SBATCH --job-name=ell-nm
#SBATCH --time=4-23:59:59
#SBATCH --mail-user=buresjan@protonmail.com
#SBATCH --partition=gp
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=ellipse_nm_%j.out
#SBATCH --error=ellipse_nm_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

python3 "$REPO_ROOT/scripts/ellipse_nm_optimization.py" "$@"
