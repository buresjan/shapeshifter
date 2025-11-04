#!/bin/bash
# Slurm launcher for the TCPC Nelder–Mead optimisation.
#
# Usage (Slurm):   sbatch playground/optimize_junction_tcpc.sh
# Usage (local):   ./playground/optimize_junction_tcpc.sh --local
#
# The Python script has all parameters embedded; no CLI flags are used.

#SBATCH --job-name=opt-tcpc-nm
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=opt-tcpc-nm-%j.out

set -euo pipefail

LOCAL_RUN=0
if [[ "${1:-}" == "--local" ]]; then
    LOCAL_RUN=1
    shift
fi

export OPT_NM_WORKERS=${OPT_NM_WORKERS:-8}

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(realpath "${SLURM_SUBMIT_DIR}")"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if (( LOCAL_RUN )); then
        REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    else
        REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
fi

SCRIPT_PATH="${REPO_ROOT}/playground/optimize_junction_tcpc.py"

cd "${REPO_ROOT}"

echo "Launching TCPC Nelder–Mead optimisation"
python3 "${SCRIPT_PATH}"
