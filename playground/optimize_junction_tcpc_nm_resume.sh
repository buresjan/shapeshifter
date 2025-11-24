#!/bin/bash
# Slurm launcher for the TCPC Nelder–Mead resume optimisation.
#
# Usage (Slurm):   sbatch playground/optimize_junction_tcpc_nm_resume.sh
# Usage (local):   ./playground/optimize_junction_tcpc_nm_resume.sh --local
#
# The Python script has all parameters embedded; no CLI flags are used.

#SBATCH --job-name=nm-opt-tcpc-resume
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=opt-tcpc-nm-resume-%j.out
# Append to stdout/stderr instead of truncating on restart/requeue
# (prevents losing earlier logs when Restarts>0)
#SBATCH --open-mode=append

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

SCRIPT_PATH="${REPO_ROOT}/playground/optimize_junction_tcpc_nm_resume.py"

cd "${REPO_ROOT}"

echo "Launching TCPC Nelder–Mead resume optimisation"
echo "Restart count: ${SLURM_RESTART_COUNT:-0} at $(date --iso-8601=seconds)"
# Unbuffered Python so progress appears promptly in Slurm logs
python3 -u "${SCRIPT_PATH}"
