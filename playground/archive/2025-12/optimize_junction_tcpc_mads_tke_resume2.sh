#!/bin/bash
# Slurm launcher for the TCPC MADS optimisation (TKE objective, resume/zoom #2).
#
# Usage (Slurm):   sbatch playground/optimize_junction_tcpc_mads_tke_resume2.sh
# Usage (local):   ./playground/optimize_junction_tcpc_mads_tke_resume2.sh --local
#
# The Python script has all parameters embedded; no CLI flags are used.

#SBATCH --job-name=mads-opt-tcpc-tke-resume2
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=opt-tcpc-tke-mads-resume2-%j.out
# Append to stdout/stderr instead of truncating on restart/requeue
# (prevents losing earlier logs when Restarts>0)
#SBATCH --open-mode=append

set -euo pipefail

LOCAL_RUN=0
if [[ "${1:-}" == "--local" ]]; then
    LOCAL_RUN=1
    shift
fi

# Stay under 16G per solver job and keep worker count aligned.
export OPT_MADS_WORKERS=${OPT_MADS_WORKERS:-4}
export TCPC_SLURM_MEM=${TCPC_SLURM_MEM:-16G}
export TCPC_SLURM_CPUS=${TCPC_SLURM_CPUS:-4}
export TCPC_SLURM_GPUS=${TCPC_SLURM_GPUS:-1}

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

SCRIPT_PATH="${REPO_ROOT}/playground/optimize_junction_tcpc_mads_tke_resume2.py"

cd "${REPO_ROOT}"

echo "Launching TCPC MADS optimisation (TKE objective, resume/zoom #2)"
echo "Restart count: ${SLURM_RESTART_COUNT:-0} at $(date --iso-8601=seconds)"
# Unbuffered Python so progress appears promptly in Slurm logs
python3 -u "${SCRIPT_PATH}"
