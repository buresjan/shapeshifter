#!/bin/bash
# Launcher for the junction TCPC geometry + solver check.
# Usage (Slurm):   sbatch playground/run_junction_tcpc.sh
# Usage (local):   ./playground/run_junction_tcpc.sh --local [python args...]

#SBATCH --job-name=tcpc-junction
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=tcpc-junction-%j.out

set -euo pipefail

LOCAL_RUN=0
if [[ "${1:-}" == "--local" ]]; then
    LOCAL_RUN=1
    shift
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(realpath "${SLURM_SUBMIT_DIR}")"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if (( LOCAL_RUN )); then
        REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    else
        # Direct invocation without --local, still treat as repository run.
        REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
fi

SCRIPT_PATH="${REPO_ROOT}/playground/run_junction_tcpc.py"

cd "${REPO_ROOT}"

echo "Launching junction TCPC pipeline"
python3 "${SCRIPT_PATH}" "$@"
