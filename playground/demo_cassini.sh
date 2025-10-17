#!/bin/bash
# Launcher for the Cassini optimisation script.
# Usage (Slurm):   sbatch playground/demo_cassini.sh
# Usage (local):   ./playground/demo_cassini.sh --local

#SBATCH --job-name=opt-cass
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=cass-%j.out

set -euo pipefail

EXECUTOR="slurm"
if [[ "${1:-}" == "--local" ]]; then
    EXECUTOR="local"
    shift
fi
export LBM_EXECUTOR="${EXECUTOR}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    # Slurm copies this script into /var/spool, so we jump back to the original repository.
    REPO_ROOT="$(realpath "${SLURM_SUBMIT_DIR}")"
    SCRIPT_PATH="${REPO_ROOT}/playground/demo_cassini.py"
    cd "${REPO_ROOT}"
else
    # Direct invocation keeps using the on-disk playground directory.
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SCRIPT_PATH="${SCRIPT_DIR}/demo_cassini.py"
    cd "${SCRIPT_DIR}/.."
fi

echo "Launching Cassini optimisation via ${EXECUTOR} executor"
python3 "${SCRIPT_PATH}" "$@"
