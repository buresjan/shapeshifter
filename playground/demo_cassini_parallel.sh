#!/bin/bash
# Launcher for the parallel Cassini optimisation script.
# Usage (local):   ./playground/demo_cassini_parallel.sh
# Usage (Slurm):   sbatch playground/demo_cassini_parallel.sh

#SBATCH --job-name=opt-cass-par
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=cass-par-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    # Slurm copies this script into /var/spool, so we jump back to the original repository.
    REPO_ROOT="$(realpath "${SLURM_SUBMIT_DIR}")"
    SCRIPT_PATH="${REPO_ROOT}/playground/demo_cassini_parallel.py"
    cd "${REPO_ROOT}"
else
    # Direct invocation keeps using the on-disk playground directory.
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SCRIPT_PATH="${SCRIPT_DIR}/demo_cassini_parallel.py"
    cd "${SCRIPT_DIR}/.."
fi

python3 "${SCRIPT_PATH}"

