#!/bin/bash
# Slurm wrapper that runs a single TCPC TKE extra-point evaluation.
#
# Required environment variables per job:
#   DIRECTION, SIGN, OFFSET, LOWER_ANGLE, UPPER_ANGLE, LOWER_FLARE, UPPER_FLARE
# Optional overrides:
#   MASTER_CSV (default: tmp/extra_points/extra_points_tke.csv)
#   CASE_LABEL, ALGORITHM_LABEL, TCPC_RUN_TAG
#
# Usage (Slurm): DIRECTION=... SIGN=... OFFSET=... LOWER_ANGLE=... UPPER_ANGLE=... \
#                LOWER_FLARE=... UPPER_FLARE=... sbatch playground/run_extra_tcpc_tke_point.sh
# Usage (local): DIRECTION=... SIGN=... OFFSET=... LOWER_ANGLE=... UPPER_ANGLE=... \
#                LOWER_FLARE=... UPPER_FLARE=... ./playground/run_extra_tcpc_tke_point.sh --local

#SBATCH --job-name=tcpc-extra-tke-point
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --output=tcpc-extra-tke-%j.out
# Append to stdout/stderr instead of truncating on restart/requeue
# (prevents losing earlier logs when Restarts>0)
#SBATCH --open-mode=append

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
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

SCRIPT_PATH="${REPO_ROOT}/playground/evaluate_tcpc_tke_extra_point.py"

for required in DIRECTION SIGN OFFSET LOWER_ANGLE UPPER_ANGLE LOWER_FLARE UPPER_FLARE; do
    if [[ -z "${!required:-}" ]]; then
        echo "error: environment variable ${required} is required" >&2
        exit 2
    fi
done

MASTER_CSV="${MASTER_CSV:-${REPO_ROOT}/tmp/extra_points/extra_points_tke.csv}"
ALGORITHM_LABEL="${ALGORITHM_LABEL:-extra_points_tke}"
CASE_LABEL="${CASE_LABEL:-${DIRECTION}_${SIGN}}"
# Default run tag keeps artefacts grouped; can be overridden per job if needed.
export TCPC_RUN_TAG="${TCPC_RUN_TAG:-extra_points_tke}"

cd "${REPO_ROOT}"

echo "Running TKE extra-point case ${CASE_LABEL}"
echo "Parameters: offset=${OFFSET}, lower_angle=${LOWER_ANGLE}, upper_angle=${UPPER_ANGLE}, lower_flare=${LOWER_FLARE}, upper_flare=${UPPER_FLARE}"
echo "Master CSV: ${MASTER_CSV}"

python3 -u "${SCRIPT_PATH}" \
    --direction "${DIRECTION}" \
    --sign "${SIGN}" \
    --offset "${OFFSET}" \
    --lower-angle "${LOWER_ANGLE}" \
    --upper-angle "${UPPER_ANGLE}" \
    --lower-flare "${LOWER_FLARE}" \
    --upper-flare "${UPPER_FLARE}" \
    --csv "${MASTER_CSV}" \
    --algorithm-label "${ALGORITHM_LABEL}" \
    --case-label "${CASE_LABEL}" \
    --job-name "${SLURM_JOB_NAME:-tcpc-extra-tke-point}" \
    --slurm-job-id "${SLURM_JOB_ID:-}"
