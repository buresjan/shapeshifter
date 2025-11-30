#!/bin/bash
# Submit Slurm jobs for the follow-up 10 TCPC extra points.
#
# Usage: sbatch playground/collect_extra_points_round2.sh
# Optional env overrides:
#   MASTER_CSV (default: tmp/extra_points/extra_points_objectives_round2.csv)
#   ALGORITHM_LABEL (default: extra_points_round2)
#   TCPC_RUN_TAG (default: extra_points_round2)

#SBATCH --job-name=tcpc-extra2-submit
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=tcpc-extra2-submit-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(realpath "${SLURM_SUBMIT_DIR}")"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

RUNNER="${REPO_ROOT}/playground/run_extra_tcpc_point.sh"
if [[ ! -x "${RUNNER}" ]]; then
    echo "error: runner script ${RUNNER} is missing or not executable" >&2
    exit 3
fi

MASTER_CSV="${MASTER_CSV:-${REPO_ROOT}/tmp/extra_points/extra_points_objectives_round2.csv}"
ALGORITHM_LABEL="${ALGORITHM_LABEL:-extra_points_round2}"
RUN_TAG="${TCPC_RUN_TAG:-extra_points_round2}"

mkdir -p "$(dirname "${MASTER_CSV}")"

POINTS=(
    "offset,minus,-0.00355224,-1.78876,6.15250,0.00196229,0.00142384"
    "offset,plus,-0.00315224,-1.78876,6.15250,0.00196229,0.00142384"
    "lower_angle,minus,-0.00335224,-2.08876,6.15250,0.00196229,0.00142384"
    "lower_angle,plus,-0.00335224,-1.48876,6.15250,0.00196229,0.00142384"
    "upper_angle,minus,-0.00335224,-1.78876,5.85250,0.00196229,0.00142384"
    "upper_angle,plus,-0.00335224,-1.78876,6.45250,0.00196229,0.00142384"
    "lower_flare,minus,-0.00335224,-1.78876,6.15250,0.00193729,0.00142384"
    "lower_flare,plus,-0.00335224,-1.78876,6.15250,0.00198729,0.00142384"
    "upper_flare,minus,-0.00335224,-1.78876,6.15250,0.00196229,0.00139884"
    "upper_flare,plus,-0.00335224,-1.78876,6.15250,0.00196229,0.00144884"
)

echo "Submitting follow-up extra-point evaluations (results -> ${MASTER_CSV})"
echo "Algorithm label: ${ALGORITHM_LABEL}, run tag: ${RUN_TAG}"

for entry in "${POINTS[@]}"; do
    IFS=',' read -r direction sign offset lower_angle upper_angle lower_flare upper_flare <<< "${entry}"
    case_label="${direction}_${sign}_r2"
    submission_id=$(
        DIRECTION="${direction}" \
        SIGN="${sign}" \
        OFFSET="${offset}" \
        LOWER_ANGLE="${lower_angle}" \
        UPPER_ANGLE="${upper_angle}" \
        LOWER_FLARE="${lower_flare}" \
        UPPER_FLARE="${upper_flare}" \
        CASE_LABEL="${case_label}" \
        MASTER_CSV="${MASTER_CSV}" \
        ALGORITHM_LABEL="${ALGORITHM_LABEL}" \
        TCPC_RUN_TAG="${RUN_TAG}" \
        sbatch --parsable "${RUNNER}"
    )
    job_id="${submission_id%%;*}"
    echo "  - ${case_label}: job ${job_id}"
done
