#!/bin/bash
# Submit Slurm jobs for the 10 requested extra TCPC points.
#
# Usage: sbatch playground/collect_extra_points.sh
# Optional env overrides:
#   MASTER_CSV (default: tmp/extra_points/extra_points_objectives.csv)
#   ALGORITHM_LABEL (default: extra_points)
#   TCPC_RUN_TAG (default: extra_points)

#SBATCH --job-name=tcpc-extra-submit
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=tcpc-extra-submit-%j.out

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

MASTER_CSV="${MASTER_CSV:-${REPO_ROOT}/tmp/extra_points/extra_points_objectives.csv}"
ALGORITHM_LABEL="${ALGORITHM_LABEL:-extra_points}"
RUN_TAG="${TCPC_RUN_TAG:-extra_points}"

mkdir -p "$(dirname "${MASTER_CSV}")"

POINTS=(
    "offset,plus,-0.002352,-1.78876,6.15250,0.00196229,0.00142384"
    "offset,minus,-0.004352,-1.78876,6.15250,0.00196229,0.00142384"
    "lower_angle,plus,-0.003352,-0.28876,6.15250,0.00196229,0.00142384"
    "lower_angle,minus,-0.003352,-3.28876,6.15250,0.00196229,0.00142384"
    "upper_angle,plus,-0.003352,-1.78876,7.65250,0.00196229,0.00142384"
    "upper_angle,minus,-0.003352,-1.78876,4.65250,0.00196229,0.00142384"
    "lower_flare,plus,-0.003352,-1.78876,6.15250,0.00208729,0.00142384"
    "lower_flare,minus,-0.003352,-1.78876,6.15250,0.00183729,0.00142384"
    "upper_flare,plus,-0.003352,-1.78876,6.15250,0.00196229,0.00154884"
    "upper_flare,minus,-0.003352,-1.78876,6.15250,0.00196229,0.00129884"
)

echo "Submitting extra-point evaluations (results -> ${MASTER_CSV})"
echo "Algorithm label: ${ALGORITHM_LABEL}, run tag: ${RUN_TAG}"

for entry in "${POINTS[@]}"; do
    IFS=',' read -r direction sign offset lower_angle upper_angle lower_flare upper_flare <<< "${entry}"
    case_label="${direction}_${sign}"
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
