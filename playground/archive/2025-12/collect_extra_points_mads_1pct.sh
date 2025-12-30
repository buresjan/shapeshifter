#!/bin/bash
# Submit Slurm jobs for the MADS (stress) extra points, logging to one CSV.
#
# Usage: sbatch playground/collect_extra_points_mads_1pct.sh
# Optional env overrides:
#   MASTER_CSV (default: tmp/extra_points/extra_points_mads_1pct.csv)
#   ALGORITHM_LABEL (default: extra_points_mads_1pct)
#   TCPC_RUN_TAG (default: extra_points_mads_1pct)
#
# Uses sim_tcpc_2 via run_extra_tcpc_point.sh (the stress objective path).

#SBATCH --job-name=tcpc-extra-mads-submit
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=tcpc-extra-mads-%j.out

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

MASTER_CSV="${MASTER_CSV:-${REPO_ROOT}/tmp/extra_points/extra_points_mads_1pct.csv}"
ALGORITHM_LABEL="${ALGORITHM_LABEL:-extra_points_mads_1pct}"
RUN_TAG="${TCPC_RUN_TAG:-extra_points_mads_1pct}"

mkdir -p "$(dirname "${MASTER_CSV}")"

POINTS=(
    # Central ±1% perturbations
    "mads_offset_minus_1pct,-0.001128,3.946,-2.98677,0.002499375,0.001241"
    "mads_offset_plus_1pct,-0.000728,3.946,-2.98677,0.002499375,0.001241"
    "mads_lower_angle_minus_1pct,-0.000928,3.646,-2.98677,0.002499375,0.001241"
    "mads_lower_angle_plus_1pct,-0.000928,4.246,-2.98677,0.002499375,0.001241"
    "mads_upper_angle_minus_1pct,-0.000928,3.946,-3.28677,0.002499375,0.001241"
    "mads_upper_angle_plus_1pct,-0.000928,3.946,-2.68677,0.002499375,0.001241"
    "mads_upper_flare_minus_1pct,-0.000928,3.946,-2.98677,0.002499375,0.001216"
    "mads_upper_flare_plus_1pct,-0.000928,3.946,-2.98677,0.002499375,0.001266"
    # One-sided lower_flare points (bound-active)
    "mads_lower_flare_minus_1pct,-0.000928,3.946,-2.98677,0.002474375,0.001241"
    "mads_lower_flare_minus_2pct,-0.000928,3.946,-2.98677,0.002449375,0.001241"
)

echo "Submitting MADS (stress) extra-point evaluations (results -> ${MASTER_CSV})"
echo "Algorithm label: ${ALGORITHM_LABEL}, run tag: ${RUN_TAG}"

for entry in "${POINTS[@]}"; do
    IFS=',' read -r case_label offset lower_angle upper_angle lower_flare upper_flare <<< "${entry}"
    submission_id=$(
        DIRECTION="${case_label}" \
        SIGN="1pct" \
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
