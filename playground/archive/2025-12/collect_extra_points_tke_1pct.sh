#!/bin/bash
# Submit Slurm jobs for the TCPC TKE +/-1% extra points (relative to the NM-TKE best).
#
# Usage: sbatch playground/collect_extra_points_tke_1pct.sh
# Optional env overrides:
#   MASTER_CSV (default: tmp/extra_points/extra_points_tke_1pct.csv)
#   ALGORITHM_LABEL (default: extra_points_tke_1pct)
#   TCPC_RUN_TAG (default: extra_points_tke_1pct)
#
# Uses the TKE objective pipeline (sim_tcpc_tke) via run_extra_tcpc_tke_point.sh.

#SBATCH --job-name=tcpc-extra-tke1pct-submit
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=tcpc-extra-tke1pct-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(realpath "${SLURM_SUBMIT_DIR}")"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

RUNNER="${REPO_ROOT}/playground/run_extra_tcpc_tke_point.sh"
if [[ ! -x "${RUNNER}" ]]; then
    echo "error: runner script ${RUNNER} is missing or not executable" >&2
    exit 3
fi

MASTER_CSV="${MASTER_CSV:-${REPO_ROOT}/tmp/extra_points/extra_points_tke_1pct.csv}"
ALGORITHM_LABEL="${ALGORITHM_LABEL:-extra_points_tke_1pct}"
RUN_TAG="${TCPC_RUN_TAG:-extra_points_tke_1pct}"

mkdir -p "$(dirname "${MASTER_CSV}")"

POINTS=(
    "tke_offset_minus_1pct,-0.0008200056,4.0741258864,-0.8672445272,0.0022396408,0.0006426584"
    "tke_offset_plus_1pct,-0.0004200056,4.0741258864,-0.8672445272,0.0022396408,0.0006426584"
    "tke_lower_angle_minus_1pct,-0.0006200056,3.7741258864,-0.8672445272,0.0022396408,0.0006426584"
    "tke_lower_angle_plus_1pct,-0.0006200056,4.3741258864,-0.8672445272,0.0022396408,0.0006426584"
    "tke_upper_angle_minus_1pct,-0.0006200056,4.0741258864,-1.1672445272,0.0022396408,0.0006426584"
    "tke_upper_angle_plus_1pct,-0.0006200056,4.0741258864,-0.5672445272,0.0022396408,0.0006426584"
    "tke_lower_flare_minus_1pct,-0.0006200056,4.0741258864,-0.8672445272,0.0022146408,0.0006426584"
    "tke_lower_flare_plus_1pct,-0.0006200056,4.0741258864,-0.8672445272,0.0022646408,0.0006426584"
    "tke_upper_flare_minus_1pct,-0.0006200056,4.0741258864,-0.8672445272,0.0022396408,0.0006176584"
    "tke_upper_flare_plus_1pct,-0.0006200056,4.0741258864,-0.8672445272,0.0022396408,0.0006676584"
)

echo "Submitting TKE +/-1% extra-point evaluations (results -> ${MASTER_CSV})"
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
