#!/bin/bash
# Submit Slurm jobs for +/- perturbations around a TCPC design point.
#
# The script is objective-agnostic: point it at the stress runner
# (run_extra_tcpc_point.sh) or the TKE runner (run_extra_tcpc_tke_point.sh)
# via the RUNNER variable.
#
# Required env:
#   BASE_POINT  - comma-separated values: offset,lower_angle,upper_angle,lower_flare,upper_flare
#   STEP_POINT  - comma-separated deltas for the same order (used for +/-)
#
# Optional env:
#   RUNNER          - runner script (default: playground/run_extra_tcpc_point.sh)
#   MASTER_CSV      - destination CSV (default: tmp/extra_points/${ALGORITHM_LABEL}_objectives.csv)
#   ALGORITHM_LABEL - forwarded to the evaluator (default: extra_points_generic)
#   TCPC_RUN_TAG    - run tag for geometry staging (default: ${ALGORITHM_LABEL})
#   CASE_SUFFIX     - appended to case labels (e.g., _r2)
#   DRY_RUN         - if set to 1, print the generated points and exit
#
# Example (MADS stress):
#   BASE_POINT="-0.00335,-1.79,6.15,0.00196,0.00142" \
#   STEP_POINT="0.0010,1.5,1.5,0.00012,0.00012" \
#   ALGORITHM_LABEL=mads_stress_extra \
#   sbatch playground/collect_extra_points_parametric.sh
#
# Example (NM TKE, using the TKE runner):
#   BASE_POINT="0.00010,4.8,-0.8,0.00195,0.00085" \
#   STEP_POINT="0.00020,0.6,0.6,0.00010,0.00008" \
#   RUNNER=playground/run_extra_tcpc_tke_point.sh \
#   ALGORITHM_LABEL=nm_tke_extra \
#   sbatch playground/collect_extra_points_parametric.sh

#SBATCH --job-name=tcpc-extra-submit-generic
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=tcpc-extra-generic-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(realpath "${SLURM_SUBMIT_DIR}")"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

RUNNER="${RUNNER:-${REPO_ROOT}/playground/run_extra_tcpc_point.sh}"
if [[ ! -x "${RUNNER}" ]]; then
    echo "error: runner script ${RUNNER} is missing or not executable" >&2
    exit 3
fi

if [[ -z "${BASE_POINT:-}" || -z "${STEP_POINT:-}" ]]; then
    echo "error: BASE_POINT and STEP_POINT must be set (comma-separated values for offset,lower_angle,upper_angle,lower_flare,upper_flare)" >&2
    exit 2
fi

ALGORITHM_LABEL="${ALGORITHM_LABEL:-extra_points_generic}"
RUN_TAG="${TCPC_RUN_TAG:-${ALGORITHM_LABEL}}"
MASTER_CSV="${MASTER_CSV:-${REPO_ROOT}/tmp/extra_points/${ALGORITHM_LABEL}_objectives.csv}"
CASE_SUFFIX="${CASE_SUFFIX:-}"

mkdir -p "$(dirname "${MASTER_CSV}")"

mapfile -t POINTS < <(BASE_POINT="${BASE_POINT}" STEP_POINT="${STEP_POINT}" python3 - <<'PY'
import os
import sys

names = ["offset", "lower_angle", "upper_angle", "lower_flare", "upper_flare"]
base_raw = os.environ.get("BASE_POINT", "")
step_raw = os.environ.get("STEP_POINT", "")

try:
    base = [float(x) for x in base_raw.split(",")]
    step = [float(x) for x in step_raw.split(",")]
except Exception:
    sys.stderr.write("error: BASE_POINT/STEP_POINT must contain numeric comma-separated values\n")
    sys.exit(1)

if len(base) != len(names) or len(step) != len(names):
    sys.stderr.write("error: BASE_POINT and STEP_POINT must each have 5 comma-separated values\n")
    sys.exit(1)

fmt = "{:.8f}"

for idx, name in enumerate(names):
    delta = step[idx]
    for sign, factor in (("minus", -1.0), ("plus", 1.0)):
        vals = list(base)
        vals[idx] = vals[idx] + factor * delta
        vals_str = ",".join(fmt.format(v) for v in vals)
        print(f"{name},{sign},{vals_str}")
PY
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Planned points (no submission due to DRY_RUN=1):"
    printf '  %s\n' "${POINTS[@]}"
    exit 0
fi

echo "Submitting extra-point evaluations via ${RUNNER}"
echo "Algorithm label: ${ALGORITHM_LABEL}, run tag: ${RUN_TAG}"
echo "Master CSV: ${MASTER_CSV}"

for entry in "${POINTS[@]}"; do
    IFS=',' read -r direction sign offset lower_angle upper_angle lower_flare upper_flare <<< "${entry}"
    case_label="${direction}_${sign}${CASE_SUFFIX}"
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
