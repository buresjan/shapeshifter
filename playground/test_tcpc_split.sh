#!/bin/bash
# Slurm helper to re-run the TCPC split post-processing for an existing solver run.
#
# Usage:
#   sbatch playground/test_tcpc_split.sh /path/to/runs_tcpc/<run_dir> [case_stem] [label]
#
# The case stem defaults to the run directory name without the timestamp suffix.
# The optional label customises the tcpc_split output folder suffix.

#SBATCH --job-name=tcpc-split-test
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=tcpc-split-test-%j.out
#SBATCH --open-mode=append

set -euo pipefail

# Mirror the optimisation launchers: resolve the repo relative to the submit dir.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(realpath "${SLURM_SUBMIT_DIR}")"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
DEFAULT_RUN_DIR="${REPO_ROOT}/submodules/tnl-lbm/runs_tcpc/run_33133a2e_junction_dcde89c39477_20251117-080134"
DEFAULT_CASE_STEM="run_33133a2e_junction_dcde89c39477"
DEFAULT_LABEL="wrapper_test"

RUN_DIR="${1:-${DEFAULT_RUN_DIR}}"
CASE_STEM="${2:-${DEFAULT_CASE_STEM}}"
OUTPUT_LABEL="${3:-${DEFAULT_LABEL}}"

TEST_SCRIPT="${REPO_ROOT}/playground/test_tcpc_split.py"

CMD_ARGS=(--run-dir "${RUN_DIR}" --output-label "${OUTPUT_LABEL}")
if [[ -n "${CASE_STEM}" ]]; then
    CMD_ARGS+=(--case-stem "${CASE_STEM}")
fi

echo "Run dir:    ${RUN_DIR}"
echo "Case stem:  ${CASE_STEM:-<auto>}"
echo "Label:      ${OUTPUT_LABEL}"
echo "Helper:     ${TEST_SCRIPT}"

cd "${REPO_ROOT}"
python3 "${TEST_SCRIPT}" "${CMD_ARGS[@]}"
