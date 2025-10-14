#!/bin/bash
# Simple launcher for the Cassini optimisation script.
# Usage: ./demo_cassini.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/demo_cassini.py"
