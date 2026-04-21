#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "${SCRIPT_DIR}/run_full_experiments.sh" all
