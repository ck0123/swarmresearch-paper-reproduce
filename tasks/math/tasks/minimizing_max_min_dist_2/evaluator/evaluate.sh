#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$1"
PYTHONPATH="/workspace:/benchmark" python /benchmark/evaluator.py "$WORKSPACE/initial_program.py"
