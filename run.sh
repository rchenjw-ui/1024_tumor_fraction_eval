#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

if [[ "${1:-}" == "--train" ]]; then
    echo "Training logistic-regression models..."
    "$PYTHON" "$SCRIPT_DIR/logistic_regression_test.py"
    shift
fi

if [[ "$#" -gt 0 ]]; then
    echo "Usage: $0 [--train]" >&2
    exit 2
fi

echo "Evaluating saved methylation models..."
"$PYTHON" "$SCRIPT_DIR/eval_given_saved.py"

echo "Comparing CNA and methylation scores..."
"$PYTHON" "$SCRIPT_DIR/cnv_tfx_estimator.py"

echo "Done. CSV outputs are in $SCRIPT_DIR/results/"
