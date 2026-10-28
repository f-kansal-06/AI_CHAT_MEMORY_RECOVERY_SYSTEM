#!/usr/bin/env bash
# End-to-end reproducible experiment execution script for E0-E7
set -e

echo "=== 1. Running pytest unit test suite ==="
export PYTHONPATH="."
python -m pytest -v tests/test_pipeline.py

echo "=== 2. Running Data Audits ==="
python scripts/audit_data.py

echo "=== 3. Running Master Experiment Suite (E0 - E7) ==="
python scripts/run_all_experiments.py

echo "=== 4. Generating Master Reports and Visualizations ==="
python scripts/make_report.py

echo "=== Pipeline execution finished successfully! ==="

