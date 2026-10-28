# PowerShell reproducible experiment execution script for E0-E7
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "."

Write-Host "=== 1. Running pytest unit test suite ===" -ForegroundColor Cyan
python -m pytest -v tests/test_pipeline.py

Write-Host "=== 2. Running Data Audits ===" -ForegroundColor Cyan
python scripts/audit_data.py

Write-Host "=== 3. Running Master Experiment Suite (E0 - E7) ===" -ForegroundColor Cyan
python scripts/run_all_experiments.py

Write-Host "=== 4. Generating Master Reports and Visualizations ===" -ForegroundColor Cyan
python scripts/make_report.py

Write-Host "=== Pipeline execution finished successfully! ===" -ForegroundColor Green

