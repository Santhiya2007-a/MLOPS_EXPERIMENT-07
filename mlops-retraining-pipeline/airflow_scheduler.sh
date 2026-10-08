#!/bin/bash
# =============================================================================
# Airflow Scheduler — run in Terminal 1
# Run: bash airflow_scheduler.sh
# =============================================================================
SCRIPT_DIR="/mnt/c/Users/user/OneDrive/Desktop/MLOPS 7 EXP/mlops-retraining-pipeline"
source "$SCRIPT_DIR/airflow_env.sh"
source /home/mlops/mlops-venv/bin/activate

echo ""
echo "=================================================="
echo "  Airflow Scheduler"
echo "  DAGS: $AIRFLOW__CORE__DAGS_FOLDER"
echo "  Press Ctrl+C to stop."
echo "=================================================="
echo ""

airflow scheduler
