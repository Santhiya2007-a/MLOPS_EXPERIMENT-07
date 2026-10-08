#!/bin/bash
# =============================================================================
# Trigger DAG manually — run in Terminal 3
# Usage:
#   bash airflow_trigger.sh              → Case 1: stable model test
#   bash airflow_trigger.sh degraded     → Case 2: inject drift then trigger
# =============================================================================
SCRIPT_DIR="/mnt/c/Users/user/OneDrive/Desktop/MLOPS 7 EXP/mlops-retraining-pipeline"
source "$SCRIPT_DIR/airflow_env.sh"
source /home/mlops/mlops-venv/bin/activate

DAG_ID="automated_model_retraining_pipeline"

echo ""
if [ "$1" == "degraded" ]; then
    echo "=================================================="
    echo "  CASE 2: Injecting data drift..."
    echo "=================================================="
    python "$SCRIPT_DIR/src/degrade_performance.py"
    echo ""
    echo "Verifying drift..."
    python "$SCRIPT_DIR/src/monitor_drift.py"
    echo ""
else
    echo "=================================================="
    echo "  CASE 1: Checking stable model..."
    echo "=================================================="
    python "$SCRIPT_DIR/src/monitor_drift.py"
    echo ""
fi

echo "Triggering DAG: $DAG_ID"
airflow dags trigger "$DAG_ID"

echo ""
echo "Open Airflow UI:"
echo "  http://localhost:8080  →  DAGs  →  $DAG_ID  →  Graph"
