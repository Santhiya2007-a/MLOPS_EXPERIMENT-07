#!/bin/bash
# =============================================================================
# Airflow DB Initialization — run ONCE before first use
# Run: bash airflow_init.sh
# =============================================================================
SCRIPT_DIR="/mnt/c/Users/user/OneDrive/Desktop/MLOPS 7 EXP/mlops-retraining-pipeline"
source "$SCRIPT_DIR/airflow_env.sh"
source /home/mlops/mlops-venv/bin/activate

echo ""
echo "=================================================="
echo "  Airflow Initialization"
echo "=================================================="

echo "[1/4] Ensuring log directories exist..."
mkdir -p "$AIRFLOW_HOME/logs/scheduler"
mkdir -p "$AIRFLOW_HOME/dags"

echo "[2/4] Running airflow db migrate..."
airflow db migrate

echo ""
echo "[3/4] Creating admin user (skip if already exists)..."
airflow users create \
  --username admin \
  --firstname MLOps \
  --lastname Admin \
  --role Admin \
  --email admin@mlops.local \
  --password adminpass 2>/dev/null || echo "  → User 'admin' already exists."

echo ""
echo "[4/4] Verifying DAG is detected..."
airflow dags list 2>/dev/null | grep -i automated || echo "  WARNING: DAG not found yet — scheduler will pick it up on start."

echo ""
echo "=================================================="
echo "  Done! Now run in TWO separate terminals:"
echo "  Terminal 1 → bash airflow_scheduler.sh"
echo "  Terminal 2 → bash airflow_webserver.sh"
echo "  Then open   → http://localhost:8080"
echo "  Login        → admin / adminpass"
echo "=================================================="
