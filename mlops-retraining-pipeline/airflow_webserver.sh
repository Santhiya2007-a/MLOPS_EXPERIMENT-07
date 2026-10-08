#!/bin/bash
# =============================================================================
# Airflow Webserver — run in Terminal 2
# Run: bash airflow_webserver.sh
# Open: http://localhost:8080   (admin / adminpass)
# =============================================================================
SCRIPT_DIR="/mnt/c/Users/user/OneDrive/Desktop/MLOPS 7 EXP/mlops-retraining-pipeline"
source "$SCRIPT_DIR/airflow_env.sh"
source /home/mlops/mlops-venv/bin/activate

echo ""
echo "=================================================="
echo "  Airflow Webserver"
echo "  URL   : http://localhost:8080"
echo "  Login : admin / adminpass"
echo "  Press Ctrl+C to stop."
echo "=================================================="
echo ""

airflow webserver --port 8080
