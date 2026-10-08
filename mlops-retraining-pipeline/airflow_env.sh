#!/bin/bash
# =============================================================================
# airflow_env.sh  —  Source this to set all Airflow environment variables
# Usage: source airflow_env.sh
# All other scripts already source this automatically.
# =============================================================================

export PROJECT_ROOT="/mnt/c/Users/user/OneDrive/Desktop/MLOPS 7 EXP/mlops-retraining-pipeline"

# Airflow DB + HOME on Linux filesystem (avoids Windows path SQLite error)
export AIRFLOW_HOME="/home/mlops/airflow"
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="sqlite:////home/mlops/airflow/airflow.db"

# Point DAGs at the existing Windows project DAGs folder
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/airflow/dags"

# Hide built-in example DAGs (keep UI clean)
export AIRFLOW__CORE__LOAD_EXAMPLES="False"

# Logs on Linux filesystem for performance
export AIRFLOW__LOGGING__BASE_LOG_FOLDER="/home/mlops/airflow/logs"
export AIRFLOW__SCHEDULER__CHILD_PROCESS_LOG_DIRECTORY="/home/mlops/airflow/logs/scheduler"

# Make src/ importable inside DAG tasks
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

echo "Airflow environment set."
echo "  AIRFLOW_HOME      : $AIRFLOW_HOME"
echo "  DAGS_FOLDER       : $AIRFLOW__CORE__DAGS_FOLDER"
echo "  DB                : $AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"
