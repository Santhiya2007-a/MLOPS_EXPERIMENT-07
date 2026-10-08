"""
Automated Machine Learning Retraining Pipeline
=================================================
DAG ID : automated_model_retraining_pipeline

Workflow:
    evaluate_production_drift
          |
     Accuracy < 0.80?
     /              \\
   YES               NO
    |                 |
trigger_retraining  skip_pipeline_execution
    |
promote_staging_to_production
    |
dispatch_slack_alert
"""

import logging
import pickle
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root so the DAG works on Windows / macOS / Linux regardless
# of where Airflow is installed.
#
# DAG file is at:  <project_root>/airflow/dags/automated_retraining_dag.py
# Project root is: <project_root>/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR  = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
SRC_DIR   = PROJECT_ROOT / "src"

# Ensure src/ is importable (needed when Airflow workers run from a different cwd)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from sklearn.metrics import accuracy_score

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.providers.http.operators.http import SimpleHttpOperator

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACCURACY_THRESHOLD   = 0.80
CHAMPION_MODEL_PATH  = MODEL_DIR / "champion_model.pkl"
STAGING_MODEL_PATH   = MODEL_DIR / "staging_model.pkl"
PRODUCTION_DATA_PATH = DATA_DIR  / "production_data.csv"
INCOMING_BATCH_PATH  = DATA_DIR  / "incoming_batch.csv"

# ---------------------------------------------------------------------------
# Default DAG arguments
# ---------------------------------------------------------------------------
default_args = {
    "owner"           : "mlops-team",
    "depends_on_past" : False,
    "email_on_failure": False,   # set True + configure SMTP to receive real emails
    "email_on_retry"  : False,
    "retries"         : 1,
    "retry_delay"     : timedelta(minutes=2),
}

# ===========================================================================
# Task callables
# ===========================================================================

def _evaluate_production_drift(**context):
    """
    Load the champion model and evaluate its accuracy on the incoming batch.
    Pushes the accuracy score to XCom so downstream tasks can use it.
    Returns the branch task ID for BranchPythonOperator.
    """
    if not CHAMPION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Champion model not found at {CHAMPION_MODEL_PATH}. "
            "Run: py -3.13 src/train_model.py  to generate it first."
        )

    if not INCOMING_BATCH_PATH.exists():
        raise FileNotFoundError(
            f"Incoming batch not found at {INCOMING_BATCH_PATH}."
        )

    with open(CHAMPION_MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    df           = pd.read_csv(INCOMING_BATCH_PATH)
    X            = df.drop(columns=["target"])
    y            = df["target"]
    predictions  = model.predict(X)
    current_acc  = accuracy_score(y, predictions)

    log.info("Operational production accuracy calculated at: %.4f", current_acc)
    print(f"Operational production accuracy calculated at: {current_acc:.4f}")

    # Push accuracy to XCom for reference in downstream tasks
    context["ti"].xcom_push(key="current_accuracy", value=round(current_acc, 4))

    if current_acc < ACCURACY_THRESHOLD:
        log.warning(
            "Accuracy %.4f is BELOW threshold %.2f — triggering retraining.",
            current_acc, ACCURACY_THRESHOLD
        )
        print("Retraining required.")
        return "trigger_retraining_sequence"
    else:
        log.info(
            "Accuracy %.4f is ABOVE threshold %.2f — pipeline stable.",
            current_acc, ACCURACY_THRESHOLD
        )
        print("System operation stable. Retraining not required.")
        return "skip_pipeline_execution"


def _trigger_retraining_sequence(**context):
    """
    Retrain a new RandomForest model on production data.
    Saves the result as staging_model.pkl (candidate model).
    """
    # Import here so Airflow workers pick it up via sys.path insertion above
    from train_model import execute_training

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Starting retraining on: %s", PRODUCTION_DATA_PATH)
    staging_acc = execute_training(
        data_path=str(PRODUCTION_DATA_PATH),
        output_model_path=str(STAGING_MODEL_PATH),
    )

    context["ti"].xcom_push(key="staging_accuracy", value=round(staging_acc, 4))
    log.info("Staging model saved at %s  (accuracy: %.4f)", STAGING_MODEL_PATH, staging_acc)


def _promote_staging_to_production(**context):
    """
    Promote staging_model.pkl → champion_model.pkl.
    The previous champion is overwritten after successful evaluation.
    """
    if not STAGING_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Staging model not found at {STAGING_MODEL_PATH}. "
            "Retraining step may have failed."
        )

    shutil.copy2(str(STAGING_MODEL_PATH), str(CHAMPION_MODEL_PATH))

    staging_acc  = context["ti"].xcom_pull(task_ids="trigger_retraining_sequence", key="staging_accuracy")
    champion_acc = context["ti"].xcom_pull(task_ids="evaluate_production_drift",   key="current_accuracy")

    log.info(
        "Model promotion complete. "
        "Previous champion accuracy: %.4f → New champion accuracy: %.4f",
        champion_acc, staging_acc
    )
    print(
        f"[PROMOTION] staging_model.pkl → champion_model.pkl\n"
        f"  Previous accuracy : {champion_acc}\n"
        f"  New accuracy      : {staging_acc}"
    )


def _skip_pipeline_execution(**context):
    """No-op task executed when the model accuracy is acceptable."""
    acc = context["ti"].xcom_pull(task_ids="evaluate_production_drift", key="current_accuracy")
    log.info("Pipeline skipped. Current production accuracy: %.4f", acc)
    print(f"[SKIP] Production model is healthy (accuracy={acc}). No retraining needed.")


def _dispatch_slack_alert(**context):
    """
    Send a Slack notification via webhook (or log the message if no webhook is configured).
    Uses python-dotenv to read SLACK_WEBHOOK_URL from .env
    """
    import os
    import json
    import requests
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    staging_acc  = context["ti"].xcom_pull(task_ids="trigger_retraining_sequence", key="staging_accuracy")
    champion_acc = context["ti"].xcom_pull(task_ids="evaluate_production_drift",   key="current_accuracy")

    message = (
        f":robot_face: *MLOps Retraining Alert*\n"
        f">  *Pipeline*: automated_model_retraining_pipeline\n"
        f">  *Drift detected* — previous accuracy: `{champion_acc}`\n"
        f">  *Retrained model* accuracy: `{staging_acc}`\n"
        f">  *Status*: `champion_model.pkl` has been updated successfully."
    )

    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")

    # Only send a real HTTP request if a real webhook URL is configured
    if webhook_url and not webhook_url.endswith("/X00"):
        try:
            response = requests.post(
                webhook_url,
                data=json.dumps({"text": message}),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            log.info("Slack notification dispatched. HTTP status: %s", response.status_code)
            print(f"[SLACK] Notification sent. Status: {response.status_code}")
        except Exception as exc:
            log.warning("Slack notification failed (non-critical): %s", exc)
            print(f"[SLACK] Notification failed (non-critical): {exc}")
    else:
        # Graceful fallback: just log the alert message
        log.info("[SLACK SIMULATION] Webhook not configured. Alert message:\n%s", message)
        print(f"\n[SLACK SIMULATION] Alert would be sent:\n{message}\n")


# ===========================================================================
# DAG definition
# ===========================================================================
with DAG(
    dag_id="automated_model_retraining_pipeline",
    description="Monitor production model drift and conditionally retrain + promote.",
    default_args=default_args,
    schedule="@daily",             # Airflow 2.4+ preferred syntax
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "retraining", "monitoring"],
) as dag:

    # ------------------------------------------------------------------
    # Task 1: Evaluate current champion model on incoming batch
    # ------------------------------------------------------------------
    evaluate_production_drift = BranchPythonOperator(
        task_id="evaluate_production_drift",
        python_callable=_evaluate_production_drift,
    )

    # ------------------------------------------------------------------
    # Task 2a (RETRAINING BRANCH): Train a new staging model
    # ------------------------------------------------------------------
    trigger_retraining_sequence = PythonOperator(
        task_id="trigger_retraining_sequence",
        python_callable=_trigger_retraining_sequence,
    )

    # ------------------------------------------------------------------
    # Task 3: Promote staging → champion
    # ------------------------------------------------------------------
    promote_staging_to_production = PythonOperator(
        task_id="promote_staging_to_production",
        python_callable=_promote_staging_to_production,
    )

    # ------------------------------------------------------------------
    # Task 4: Dispatch Slack alert
    # ------------------------------------------------------------------
    dispatch_slack_alert = PythonOperator(
        task_id="dispatch_slack_alert",
        python_callable=_dispatch_slack_alert,
    )

    # ------------------------------------------------------------------
    # Task 2b (STABLE BRANCH): Skip retraining — model is healthy
    # ------------------------------------------------------------------
    skip_pipeline_execution = PythonOperator(
        task_id="skip_pipeline_execution",
        python_callable=_skip_pipeline_execution,
    )

    # ==================================================================
    # DAG edges (dependencies)
    # ==================================================================
    #
    #  evaluate_production_drift
    #       |               |
    #  (drift)          (stable)
    #       |               |
    #  trigger_retraining  skip_pipeline_execution
    #       |
    #  promote_staging_to_production
    #       |
    #  dispatch_slack_alert
    #
    evaluate_production_drift >> [trigger_retraining_sequence, skip_pipeline_execution]
    trigger_retraining_sequence >> promote_staging_to_production >> dispatch_slack_alert
