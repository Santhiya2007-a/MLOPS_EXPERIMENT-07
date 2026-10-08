# Automated Machine Learning Retraining Pipeline using Apache Airflow

A production-ready MLOps experiment demonstrating conditional model retraining with Apache Airflow orchestration.

---

## Aim

To design and implement an automated MLOps pipeline that monitors production model performance and automatically retrains and promotes a new model when performance falls below a defined threshold.

---

## Requirements

| Component       | Version / Tool     |
|-----------------|--------------------|
| OS              | Windows 11         |
| IDE             | Visual Studio Code |
| Python          | 3.13               |
| Apache Airflow  | 2.10.2             |
| Pandas          | 2.1.4              |
| NumPy           | 1.26.2             |
| Scikit-learn    | 1.3.2              |
| Requests        | 2.31.0             |
| python-dotenv   | 1.0.0              |

---

## Project Structure

```
mlops-retraining-pipeline/
│
├── airflow/
│   ├── dags/
│   │   └── automated_retraining_dag.py   ← Main Airflow DAG
│   ├── logs/
│   └── plugins/
│
├── data/
│   ├── production_data.csv               ← Synthetic training dataset (1000 rows)
│   └── incoming_batch.csv                ← Live batch for monitoring
│
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py               ← Auto-generates datasets
│   ├── degrade_performance.py            ← Injects drift for testing
│   ├── monitor_drift.py                  ← Standalone drift evaluator
│   └── train_model.py                    ← RandomForest training logic
│
├── models/
│   ├── champion_model.pkl                ← Current production model
│   └── staging_model.pkl                 ← Candidate (retrained) model
│
├── .env                                  ← Environment config
├── .gitignore
└── requirements.txt
```

---

## Algorithm

1. Create the MLOps project directory structure.
2. Create and activate a Python virtual environment.
3. Install all required dependencies from `requirements.txt`.
4. Generate a synthetic classification dataset (`production_data.csv`, 1000 samples, 6 features).
5. Train an initial Random Forest model on production data.
6. Save the trained model as `models/champion_model.pkl`.
7. Create `data/incoming_batch.csv` as the live monitoring stream (initially clean copy).
8. Monitor production model accuracy against the incoming batch.
9. Compare accuracy against threshold `0.80`.
10. If accuracy **< 0.80** → trigger retraining sequence.
11. Train a new Random Forest model and save as `models/staging_model.pkl`.
12. Evaluate the staging model's accuracy.
13. Promote staging model: copy `staging_model.pkl` → `champion_model.pkl`.
14. Dispatch Slack/log notification with performance summary.
15. If accuracy **≥ 0.80** → skip retraining (model is healthy).
16. Visualise the complete conditional workflow in the Airflow Graph UI.

---

## Architecture

```
                 ┌─────────────────────────────┐
                 │     Production Model        │
                 │    (champion_model.pkl)      │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │  evaluate_production_drift  │
                 │  (load model + batch data,  │
                 │   compute accuracy)         │
                 └──────────────┬──────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
              Accuracy < 0.80         Accuracy ≥ 0.80
                    │                       │
                    ▼                       ▼
     ┌──────────────────────┐  ┌───────────────────────┐
     │ trigger_retraining_  │  │  skip_pipeline_       │
     │     sequence         │  │     execution         │
     │ (RandomForest on     │  │ (no-op, log message)  │
     │  production_data)    │  └───────────────────────┘
     └──────────┬───────────┘
                │
                ▼
     ┌──────────────────────┐
     │ promote_staging_to_  │
     │     production       │
     │ (staging → champion) │
     └──────────┬───────────┘
                │
                ▼
     ┌──────────────────────┐
     │  dispatch_slack_     │
     │       alert          │
     │ (Slack / log notify) │
     └──────────────────────┘
```

---

## Setup & Procedure

### Step 1 — Open Project in VS Code

```powershell
cd "c:\Users\user\OneDrive\Desktop\MLOPS 7 EXP\mlops-retraining-pipeline"
code .
```

### Step 2 — Create & Activate Virtual Environment

```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
```

> If you see a script execution policy error, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Step 3 — Install Dependencies

```powershell
pip install -r requirements.txt
```

> **Note:** Apache Airflow 2.10.2 requires Python ≤ 3.12 for full compatibility.  
> The project source scripts (`src/`) run on Python 3.13 with the system-installed
> packages (pandas, scikit-learn, numpy) and do not require the venv for standalone testing.

### Step 4 — Generate Dataset & Train Initial Model

```powershell
# Using system Python 3.13 (which has the required packages pre-installed)
py -3.13 src\generate_dataset.py
py -3.13 src\train_model.py
```

Expected output:
```
Generated production dataset at ...\data\production_data.csv (Shape: (1000, 7))
Initialized incoming batch at ...\data\incoming_batch.csv (Shape: (1000, 7))
Retrained Model Production Accuracy evaluated: 0.9650
```

### Step 5 — Configure Airflow

Open **Terminal 1** in VS Code:

```powershell
.\venv\Scripts\Activate.ps1
$env:AIRFLOW_HOME="$(Get-Location)\airflow"
airflow db init
```

Create admin user:
```powershell
airflow users create `
  --username admin `
  --firstname MLOps `
  --lastname Admin `
  --role Admin `
  --email admin@mlops.local `
  --password adminpass
```

### Step 6 — Start Airflow Services

**Terminal 1 — Scheduler:**
```powershell
.\venv\Scripts\Activate.ps1
$env:AIRFLOW_HOME="$(Get-Location)\airflow"
airflow scheduler
```

**Terminal 2 — Webserver:**
```powershell
.\venv\Scripts\Activate.ps1
$env:AIRFLOW_HOME="$(Get-Location)\airflow"
airflow webserver --port 8080
```

Open browser: **http://localhost:8080**  
Login: `admin` / `adminpass`

### Step 7 — Navigate to the DAG

```
Airflow UI → DAGs → automated_model_retraining_pipeline → Graph
```

---

## Testing the Pipeline

### CASE 1 — Stable Model (no retraining)

The default `incoming_batch.csv` is a clean copy of production data.

**Trigger DAG manually** (Terminal 3):
```powershell
.\venv\Scripts\Activate.ps1
$env:AIRFLOW_HOME="$(Get-Location)\airflow"
airflow dags trigger automated_model_retraining_pipeline
```

**Expected task results:**
```
evaluate_production_drift    → success (accuracy: 0.9930)
skip_pipeline_execution      → success
```

---

### CASE 2 — Degraded Model (retraining triggered)

**Step 1** — Inject performance drift:
```powershell
py -3.13 src\degrade_performance.py
```
```
Successfully injected synthetic performance degradation into dataset.
```

**Step 2** — Verify drift:
```powershell
py -3.13 src\monitor_drift.py
```
```
Operational production accuracy calculated at: 0.7340
Retraining required.
```

**Step 3** — Trigger DAG:
```powershell
airflow dags trigger automated_model_retraining_pipeline
```

**Expected task sequence (Airflow UI):**
```
evaluate_production_drift      → success  (accuracy < 0.80)
trigger_retraining_sequence    → success  (new staging model trained)
promote_staging_to_production  → success  (staging → champion)
dispatch_slack_alert           → success  (alert logged/sent)
```

**Step 4** — Restore clean batch after testing:
```powershell
py -3.13 -c "import pandas as pd; df=pd.read_csv('data/production_data.csv'); df.to_csv('data/incoming_batch.csv',index=False)"
```

---

## Model Artifacts

| File                    | Description                                      |
|-------------------------|--------------------------------------------------|
| `champion_model.pkl`    | Current production model serving live traffic    |
| `staging_model.pkl`     | Newly retrained candidate awaiting promotion     |

After promotion: `staging_model.pkl` → overwrites → `champion_model.pkl`

---

## Environment Variables (`.env`)

| Variable          | Description                           |
|-------------------|---------------------------------------|
| `AIRFLOW_HOME`    | Path to Airflow configuration folder |
| `SLACK_WEBHOOK_URL` | Slack Incoming Webhook URL (optional) |
| `ALERT_EMAIL`     | Notification email address (optional) |

---

## Result

The automated MLOps pipeline successfully monitored production model performance and conditionally triggered model retraining when the accuracy dropped below the defined threshold of 0.80. The retrained staging model was promoted to the production champion model through Apache Airflow orchestration.
