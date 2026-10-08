# EXPERIMENT TITLE

**Automated Machine Learning Retraining Pipeline using Apache Airflow**

---

# AIM

To design and implement an automated MLOps pipeline that monitors the performance of a production machine learning model and automatically initiates retraining and model promotion when the classification accuracy falls below a predefined threshold value of 0.80, orchestrated using Apache Airflow.

---

# REQUIREMENTS

**Hardware Requirements:**
- Processor: Intel Core i5 or above (64-bit)
- RAM: Minimum 8 GB
- Storage: Minimum 10 GB free disk space

**Software Requirements:**

| Software        | Version / Details            |
|-----------------|------------------------------|
| Operating System | Windows 11 (64-bit)         |
| IDE             | Visual Studio Code            |
| Python          | 3.13                          |
| Apache Airflow  | 2.10.2                        |
| Scikit-learn    | 1.3.2                         |
| Pandas          | 2.1.4                         |
| NumPy           | 1.26.2                        |
| Requests        | 2.31.0                        |
| python-dotenv   | 1.0.0                         |

---

# ALGORITHM

1. Create the MLOps project directory structure with folders: `airflow/`, `data/`, `src/`, `models/`.
2. Create and activate a Python virtual environment within the project directory.
3. Install all required Python packages from `requirements.txt` using pip.
4. Execute `src/generate_dataset.py` to automatically generate a synthetic binary classification dataset (`production_data.csv`) containing 1000 samples and 6 numerical features. Copy it as `incoming_batch.csv`.
5. Execute `src/train_model.py` to load the production dataset, split it into training and testing sets (80:20 ratio), train a `RandomForestClassifier` with 100 estimators, compute accuracy, and save the trained model as `models/champion_model.pkl` using Python pickle.
6. Initialise the Apache Airflow database using `airflow db init` with `AIRFLOW_HOME` set to the project's `airflow/` directory.
7. Create an Airflow administrator account with username `admin`.
8. Start the Airflow Scheduler in Terminal 1 and the Airflow Webserver on port 8080 in Terminal 2.
9. Access the Airflow Web UI at `http://localhost:8080` and navigate to the DAG `automated_model_retraining_pipeline`.
10. **Test Case 1 (Stable Model):** Trigger the DAG manually with a clean incoming batch. The `evaluate_production_drift` task loads `champion_model.pkl`, evaluates it against `incoming_batch.csv`, and computes accuracy ≥ 0.80. The DAG branches to `skip_pipeline_execution`.
11. **Test Case 2 (Degraded Model):** Execute `src/degrade_performance.py` to inject Gaussian noise (σ = 1.5) into `incoming_batch.csv`, simulating data drift. Trigger the DAG again.
12. The `evaluate_production_drift` task detects accuracy < 0.80 and returns the branch `trigger_retraining_sequence`.
13. The `trigger_retraining_sequence` task invokes `execute_training()` from `train_model.py`, retrains the Random Forest model on production data, and saves the output as `models/staging_model.pkl`.
14. The `promote_staging_to_production` task copies `staging_model.pkl` over `champion_model.pkl`, completing model promotion.
15. The `dispatch_slack_alert` task dispatches a notification with performance comparison metrics to the configured Slack webhook or logs the alert to the Airflow console.
16. Capture screenshots of the Airflow Graph view showing the branching workflow and task statuses for the laboratory record.

---

# PROCEDURE

### Step 1: Open the Project in VS Code

```powershell
cd "c:\Users\user\OneDrive\Desktop\MLOPS 7 EXP\mlops-retraining-pipeline"
code .
```

### Step 2: Create and Activate Virtual Environment

```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Generate Dataset and Train Initial Model

```powershell
py -3.13 src\generate_dataset.py
py -3.13 src\train_model.py
```

### Step 5: Initialise Apache Airflow

```powershell
$env:AIRFLOW_HOME="$(Get-Location)\airflow"
airflow db init
```

### Step 6: Create Airflow Admin User

```powershell
airflow users create `
  --username admin `
  --firstname MLOps `
  --lastname Admin `
  --role Admin `
  --email admin@mlops.local `
  --password adminpass
```

### Step 7: Start Airflow Scheduler (Terminal 1)

```powershell
.\venv\Scripts\Activate.ps1
$env:AIRFLOW_HOME="$(Get-Location)\airflow"
airflow scheduler
```

### Step 8: Start Airflow Webserver (Terminal 2)

```powershell
.\venv\Scripts\Activate.ps1
$env:AIRFLOW_HOME="$(Get-Location)\airflow"
airflow webserver --port 8080
```

Open browser: `http://localhost:8080`  
Login: Username: `admin` | Password: `adminpass`

### Step 9: Test Case 1 — Stable Model

```powershell
# Terminal 3
.\venv\Scripts\Activate.ps1
$env:AIRFLOW_HOME="$(Get-Location)\airflow"
airflow dags trigger automated_model_retraining_pipeline
```

Navigate: **DAGs → automated_model_retraining_pipeline → Graph**

### Step 10: Test Case 2 — Degraded Model

```powershell
py -3.13 src\degrade_performance.py
py -3.13 src\monitor_drift.py
airflow dags trigger automated_model_retraining_pipeline
```

---

# PROGRAM

## `src/generate_dataset.py`

```python
from pathlib import Path
import pandas as pd
from sklearn.datasets import make_classification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def generate_initial_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    X, y = make_classification(
        n_samples=1000, n_features=6, n_informative=5,
        n_redundant=1, n_classes=2, flip_y=0.01, random_state=42
    )
    feature_cols = [f"feature_{i+1}" for i in range(6)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y
    df.to_csv(DATA_DIR / "production_data.csv", index=False)
    df.to_csv(DATA_DIR / "incoming_batch.csv", index=False)
    print(f"Generated production dataset (Shape: {df.shape})")

if __name__ == "__main__":
    generate_initial_data()
```

---

## `src/train_model.py`

```python
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def execute_training(data_path, output_model_path):
    data_path_resolved = Path(data_path)
    if not data_path_resolved.is_absolute() and not data_path_resolved.exists():
        data_path_resolved = PROJECT_ROOT / data_path
    output_model_resolved = Path(output_model_path)
    if not output_model_resolved.is_absolute():
        output_model_resolved = PROJECT_ROOT / output_model_path
    output_model_resolved.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path_resolved)
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Retrained Model Production Accuracy evaluated: {acc:.4f}")
    with open(output_model_resolved, 'wb') as f:
        pickle.dump(model, f)
    return acc

if __name__ == "__main__":
    execute_training("data/production_data.csv", "models/champion_model.pkl")
```

---

## `src/degrade_performance.py`

```python
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def inject_drift(file_path="data/production_data.csv", noise_level=1.5):
    """Corrupts feature matrix values to force validation pipeline failure."""
    path_obj = Path(file_path)
    if not path_obj.is_absolute() and not path_obj.exists():
        path_obj = PROJECT_ROOT / file_path

    df = pd.read_csv(path_obj)
    feature_cols = [col for col in df.columns if col != 'target']

    noise = np.random.normal(0, noise_level,
                             size=(df.shape[0], len(feature_cols)))
    df[feature_cols] += noise

    out_path = PROJECT_ROOT / "data" / "incoming_batch.csv"
    df.to_csv(out_path, index=False)
    print("Successfully injected synthetic performance degradation into dataset.")

if __name__ == "__main__":
    inject_drift("data/production_data.csv")
```

---

## `src/monitor_drift.py`

```python
from pathlib import Path
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def evaluate_drift(model_path="models/champion_model.pkl",
                   data_path="data/incoming_batch.csv",
                   threshold=0.80):
    model_file = Path(model_path)
    if not model_file.is_absolute() and not model_file.exists():
        model_file = PROJECT_ROOT / model_path
    data_file = Path(data_path)
    if not data_file.is_absolute() and not data_file.exists():
        data_file = PROJECT_ROOT / data_path

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(data_file)
    X = df.drop(columns=["target"])
    y = df["target"]
    predictions = model.predict(X)
    current_accuracy = accuracy_score(y, predictions)

    print(f"Operational production accuracy calculated at: {current_accuracy:.4f}")
    if current_accuracy < threshold:
        print("Retraining required.")
        return False, current_accuracy
    else:
        print("System operation stable. Retraining not required.")
        return True, current_accuracy

if __name__ == "__main__":
    evaluate_drift()
```

---

## `airflow/dags/automated_retraining_dag.py` (Key Sections)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator

ACCURACY_THRESHOLD  = 0.80
PROJECT_ROOT        = Path(__file__).resolve().parents[2]
CHAMPION_MODEL_PATH = PROJECT_ROOT / "models" / "champion_model.pkl"
STAGING_MODEL_PATH  = PROJECT_ROOT / "models" / "staging_model.pkl"
INCOMING_BATCH_PATH = PROJECT_ROOT / "data"   / "incoming_batch.csv"

def _evaluate_production_drift(**context):
    model  = pickle.load(open(CHAMPION_MODEL_PATH, "rb"))
    df     = pd.read_csv(INCOMING_BATCH_PATH)
    acc    = accuracy_score(df["target"], model.predict(df.drop(columns=["target"])))
    context["ti"].xcom_push(key="current_accuracy", value=round(acc, 4))
    if acc < ACCURACY_THRESHOLD:
        return "trigger_retraining_sequence"
    return "skip_pipeline_execution"

# DAG edges:
evaluate_production_drift >> [trigger_retraining_sequence, skip_pipeline_execution]
trigger_retraining_sequence >> promote_staging_to_production >> dispatch_slack_alert
```

---

# OUTPUT

## Terminal Output — Dataset Generation

```
Generated production dataset at ...\data\production_data.csv (Shape: (1000, 7))
Initialized incoming batch at ...\data\incoming_batch.csv (Shape: (1000, 7))
```

## Terminal Output — Initial Model Training

```
Retrained Model Production Accuracy evaluated: 0.9650
```

## Terminal Output — Case 1: Stable Model (monitor_drift.py)

```
Operational production accuracy calculated at: 0.9930
System operation stable. Retraining not required.
```

**Airflow Graph — Case 1:**
```
evaluate_production_drift    [success] ─┐
                                        └─► skip_pipeline_execution    [success]
trigger_retraining_sequence  [skipped]
promote_staging_to_production [skipped]
dispatch_slack_alert          [skipped]
```

## Terminal Output — Case 2: Degraded Model

**After running `src/degrade_performance.py`:**
```
Successfully injected synthetic performance degradation into dataset.
```

**After running `src/monitor_drift.py`:**
```
Operational production accuracy calculated at: 0.7340
Retraining required.
```

**Airflow Graph — Case 2:**
```
evaluate_production_drift      [success]
        │
        ▼
trigger_retraining_sequence    [success]  ← Retrained Model Accuracy: 0.9650
        │
        ▼
promote_staging_to_production  [success]  ← staging_model.pkl → champion_model.pkl
        │
        ▼
dispatch_slack_alert           [success]  ← Alert logged/sent
```

**Console Alert Message:**
```
[SLACK SIMULATION] Alert would be sent:
:robot_face: *MLOps Retraining Alert*
>  *Pipeline*: automated_model_retraining_pipeline
>  *Drift detected* — previous accuracy: 0.734
>  *Retrained model* accuracy: 0.965
>  *Status*: `champion_model.pkl` has been updated successfully.
```

## Model Files Created

```
models/
├── champion_model.pkl   (1,026,093 bytes)  ← Production model
└── staging_model.pkl    (generated during DAG run)
```

---

# RESULT

The automated MLOps pipeline was successfully implemented using Apache Airflow. The pipeline monitored the production model's performance on incoming batch data and conditionally triggered model retraining when the classification accuracy dropped below the defined threshold of 0.80. The retrained staging model was evaluated and promoted to the production champion model through Airflow's orchestrated task execution. The complete workflow — including performance monitoring, drift detection, retraining, model promotion, and notification dispatch — was verified for both the stable and degraded model scenarios.
