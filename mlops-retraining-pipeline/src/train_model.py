import os
from pathlib import Path
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def execute_training(data_path, output_model_path):
    # Resolve paths reliably across Windows and relative executions
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
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"Retrained Model Production Accuracy evaluated: {acc:.4f}")

    with open(output_model_resolved, 'wb') as f:
        pickle.dump(model, f)

    return acc


if __name__ == "__main__":
    execute_training(
        "data/production_data.csv",
        "models/champion_model.pkl"
    )
