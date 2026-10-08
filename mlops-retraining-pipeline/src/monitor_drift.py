from pathlib import Path
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def evaluate_drift(
    model_path="models/champion_model.pkl",
    data_path="data/incoming_batch.csv",
    threshold=0.80
):
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
