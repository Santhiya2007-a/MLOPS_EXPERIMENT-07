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

    # Inject intense Gaussian noise to shift data distributions
    noise = np.random.normal(
        0,
        noise_level,
        size=(df.shape[0], len(feature_cols))
    )

    df[feature_cols] += noise

    out_path = Path("data/incoming_batch.csv")
    if not out_path.parent.exists():
        out_path = PROJECT_ROOT / "data" / "incoming_batch.csv"

    df.to_csv(out_path, index=False)

    print("Successfully injected synthetic performance degradation into dataset.")


if __name__ == "__main__":
    inject_drift("data/production_data.csv")
