"""
Utility to generate initial synthetic classification dataset for the MLOps pipeline.
"""
from pathlib import Path
import pandas as pd
from sklearn.datasets import make_classification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def generate_initial_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate 1000 samples with 6 informative features and binary target
    X, y = make_classification(
        n_samples=1000,
        n_features=6,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        flip_y=0.01,
        random_state=42
    )
    
    feature_cols = [f"feature_{i+1}" for i in range(6)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y
    
    # Save production data
    prod_path = DATA_DIR / "production_data.csv"
    df.to_csv(prod_path, index=False)
    print(f"Generated production dataset at {prod_path} (Shape: {df.shape})")
    
    # Create incoming_batch.csv as an initial clean copy (representing stable operations)
    batch_path = DATA_DIR / "incoming_batch.csv"
    df.to_csv(batch_path, index=False)
    print(f"Initialized incoming batch at {batch_path} (Shape: {df.shape})")

if __name__ == "__main__":
    generate_initial_data()
