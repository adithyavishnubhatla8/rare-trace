import pytest
import pandas as pd
from database.db import DatabaseManager
from data.generate_dataset import generate_synthetic_dataset

def test_database_manager_crud():
    db = DatabaseManager()
    db.create_tables()

    df = generate_synthetic_dataset(n_samples=50, seed=42)
    db.save_patients(df)

    run_id = db.save_model_run({
        "algorithm": "DBSCAN", "eps": 1.2, "min_samples": 5,
        "number_of_clusters": 2, "number_of_anomalies": 5,
        "dataset_size": 50, "silhouette_score": 0.35
    })
    assert run_id >= 1

    df["cluster_label"] = 0
    df["is_anomaly"] = 0
    df["anomaly_score"] = 15.0
    df["review_priority"] = "Normal Profile"
    df["key_abnormal_features"] = "None"
    df["interpretation"] = "Normal Profile"

    db.save_analysis_results(df, run_id=run_id)

    df_fetched = db.get_full_patient_analysis()
    assert len(df_fetched) > 0
