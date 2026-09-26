import pytest
import pandas as pd
import numpy as np
from data.generate_dataset import generate_synthetic_dataset
from models.preprocessing import ClinicalDataPreprocessor
from analysis.run_analysis import execute_full_pipeline

def test_dataset_generation():
    df = generate_synthetic_dataset(n_samples=200, seed=42)
    assert len(df) == 200
    assert "patient_id" in df.columns
    assert "creatinine" in df.columns
    assert "diagnosis_category" in df.columns

def test_preprocessor_pipeline():
    df = generate_synthetic_dataset(n_samples=150, seed=42)
    preprocessor = ClinicalDataPreprocessor()
    df_clean, df_eng, X_scaled = preprocessor.fit_transform(df)

    assert len(df_clean) > 0
    assert X_scaled.shape[0] == len(df_clean)
    assert not np.isnan(X_scaled).any()
    assert preprocessor.is_fitted is True

def test_arbitrary_custom_csv_preprocessing():
    df_custom = pd.DataFrame({
        "subject_code": [f"SUBJ-{i}" for i in range(50)],
        "body_temp_c": np.random.normal(36.6, 0.2, 50),
        "lab_val_alpha": np.random.normal(100, 10, 50),
        "lab_val_beta": np.random.normal(50, 5, 50),
        "status_code": np.random.choice(["TypeA", "TypeB"], size=50)
    })
    
    df_custom.loc[49, "body_temp_c"] = 41.5
    df_custom.loc[49, "lab_val_alpha"] = 650.0

    res = execute_full_pipeline(dataset_df=df_custom)
    df_analyzed = res["df_analyzed"]

    assert len(df_analyzed) == 50
    assert "is_anomaly" in df_analyzed.columns
    assert df_analyzed.loc[df_analyzed["patient_id"] == "SUBJ-49", "is_anomaly"].values[0] == 1
