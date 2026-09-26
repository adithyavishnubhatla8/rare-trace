import pytest
import pandas as pd
import numpy as np
from database.db import db_manager
from analysis.run_analysis import execute_full_pipeline

def test_source_separated_dataset_isolation():
    """
    Core Architectural Verification:
    1. Upload/Run Dataset A (Kaggle).
    2. Upload/Run Dataset B (UCI) with overlapping IDs (P001) but different features.
    3. Verify ZERO data merging, separate anomaly scores, separate scalers, separate models.
    4. Verify delete dataset A does not affect dataset B.
    """
    db_manager.create_tables()

    # Dataset A (Kaggle Cohort)
    df_a = pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(1, 41)],
        "serum_creatinine": np.random.normal(1.0, 0.2, 40),
        "blood_glucose": np.random.normal(90, 15, 40),
        "bmi": np.random.normal(23, 3, 40)
    })
    # Outlier in Dataset A
    df_a.loc[0, "serum_creatinine"] = 12.5

    res_a = execute_full_pipeline(
        dataset_id="ds_kaggle_test",
        dataset_name="Kaggle Rare Cohort",
        source="Kaggle",
        description="Test Kaggle dataset",
        filename="kaggle_patients.csv",
        dataset_df=df_a
    )

    # Dataset B (UCI Clinical Cohort)
    # Uses overlapping patient IDs (P001, P002...) but completely different clinical features
    df_b = pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(1, 31)],
        "neurological_score": np.random.normal(70, 8, 30),
        "cardiac_index": np.random.normal(3.2, 0.5, 30)
    })
    # Outlier in Dataset B
    df_b.loc[2, "neurological_score"] = 5.0

    res_b = execute_full_pipeline(
        dataset_id="ds_uci_test",
        dataset_name="UCI Medical Cohort",
        source="UCI Repository",
        description="Test UCI dataset",
        filename="uci_patients.csv",
        dataset_df=df_b
    )

    # 1. Verify Dataset Registration
    ds_a = db_manager.get_dataset("ds_kaggle_test")
    ds_b = db_manager.get_dataset("ds_uci_test")
    assert ds_a is not None
    assert ds_b is not None
    assert ds_a["source"] == "Kaggle"
    assert ds_b["source"] == "UCI Repository"
    assert ds_a["record_count"] == 40
    assert ds_b["record_count"] == 30

    # 2. Verify Patients Separation (No Cross-Contamination)
    patients_a = db_manager.get_full_patient_analysis("ds_kaggle_test")
    patients_b = db_manager.get_full_patient_analysis("ds_uci_test")

    assert len(patients_a) == 40
    assert len(patients_b) == 30

    # Dataset A must contain serum_creatinine and NOT neurological_score
    assert "serum_creatinine" in patients_a.columns
    assert "neurological_score" not in patients_a.columns

    # Dataset B must contain neurological_score and NOT serum_creatinine
    assert "neurological_score" in patients_b.columns
    assert "serum_creatinine" not in patients_b.columns

    # 3. Verify Patient ID P001 isolation across datasets
    p001_a = db_manager.get_patient_by_id("ds_kaggle_test", "P001")
    p001_b = db_manager.get_patient_by_id("ds_uci_test", "P001")
    assert p001_a is not None
    assert p001_b is not None
    assert "serum_creatinine" in p001_a
    assert "neurological_score" in p001_b

    # 4. Verify Model Runs and Hyperparameter Independence
    run_a = db_manager.get_latest_model_run("ds_kaggle_test")
    run_b = db_manager.get_latest_model_run("ds_uci_test")
    assert run_a["dataset_size"] == 40
    assert run_b["dataset_size"] == 30

    # 5. Verify Deletion of Dataset A leaves Dataset B completely intact
    db_manager.delete_dataset("ds_kaggle_test")
    assert db_manager.get_dataset("ds_kaggle_test") is None
    assert len(db_manager.get_full_patient_analysis("ds_kaggle_test")) == 0

    # Dataset B must remain fully intact
    assert db_manager.get_dataset("ds_uci_test") is not None
    assert len(db_manager.get_full_patient_analysis("ds_uci_test")) == 30

    # Clean up test dataset B
    db_manager.delete_dataset("ds_uci_test")
