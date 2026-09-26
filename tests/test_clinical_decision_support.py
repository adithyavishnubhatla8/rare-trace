"""
tests/test_clinical_decision_support.py
=============================================================================
Unit and Integration Tests for Clinical Decision Support & Medical Recommendations
=============================================================================
"""

import pytest
import numpy as np
import pandas as pd
from models.clinical_decision_support import ClinicalDecisionSupportEngine
from database.db import DatabaseManager
from app import app


@pytest.fixture
def cds_engine():
    return ClinicalDecisionSupportEngine()


@pytest.fixture
def mock_cohort():
    """Create a mock population cohort with defined means and standard deviations."""
    df = pd.DataFrame({
        "patient_id": [f"PAT-{1000+i}" for i in range(50)],
        "creatinine": np.random.normal(1.0, 0.2, 50),
        "systolic_bp": np.random.normal(120.0, 10.0, 50),
        "hemoglobin": np.random.normal(14.0, 1.5, 50),
        "glucose": np.random.normal(95.0, 10.0, 50),
        "cluster_label": [0]*48 + [-1, -1],
        "is_anomaly": [0]*48 + [1, 1],
        "anomaly_score": [15.0]*48 + [88.5, 92.0]
    })
    return df


def test_cds_domain_and_z_score_detection(cds_engine):
    """Verify that clinical features are accurately mapped to their physiological domains."""
    assert cds_engine.identify_feature_domain("serum_creatinine") == "RENAL"
    assert cds_engine.identify_feature_domain("systolic_bp") == "CARDIOVASCULAR"
    assert cds_engine.identify_feature_domain("hemoglobin") == "HEMATOLOGICAL"
    assert cds_engine.identify_feature_domain("fasting_glucose") == "METABOLIC"
    assert cds_engine.identify_feature_domain("oxygen_saturation") == "RESPIRATORY"
    assert cds_engine.identify_feature_domain("custom_unclassified_metric") == "GENERAL"


def test_cds_multi_system_and_priority(cds_engine, mock_cohort):
    """Verify that multi-system signals and clinical priority are appropriately assigned."""
    pop_means = mock_cohort.mean(numeric_only=True)
    pop_stds = mock_cohort.std(numeric_only=True)

    # Multi-system outlier patient (Renal + Hematological + Cardiovascular)
    patient_multi = {
        "patient_id": "TEST-MULTI",
        "creatinine": 3.8,      # Highly elevated (Renal)
        "systolic_bp": 185.0,   # Highly elevated (Cardiovascular)
        "hemoglobin": 6.5,      # Highly depressed (Hematological)
        "glucose": 95.0,        # Normal
        "cluster_label": -1,
        "is_anomaly": 1,
        "anomaly_score": 94.0
    }

    suggestion = cds_engine.analyze_patient(patient_multi, pop_means=pop_means, pop_stds=pop_stds)

    assert suggestion["multi_system_signal"] == 1
    assert suggestion["clinical_priority"] == "HIGH"
    assert "RENAL" in suggestion["all_domains"]
    assert "CARDIOVASCULAR" in suggestion["all_domains"]
    assert "HEMATOLOGICAL" in suggestion["all_domains"]
    assert len(suggestion["z_score_drivers"]) >= 3


def test_cds_medical_safety_and_non_prescriptive_phrasing(cds_engine, mock_cohort):
    """Verify that the engine strictly adheres to medical safety rules (no prescriptions or surgery mandates)."""
    pop_means = mock_cohort.mean(numeric_only=True)
    pop_stds = mock_cohort.std(numeric_only=True)

    patient_case = {
        "patient_id": "TEST-SAFETY",
        "creatinine": 4.5,
        "systolic_bp": 190.0,
        "cluster_label": -1,
        "is_anomaly": 1,
        "anomaly_score": 96.0
    }

    suggestion = cds_engine.analyze_patient(patient_case, pop_means=pop_means, pop_stds=pop_stds)

    # Mandatory disclaimer must be present
    assert "not a medical diagnosis or prescription" in suggestion["disclaimer"].lower()
    assert "qualified healthcare professional" in suggestion["disclaimer"].lower()

    # All investigation suggestions must use 'Consider' phrasing
    for inv in suggestion["investigation_suggestions"]:
        assert inv.startswith("Consider")

    # Medication notes must state clinician review, not prescription
    med_text = suggestion["medication_review_notes"].lower()
    assert "prescribe" not in med_text
    assert "take" not in med_text
    assert "clinician should review" in med_text

    # Procedure notes must never mandate surgery autonomously
    proc_text = suggestion["procedure_review_notes"].lower()
    assert "requires surgery" not in proc_text
    assert "no automatic surgical" in proc_text


def test_cds_dataset_and_patient_isolation():
    """Verify that clinical suggestions are strictly isolated between datasets with zero cross-contamination."""
    db = DatabaseManager()
    db.create_tables()

    ds_alpha = "test_cds_alpha"
    ds_beta = "test_cds_beta"

    # Register both datasets
    db.save_dataset_metadata({"dataset_id": ds_alpha, "dataset_name": "Alpha Dataset", "source": "Source A"})
    db.save_dataset_metadata({"dataset_id": ds_beta, "dataset_name": "Beta Dataset", "source": "Source B"})

    # Save patients with the same patient_id
    shared_id = "SHARED-PATIENT-001"
    df_a = pd.DataFrame([{"patient_id": shared_id, "creatinine": 3.5, "systolic_bp": 170.0}])
    df_b = pd.DataFrame([{"patient_id": shared_id, "creatinine": 0.9, "systolic_bp": 118.0}])
    db.save_patients(df_a, dataset_id=ds_alpha)
    db.save_patients(df_b, dataset_id=ds_beta)

    run_a = db.save_model_run({"eps": 1.5, "min_samples": 5, "number_of_clusters": 1, "number_of_anomalies": 1, "dataset_size": 1, "silhouette_score": 0.0}, dataset_id=ds_alpha)
    run_b = db.save_model_run({"eps": 1.5, "min_samples": 5, "number_of_clusters": 1, "number_of_anomalies": 0, "dataset_size": 1, "silhouette_score": 0.0}, dataset_id=ds_beta)

    # Create mock suggestions
    s_alpha = [{
        "patient_id": shared_id,
        "analysis_run_id": run_a,
        "clinical_priority": "HIGH",
        "primary_domain": "RENAL",
        "all_domains": ["RENAL"],
        "multi_system_signal": 0,
        "key_signals": [{"title": "Renal Signal Alpha"}],
        "z_score_drivers": [{"feature": "creatinine", "z_score": 3.5}],
        "investigation_suggestions": ["Consider renal panel review"],
        "specialist_referrals": [{"specialty": "Nephrology"}],
        "medication_review_notes": "Review nephrotoxins",
        "procedure_review_notes": "No surgery",
        "medical_references": []
    }]

    s_beta = [{
        "patient_id": shared_id,
        "analysis_run_id": run_b,
        "clinical_priority": "LOW",
        "primary_domain": "GENERAL",
        "all_domains": [],
        "multi_system_signal": 0,
        "key_signals": [{"title": "Normal Variation Beta"}],
        "z_score_drivers": [],
        "investigation_suggestions": ["Consider routine wellness"],
        "specialist_referrals": [{"specialty": "Primary Care"}],
        "medication_review_notes": "Standard review",
        "procedure_review_notes": "No intervention",
        "medical_references": []
    }]

    db.save_clinical_suggestions(s_alpha, dataset_id=ds_alpha, run_id=run_a)
    db.save_clinical_suggestions(s_beta, dataset_id=ds_beta, run_id=run_b)

    # Verify retrieval respects dataset isolation
    retrieved_alpha = db.get_clinical_suggestions(dataset_id=ds_alpha)
    retrieved_beta = db.get_clinical_suggestions(dataset_id=ds_beta)

    assert len(retrieved_alpha) == 1
    assert len(retrieved_beta) == 1
    assert retrieved_alpha[0]["clinical_priority"] == "HIGH"
    assert retrieved_beta[0]["clinical_priority"] == "LOW"
    assert retrieved_alpha[0]["analysis_run_id"] == run_a
    assert retrieved_beta[0]["analysis_run_id"] == run_b

    # Verify patient-specific retrieval
    p_alpha = db.get_patient_clinical_suggestion(ds_alpha, shared_id)
    p_beta = db.get_patient_clinical_suggestion(ds_beta, shared_id)
    assert p_alpha["clinical_priority"] == "HIGH"
    assert p_beta["clinical_priority"] == "LOW"

    # Verify cascading delete of Dataset Alpha does not affect Dataset Beta
    db.delete_dataset(ds_alpha)
    assert len(db.get_clinical_suggestions(dataset_id=ds_alpha)) == 0
    assert len(db.get_clinical_suggestions(dataset_id=ds_beta)) == 1

    # Clean up beta
    db.delete_dataset(ds_beta)


def test_cds_web_routes():
    """Verify Flask routes for Clinical Decision Support return HTTP 200."""
    client = app.test_client()

    res_cohort = client.get("/clinical-suggestions")
    assert res_cohort.status_code == 200
    assert b"Clinical Decision Support" in res_cohort.data

    res_csv = client.get("/export-clinical-suggestions-csv")
    assert res_csv.status_code == 200
    assert res_csv.content_type.startswith("text/csv")
