import pytest
import pandas as pd
import numpy as np
from models.clinical_recommendation_engine import ClinicalRecommendationEngine
from database.db import db_manager
from analysis.report_generator import AnalysisReportGenerator
from app import app

@pytest.fixture
def recommendation_engine():
    return ClinicalRecommendationEngine(z_threshold=1.7, z_extreme=3.0)

@pytest.fixture
def sample_patient_data():
    patient = {
        "patient_id": "TEST-REC-001",
        "systolic_bp": 185.0,
        "diastolic_bp": 115.0,
        "creatinine": 3.4,
        "glucose": 95.0,
        "hemoglobin": 13.5,
        "is_anomaly": 1,
        "cluster_label": -1,
        "anomaly_score": 88.5
    }
    pop_means = pd.Series({
        "systolic_bp": 120.0,
        "diastolic_bp": 80.0,
        "creatinine": 1.0,
        "glucose": 90.0,
        "hemoglobin": 14.0
    })
    pop_stds = pd.Series({
        "systolic_bp": 12.0,
        "diastolic_bp": 8.0,
        "creatinine": 0.3,
        "glucose": 15.0,
        "hemoglobin": 1.5
    })
    return patient, pop_means, pop_stds

def test_recommendation_engine_categories(recommendation_engine, sample_patient_data):
    patient, pop_means, pop_stds = sample_patient_data
    rec = recommendation_engine.generate_recommendations_for_patient(
        patient_row=patient,
        pop_means=pop_means,
        pop_stds=pop_stds,
        dataset_id="test_ds",
        analysis_run_id=1
    )

    # Priority
    assert rec["priority"] in ["URGENT REVIEW", "PRIORITY REVIEW", "ROUTINE FOLLOW-UP", "GENERAL MONITORING"]
    assert rec["priority"] == "URGENT REVIEW" # High score & extreme Z-scores

    # Category A: What was detected?
    assert "category_a_what_detected" in rec
    assert len(rec["category_a_what_detected"]["deviations"]) >= 2
    assert "CARDIOVASCULAR" in rec["category_a_what_detected"]["domains"]
    assert "RENAL" in rec["category_a_what_detected"]["domains"]

    # Category B: Why flagged?
    assert "category_b_why_flagged" in rec
    assert "DBSCAN" in rec["category_b_why_flagged"]["summary"]

    # Category C: Recommended investigations
    assert "category_c_investigations" in rec
    assert len(rec["category_c_investigations"]) > 0

    # Category D: Recommended specialist
    assert "category_d_specialists" in rec
    assert "primary_specialist" in rec["category_d_specialists"]

    # Category E: General risk-reduction guidance
    assert "category_e_risk_reduction" in rec
    assert len(rec["category_e_risk_reduction"]) >= 3

    # Category F: Management considerations
    assert "category_f_management_considerations" in rec
    assert len(rec["category_f_management_considerations"]) > 0

    # Safety disclaimer
    assert "disclaimer" in rec
    assert "medical diagnosis" in rec["disclaimer"]

def test_database_recommendation_persistence(recommendation_engine, sample_patient_data):
    patient, pop_means, pop_stds = sample_patient_data
    rec = recommendation_engine.generate_recommendations_for_patient(
        patient_row=patient,
        pop_means=pop_means,
        pop_stds=pop_stds,
        dataset_id="test_ds_rec",
        analysis_run_id=999
    )

    db_records = rec["db_records"]
    assert len(db_records) > 0

    # Test save and retrieval
    db_manager.create_tables()
    db_manager.save_dataset_metadata({
        "dataset_id": "test_ds_rec",
        "dataset_name": "Test Rec Dataset",
        "source": "Pytest",
        "description": "Test dataset",
        "filename": "test.csv",
        "record_count": 1,
        "feature_count": 5,
        "status": "Analyzed"
    })
    db_manager.save_patients(pd.DataFrame([patient]), dataset_id="test_ds_rec")
    db_manager.save_clinical_recommendations(db_records, dataset_id="test_ds_rec", run_id=None)
    retrieved = db_manager.get_clinical_recommendations(dataset_id="test_ds_rec", patient_id="TEST-REC-001")
    assert len(retrieved) == len(db_records)
    assert retrieved[0]["patient_id"] == "TEST-REC-001"
    assert retrieved[0]["dataset_id"] == "test_ds_rec"

def test_recommendation_web_routes(client=None):
    with app.test_client() as test_client:
        # Test API endpoint
        resp = test_client.get("/api/patients/PAT-11027/recommendations")
        if resp.status_code == 200:
            data = resp.get_json()
            assert "priority" in data
            assert "category_a_what_detected" in data
            assert "category_c_investigations" in data
            assert "disclaimer" in data

        # Test HTML page
        html_resp = test_client.get("/patients/PAT-11027/recommendations")
        if html_resp.status_code == 200:
            assert b"Clinical Recommendations" in html_resp.data
            assert b"MANDATORY ACADEMIC" in html_resp.data
