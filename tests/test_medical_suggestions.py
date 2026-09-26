"""
tests/test_medical_suggestions.py
=============================================================================
Comprehensive Test Suite for Medical Suggestions, Feature Explainability & Specialist Recommendation
=============================================================================
"""

import pytest
import numpy as np
import pandas as pd
from models.clinical_decision_support import ClinicalDecisionSupportEngine
from database.db import DatabaseManager
from app import app


@pytest.fixture
def cds():
    return ClinicalDecisionSupportEngine()


@pytest.fixture
def test_db():
    return DatabaseManager()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# =============================================================================
# 1. FEATURE EXPLANATION LIBRARY TESTS
# =============================================================================

def test_feature_explanation_library_coverage(cds):
    """Verify that core clinical parameters have MedlinePlus/NIH-aligned explanations."""
    core_features = [
        "creatinine", "bun", "hemoglobin", "wbc", "platelets",
        "systolic_bp", "diastolic_bp", "heart_rate", "glucose",
        "cholesterol", "oxygen_saturation", "body_temp"
    ]
    for feat in core_features:
        assert feat in cds.FEATURE_EXPLANATION_LIBRARY, f"Missing feature in library: {feat}"
        entry = cds.FEATURE_EXPLANATION_LIBRARY[feat]
        assert "what_it_means" in entry
        assert "clinical_area" in entry
        assert "suggested_action" in entry
        assert len(entry["what_it_means"]) > 20


def test_get_feature_explanation_structure(cds):
    """Verify that get_feature_explanation returns the required 3-part structured format."""
    explanation = cds.get_feature_explanation("serum_creatinine", z_score=3.2)
    assert explanation["clinical_area"] == "Renal"
    assert "creatinine" in explanation["what_it_means"].lower()
    assert "3.2 standard deviations" in explanation["why_flagged"]
    assert "higher than" in explanation["why_flagged"]
    assert "Consider" in explanation["suggested_action"]
    assert explanation["severity"] == "Severe Deviation"


# =============================================================================
# 2. EMERGENCY VITAL SIGNS SAFETY BANNER TESTS
# =============================================================================

def test_emergency_vital_signs_detection(cds):
    """Verify that critical vital thresholds trigger the emergency safety notice."""
    # Patient with severe hypoxemia (SpO2 88%) and malignant hypertension (SBP 185)
    row_crit = {
        "patient_id": "P-EMERGENCY",
        "oxygen_saturation": 88.0,
        "systolic_bp": 185.0,
        "heart_rate": 140.0
    }
    emg = cds.detect_emergency_vitals(row_crit)
    assert emg["is_emergency"] is True
    assert len(emg["critical_vitals"]) >= 2

    # Verify exact required wording in notice
    required_text = (
        "URGENT CLINICAL REVIEW MAY BE APPROPRIATE. If the patient has severe, "
        "sudden, or worsening symptoms, seek immediate medical attention or contact "
        "local emergency services rather than relying on this application."
    )
    assert required_text in cds.EMERGENCY_NOTICE or required_text in emg["emergency_notice"]


def test_non_emergency_patient(cds):
    """Verify that patients with normal vital signs do not trigger emergency alert."""
    row_normal = {
        "patient_id": "P-STABLE",
        "oxygen_saturation": 98.0,
        "systolic_bp": 120.0,
        "diastolic_bp": 80.0,
        "heart_rate": 72.0,
        "body_temp": 37.0
    }
    emg = cds.detect_emergency_vitals(row_normal)
    assert emg["is_emergency"] is False
    assert len(emg["critical_vitals"]) == 0


# =============================================================================
# 3. HIERARCHICAL SPECIALIST RECOMMENDATION TESTS
# =============================================================================

def test_multi_system_specialist_hierarchy(cds):
    """
    Verify that multi-system cases prioritize General Physician / Internal Medicine Specialist
    as primary coordinator, listing organ-specific subspecialists.
    """
    domains = ["RENAL", "CARDIOVASCULAR", "HEMATOLOGICAL"]
    spec_info = cds.determine_specialist_hierarchy(domains, multi_system=True)

    assert spec_info["hierarchy_type"] == "MULTI_SYSTEM_GENERAL_FIRST"
    assert "General Physician" in spec_info["primary_specialist"]
    assert "Internal Medicine" in spec_info["primary_specialist"]
    assert len(spec_info["additional_specialists"]) >= 2
    
    sub_specialties = [s["specialty"] for s in spec_info["additional_specialists"]]
    assert "Nephrologist" in sub_specialties
    assert "Cardiologist" in sub_specialties


def test_single_domain_specialist_routing(cds):
    """Verify that single-domain cases route directly to the organ-specific specialist."""
    domains_renal = ["RENAL"]
    spec_renal = cds.determine_specialist_hierarchy(domains_renal, multi_system=False)
    assert spec_renal["primary_specialist"] == "Nephrologist"
    assert spec_renal["hierarchy_type"] == "TARGETED_ORGAN_SPECIALIST"

    domains_cardio = ["CARDIOVASCULAR"]
    spec_cardio = cds.determine_specialist_hierarchy(domains_cardio, multi_system=False)
    assert spec_cardio["primary_specialist"] == "Cardiologist"


# =============================================================================
# 4. LIFESTYLE & GENERAL HEALTH CONSIDERATIONS TESTS
# =============================================================================

def test_lifestyle_considerations(cds):
    """Verify safe, non-interventional lifestyle guidance is provided."""
    lifestyle = cds.LIFESTYLE_CONSIDERATIONS
    assert len(lifestyle) >= 4
    categories = [item["category"] for item in lifestyle]
    assert "Hydration & Fluid Intake" in categories
    assert "Nutrition & Dietary Balance" in categories
    assert "Vital Sign Self-Monitoring" in categories
    for item in lifestyle:
        assert len(item["recommendation"]) > 10
        assert len(item["rationale"]) > 10


# =============================================================================
# 5. DATABASE PERSISTENCE & DATASET ISOLATION TESTS
# =============================================================================

def test_clinical_recommendations_persistence_and_isolation(test_db):
    """Verify clinical_recommendations table isolation across datasets."""
    ds1 = "ds_test_rec_1"
    ds2 = "ds_test_rec_2"

    test_db.save_dataset_metadata({
        "dataset_id": ds1,
        "dataset_name": "Test Dataset 1",
        "source": "Source 1",
        "description": "Test dataset",
        "filename": "d1.csv",
        "record_count": 1,
        "feature_count": 2,
        "status": "Analyzed"
    })
    test_db.save_dataset_metadata({
        "dataset_id": ds2,
        "dataset_name": "Test Dataset 2",
        "source": "Source 2",
        "description": "Test dataset",
        "filename": "d2.csv",
        "record_count": 1,
        "feature_count": 2,
        "status": "Analyzed"
    })

    # Save patient records first (required for FK constraints)
    test_db.save_patients(pd.DataFrame([{"patient_id": "P1", "age": 45, "creatinine": 2.5}]), dataset_id=ds1)
    test_db.save_patients(pd.DataFrame([{"patient_id": "P2", "age": 55, "glucose": 180.0}]), dataset_id=ds2)

    rec1 = [{
        "dataset_id": ds1,
        "patient_id": "P1",
        "analysis_run_id": None,
        "clinical_domain": "RENAL",
        "abnormal_feature": "Creatinine",
        "z_score": 3.1,
        "severity": "Severe Deviation",
        "clinical_explanation": "Creatinine is elevated above population reference.",
        "medical_suggestion": "Consider renal evaluation.",
        "investigation_suggestion": "Consider repeat creatinine and eGFR.",
        "primary_specialist": "Nephrologist",
        "additional_specialists": [],
        "medication_review": "Clinician medication review.",
        "procedure_review": "No automatic surgical recommendation.",
        "priority": "HIGH",
        "evidence_reference": "KDIGO Guidelines"
    }]

    rec2 = [{
        "dataset_id": ds2,
        "patient_id": "P2",
        "analysis_run_id": None,
        "clinical_domain": "METABOLIC",
        "abnormal_feature": "Glucose",
        "z_score": 2.8,
        "severity": "Moderate Shift",
        "clinical_explanation": "Fasting glucose elevated.",
        "medical_suggestion": "Consider glycemic evaluation.",
        "investigation_suggestion": "Consider HbA1c testing.",
        "primary_specialist": "Endocrinologist",
        "additional_specialists": [],
        "medication_review": "Clinician review.",
        "procedure_review": "No surgical mandate.",
        "priority": "MEDIUM",
        "evidence_reference": "ADA Guidelines"
    }]

    test_db.save_clinical_recommendations(rec1, dataset_id=ds1)
    test_db.save_clinical_recommendations(rec2, dataset_id=ds2)

    recs_ds1 = test_db.get_clinical_recommendations(dataset_id=ds1)
    recs_ds2 = test_db.get_clinical_recommendations(dataset_id=ds2)

    assert len(recs_ds1) == 1
    assert recs_ds1[0]["abnormal_feature"] == "Creatinine"
    assert recs_ds1[0]["patient_id"] == "P1"

    assert len(recs_ds2) == 1
    assert recs_ds2[0]["abnormal_feature"] == "Glucose"
    assert recs_ds2[0]["patient_id"] == "P2"

    # Clean up test datasets
    test_db.delete_dataset(ds1)
    test_db.delete_dataset(ds2)


# =============================================================================
# 6. WEB ROUTE INTEGRATION TESTS
# =============================================================================

def test_medical_suggestions_routes(client, test_db):
    """Verify that /medical-suggestions and /patients/<ds>/<id>/medical-suggestions return 200 OK."""
    # Cohort suggestions page
    res = client.get("/medical-suggestions")
    assert res.status_code == 200
    assert b"Medical Suggestions" in res.data
    assert b"URGENT CLINICAL REVIEW MAY BE APPROPRIATE" in res.data

    # Legacy alias route
    res_legacy = client.get("/clinical-suggestions")
    assert res_legacy.status_code == 200

    # Single patient medical suggestions route
    df_pats = test_db.get_full_patient_analysis("ds_default")
    pat_id = df_pats["patient_id"].iloc[0] if not df_pats.empty else "PAT-11027"

    res_pat = client.get(f"/patients/ds_default/{pat_id}/medical-suggestions")
    assert res_pat.status_code == 200
    assert pat_id.encode() in res_pat.data
    assert b"Clinical Feature Abnormality Analysis" in res_pat.data
    assert b"Hierarchical Specialist Guidance" in res_pat.data
    assert b"URGENT CLINICAL REVIEW MAY BE APPROPRIATE" in res_pat.data

    # Export CSV endpoint
    res_csv = client.get("/export-medical-suggestions-csv")
    assert res_csv.status_code == 200
    assert res_csv.mimetype == "text/csv"
