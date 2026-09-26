"""
models/clinical_recommendation_engine.py
=============================================================================
RARETRACE — Clinical Recommendation & Solution Engine
=============================================================================
Evidence-aligned decision-support module providing structured, explainable
next-step recommendations for candidate rare-disease cases flagged by DBSCAN.

STRICT MEDICAL SAFETY & REGULATORY NOTICE:
RARETRACE is an academic research and clinical decision-support prototype.
Statistical anomaly detection DOES NOT constitute a medical diagnosis.
This engine generates non-prescriptive, evidence-based recommendations
exclusively for clinician review and evaluation. It does NOT diagnose
conditions, prescribe pharmaceutical agents, or direct patient treatment.
=============================================================================
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


class ClinicalRecommendationEngine:
    """
    Evidence-based Next-Step Recommendation & Solution Engine for RARETRACE.
    Transforms raw statistical outliers, DBSCAN noise flags, and Z-score feature
    attributions into transparent, clinical-domain-specific recommendations.
    """

    MANDATORY_DISCLAIMER = (
        "RARETRACE is an academic research and clinical decision-support prototype. "
        "Statistical anomaly detection does not constitute a medical diagnosis. "
        "Results and suggested next steps must be evaluated and verified by qualified healthcare professionals."
    )

    EMERGENCY_DISCLAIMER = (
        "CRITICAL VITAL NOTICE: If the patient exhibits acute, severe, or deteriorating symptoms, "
        "immediate emergency clinical care should be accessed without delay."
    )

    # Recognizable clinical patterns based on multi-system combinations
    CLINICAL_PATTERNS = {
        "CARDIO_RENAL_SYNDROME_LIKE": {
            "name": "Cardiorenal Axis Vulnerability Pattern",
            "required_domains": ["CARDIOVASCULAR", "RENAL"],
            "description": "Concurrent deviation in hemodynamic pressures (systolic/diastolic BP) and renal filtration markers (creatinine, BUN).",
            "suggested_specialist": "Cardiologist & Nephrologist (Multidisciplinary)",
            "recommended_tests": [
                "Comprehensive metabolic panel (BUN, serum creatinine, eGFR, electrolytes)",
                "Spot urine albumin-to-creatinine ratio (uACR) and microscopic urinalysis",
                "24-hour ambulatory blood pressure monitoring (ABPM)",
                "12-lead Electrocardiogram (ECG) and Transthoracic Echocardiogram (TTE)"
            ],
            "general_next_steps": [
                "Schedule formal multidisciplinary cardiorenal clinical evaluation",
                "Maintain accurate daily resting blood pressure and fluid log",
                "Discuss dietary sodium moderation with a clinical dietitian"
            ],
            "management_considerations": [
                "Evaluation of renal protective hemodynamic strategies by an attending physician",
                "Review of potential nephrotoxic and antihypertensive medication interactions",
                "Assessment for secondary causes of resistant hypertension"
            ]
        },
        "METABOLIC_HEMATOLOGICAL_PATTERN": {
            "name": "Metabolic-Hematological Discordance Pattern",
            "required_domains": ["METABOLIC", "HEMATOLOGICAL"],
            "description": "Co-occurring abnormalities across systemic metabolic regulators (glucose, cholesterol, BMI) and cellular hematology (hemoglobin, platelets, WBC).",
            "suggested_specialist": "Endocrinologist & Hematologist",
            "recommended_tests": [
                "Fasting plasma glucose, HbA1c, and fasting lipid fraction panel",
                "Complete blood count (CBC) with manual peripheral blood smear",
                "Iron studies (serum ferritin, transferrin saturation, TIBC)",
                "Comprehensive hepatic function panel and inflammatory markers (hs-CRP)"
            ],
            "general_next_steps": [
                "Consult an internal medicine specialist for comprehensive systemic review",
                "Implement structured glycemic and cardiovascular lifestyle risk reduction",
                "Verify hematological stability with repeat CBC at 4-week interval"
            ],
            "management_considerations": [
                "Evaluation for underlying endocrine or hematological etiology by specialist",
                "Assessment of metabolic syndrome criteria and associated microvascular risk",
                "Review of nutritional status and systemic metabolic clearance"
            ]
        },
        "MULTI_ORGAN_INFLAMMATORY_PATTERN": {
            "name": "Multi-System Complex Anomaly Pattern",
            "required_domains": ["RENAL", "HEMATOLOGICAL", "CARDIOVASCULAR"],
            "description": "Pervasive statistical outlier across three or more physiological systems, characteristic of systemic or rare multisystem pathologies.",
            "suggested_specialist": "Internal Medicine / Rheumatologist / Clinical Geneticist",
            "recommended_tests": [
                "Comprehensive autoantibody panel (ANA, anti-dsDNA, complement C3/C4)",
                "Serum and urine protein electrophoresis with immunofixation",
                "Full systemic metabolic, hematological, and organ biomarker panel",
                "Consider referral to a specialized diagnostic clinic or rare-disease consortium"
            ],
            "general_next_steps": [
                "Prioritize comprehensive clinical assessment with internal medicine lead",
                "Compile detailed three-generation pedigree and family history",
                "Coordinate unified inter-specialty case review"
            ],
            "management_considerations": [
                "Holistic diagnostic workup to exclude systemic autoimmune, infiltrative, or rare genetic conditions",
                "Careful surveillance for acute organ dysfunction or decompensation",
                "Avoid polypharmacy until clear clinical trajectory is established"
            ]
        },
        "ISOLATED_RENAL_OUTLIER": {
            "name": "Isolated Renal Biomarker Elevation Pattern",
            "required_domains": ["RENAL"],
            "description": "Disproportionate deviation in renal function markers while other physiological domains remain largely normative.",
            "suggested_specialist": "Nephrologist",
            "recommended_tests": [
                "Repeat serum creatinine and cystatin-C to confirm baseline eGFR",
                "Renal duplex ultrasound to assess kidney size, echogenicity, and corticomedullary differentiation",
                "Complete urinalysis for hematuria and proteinuria evaluation"
            ],
            "general_next_steps": [
                "Ensure adequate clinical hydration unless fluid-restricted by clinician",
                "Avoid non-steroidal anti-inflammatory drugs (NSAIDs) and nephrotoxic exposures",
                "Schedule nephrology outpatient consultation"
            ],
            "management_considerations": [
                "Identification of acute kidney injury versus chronic kidney disease etiology",
                "Staging and etiology workup under guidance of attending nephrologist",
                "Renal dose adjustments for all current and future medications"
            ]
        },
        "ISOLATED_CARDIOVASCULAR_OUTLIER": {
            "name": "Marked Hemodynamic Disruption Pattern",
            "required_domains": ["CARDIOVASCULAR"],
            "description": "Marked divergence in arterial blood pressures or cardiac rhythm metrics from the reference cohort.",
            "suggested_specialist": "Cardiologist",
            "recommended_tests": [
                "Serial seated and standing blood pressure measurements (orthostatic vitals)",
                "Standard 12-lead ECG to evaluate for chamber enlargement or ischemic changes",
                "Echocardiogram and baseline cardiac biomarkers where indicated"
            ],
            "general_next_steps": [
                "Maintain home blood pressure log with certified automated cuff",
                "Reduce dietary sodium and limit stimulants (caffeine, nicotine)",
                "Discuss findings with primary care physician or cardiologist"
            ],
            "management_considerations": [
                "Cardiovascular risk stratification and secondary hypertension screening",
                "Optimization of evidence-based hemodynamic therapy by attending physician",
                "Monitoring for end-organ target damage (retinopathy, nephropathy)"
            ]
        },
        "ISOLATED_HEMATOLOGICAL_OUTLIER": {
            "name": "Cytopenia or Proliferative Hematology Pattern",
            "required_domains": ["HEMATOLOGICAL"],
            "description": "Isolated extreme departure in one or more peripheral blood cell lineages.",
            "suggested_specialist": "Hematologist",
            "recommended_tests": [
                "Repeat complete blood count with reticulocyte count and manual differential",
                "Peripheral blood smear review by a hematopathologist",
                "Coagulation profile (PT, INR, aPTT, fibrinogen) if platelet counts deviant"
            ],
            "general_next_steps": [
                "Monitor for clinical signs of anemia, unusual bleeding, petechiae, or infection",
                "Avoid medications that impair platelet function until evaluated",
                "Seek prompt clinical evaluation if systemic fevers or bruising occur"
            ],
            "management_considerations": [
                "Differentiation between reactive/secondary vs primary bone marrow conditions",
                "Hematinic repletion or specialized workup as directed by hematologist",
                "Infection prophylaxis and bleeding precautions if cytopenias severe"
            ]
        },
        "GENERAL_ANOMALOUS_PROFILE": {
            "name": "Atypical Clinical Profile Requiring Investigation",
            "required_domains": [],
            "description": "Statistically unusual multivariate profile isolated by density clustering warranting holistic medical review.",
            "suggested_specialist": "General Physician / Internal Medicine Specialist",
            "recommended_tests": [
                "Comprehensive clinical examination and updated medical history",
                "Baseline metabolic and hematological health panel",
                "Targeted repeat measurement of anomalous features"
            ],
            "general_next_steps": [
                "Review findings during scheduled visit with primary care clinician",
                "Maintain healthy sleep, nutrition, and exercise routines",
                "Document any new, unusual, or recurring symptoms"
            ],
            "management_considerations": [
                "Correlation of statistical findings with clinical presentation and history",
                "Determination of appropriate follow-up intervals based on clinician judgment",
                "Preventive lifestyle interventions"
            ]
        }
    }

    # Domain to Specialist Mapping
    SPECIALIST_MAP = {
        "RENAL": "Nephrologist",
        "CARDIOVASCULAR": "Cardiologist",
        "METABOLIC": "Endocrinologist",
        "HEMATOLOGICAL": "Hematologist",
        "RESPIRATORY": "Pulmonologist",
        "NEUROLOGICAL": "Neurologist",
        "GENERAL": "General Physician / Internal Medicine Specialist"
    }

    def __init__(self, z_threshold: float = 1.7, z_extreme: float = 3.0):
        self.z_threshold = z_threshold
        self.z_extreme = z_extreme

    def generate_recommendations_for_patient(
        self,
        patient_row: Dict[str, Any],
        pop_means: Optional[pd.Series] = None,
        pop_stds: Optional[pd.Series] = None,
        dataset_id: str = "ds_default",
        analysis_run_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate complete evidence-based recommendation dossier for a single patient.
        Adheres to all 6 required categories:
          A. What was detected?
          B. Why was this profile flagged?
          C. Recommended next investigation
          D. Recommended specialist
          E. General risk-reduction guidance
          F. Possible management considerations
        """
        patient_id = str(patient_row.get("patient_id", "Unknown"))
        is_anomaly = int(patient_row.get("is_anomaly", 0))
        cluster_label = int(patient_row.get("cluster_label", 0))
        anomaly_score = float(patient_row.get("anomaly_score", 0.0))

        # 1. Analyze Feature Deviations (Explainability & Attribution)
        deviations = []
        domains_detected = set()

        if pop_means is not None and pop_stds is not None:
            cols = list(pop_means.keys()) if isinstance(pop_means, dict) else list(pop_means.index)
            for col in cols:
                if col in patient_row and pd.notna(patient_row[col]):
                    val = float(patient_row[col])
                    mean = float(pop_means[col])
                    std = float(pop_stds[col]) if pop_stds[col] != 0 else 1.0
                    z = (val - mean) / std

                    if abs(z) >= self.z_threshold:
                        domain = self._classify_feature_domain(col)
                        domains_detected.add(domain)
                        direction = "Elevated" if z > 0 else "Depressed"
                        severity = "Critical" if abs(z) >= self.z_extreme else "Significant"
                        
                        deviations.append({
                            "feature": col,
                            "feature_label": col.replace("_", " ").title(),
                            "patient_value": round(val, 2),
                            "pop_mean": round(mean, 2),
                            "pop_std": round(std, 2),
                            "z_score": round(z, 2),
                            "direction": direction,
                            "severity": severity,
                            "domain": domain,
                            "finding_summary": f"{direction} {col.replace('_', ' ').title()} ({round(val, 2)} vs mean {round(mean, 2)}, Z = {round(z, 2)})",
                            "evidence": f"Patient value ({round(val, 2)}) deviates by {abs(round(z, 2))} standard deviations from reference population mean ({round(mean, 2)})."
                        })

        deviations.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        domains_list = sorted(list(domains_detected))

        # 2. Determine Recommendation Priority
        extreme_count = sum(1 for d in deviations if abs(d["z_score"]) >= self.z_extreme)
        
        if anomaly_score >= 80.0 or extreme_count >= 2 or (len(domains_list) >= 3 and is_anomaly == 1):
            priority = "URGENT REVIEW"
            priority_badge = "badge-danger"
            priority_desc = "Profile exhibits extreme multivariate distance and multi-organ divergence requiring timely clinician review."
        elif anomaly_score >= 50.0 or len(deviations) >= 3 or is_anomaly == 1:
            priority = "PRIORITY REVIEW"
            priority_badge = "badge-warning"
            priority_desc = "Profile shows notable statistical density isolation and multiple significant biomarker departures."
        elif len(deviations) >= 1:
            priority = "ROUTINE FOLLOW-UP"
            priority_badge = "badge-info"
            priority_desc = "Mild or isolated biomarker departure suitable for routine clinical follow-up."
        else:
            priority = "GENERAL MONITORING"
            priority_badge = "badge-success"
            priority_desc = "Clinical vitals and biomarkers align within expected population distributions."

        # 3. Category A: What was detected?
        if deviations:
            findings_bullets = [d["finding_summary"] for d in deviations[:6]]
            detected_summary = (
                f"{len(deviations)} significant clinical parameter departure(s) detected across "
                f"{len(domains_list)} physiological domain(s) ({', '.join(domains_list)})."
            )
        else:
            findings_bullets = ["All measured clinical parameters reside within normative reference distributions."]
            detected_summary = "No statistically significant biomarker or vital-sign deviations detected."

        # 4. Category B: Why was this profile flagged?
        flag_reasons = []
        if is_anomaly == 1:
            flag_reasons.append(
                f"Classified as DBSCAN density noise (Cluster label: {cluster_label}) because the profile "
                f"did not fall within any high-density normative patient clusters."
            )
        if anomaly_score > 0:
            flag_reasons.append(
                f"Continuous spatial anomaly score of {anomaly_score:.1f}/100, reflecting substantial Euclidean "
                f"distance from core cluster population centers in scaled clinical space."
            )
        if deviations:
            flag_reasons.append(
                f"Presence of {len(deviations)} feature(s) exceeding statistical deviation threshold (|Z| >= {self.z_threshold}), "
                f"including {extreme_count} extreme deviation(s) (|Z| >= {self.z_extreme})."
            )
        if not flag_reasons:
            flag_reasons.append("Normative case included for comparative baseline evaluation.")

        why_flagged_text = " ".join(flag_reasons)

        # 5. Match Recognizable Clinical Patterns
        matched_pattern = self._match_pattern(domains_list, is_anomaly)

        # 6. Category C: Recommended next investigation
        recommended_investigations = []
        for test in matched_pattern["recommended_tests"]:
            recommended_investigations.append({
                "test_name": test,
                "domain": matched_pattern.get("required_domains", ["GENERAL"])[0] if matched_pattern.get("required_domains") else "GENERAL",
                "purpose": "Confirmatory objective laboratory evaluation directed by attending physician.",
                "reason": f"Indicated based on observed deviations in {', '.join(domains_list) if domains_list else 'general screening'}."
            })

        # 7. Category D: Recommended specialist
        suggested_specialist = matched_pattern["suggested_specialist"]
        additional_specialists = [
            self.SPECIALIST_MAP.get(dom, "Specialist") 
            for dom in domains_list 
            if self.SPECIALIST_MAP.get(dom) not in suggested_specialist
        ]

        # 8. Category E: General risk-reduction guidance
        risk_reduction_guidance = [
            {
                "topic": "Clinical Consultation",
                "guidance": "Schedule a follow-up appointment with a qualified primary care or specialist clinician to review these findings.",
                "rationale": "Objective clinical correlation is required to establish clinical relevance."
            },
            {
                "topic": "Vitals & Symptom Monitoring",
                "guidance": "Maintain a daily log of resting blood pressure, heart rate, and any unexplained symptoms (e.g. fatigue, shortness of breath).",
                "rationale": "Serial measurements provide context on whether deviations are acute or chronic."
            },
            {
                "topic": "Lifestyle & Metabolic Health",
                "guidance": "Follow heart-healthy nutritional practices, maintain age-appropriate physical activity, and maintain proper hydration.",
                "rationale": "Supports systemic hemodynamic, renal, and metabolic stability."
            },
            {
                "topic": "Medication Review Safety",
                "guidance": "Do not initiate, discontinue, or adjust prescription medications based solely on automated software results.",
                "rationale": "Medication decisions require comprehensive pharmacological evaluation by licensed providers."
            }
        ]

        # 9. Category F: Possible management considerations
        management_considerations = matched_pattern.get("management_considerations", [
            "Comprehensive review of past medical records and medication history.",
            "Repeat testing of anomalous parameters to exclude transient physiological variation.",
            "Formulation of individualized monitoring plan by attending medical team."
        ])

        # 10. Compile DB Records for Traceability (clinical_recommendations table)
        db_records = []
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        if deviations:
            for dev in deviations:
                db_records.append({
                    "dataset_id": dataset_id,
                    "patient_id": patient_id,
                    "analysis_run_id": analysis_run_id,
                    "clinical_domain": dev["domain"],
                    "abnormal_feature": dev["feature_label"],
                    "z_score": dev["z_score"],
                    "severity": dev["severity"],
                    "clinical_explanation": f"{dev['finding_summary']}. Classified under {matched_pattern['name']}.",
                    "medical_suggestion": f"Evaluate {dev['feature_label']} alongside {suggested_specialist}.",
                    "investigation_suggestion": recommended_investigations[0]["test_name"] if recommended_investigations else "Repeat testing",
                    "primary_specialist": suggested_specialist,
                    "additional_specialists_json": json.dumps(additional_specialists),
                    "medication_review": "Review current medication regimen with prescribing physician.",
                    "procedure_review": "No automatic procedures recommended. Specialist evaluation required.",
                    "priority": priority,
                    "evidence_reference": dev["evidence"],
                    "created_at": now_ts
                })
        else:
            db_records.append({
                "dataset_id": dataset_id,
                "patient_id": patient_id,
                "analysis_run_id": analysis_run_id,
                "clinical_domain": "GENERAL",
                "abnormal_feature": "Normative Concordance",
                "z_score": 0.0,
                "severity": "Normal Profile",
                "clinical_explanation": "All quantitative measures within population norms.",
                "medical_suggestion": "Continue routine preventive clinical screening.",
                "investigation_suggestion": "Standard periodic physical examination.",
                "primary_specialist": "General Physician / Internal Medicine Specialist",
                "additional_specialists_json": json.dumps([]),
                "medication_review": "Maintain regular physician-supervised medication reviews.",
                "procedure_review": "None indicated.",
                "priority": priority,
                "evidence_reference": "All features within normal variance.",
                "created_at": now_ts
            })

        return {
            "dataset_id": dataset_id,
            "patient_id": patient_id,
            "analysis_run_id": analysis_run_id,
            "priority": priority,
            "priority_badge": priority_badge,
            "priority_description": priority_desc,
            "anomaly_score": anomaly_score,
            "cluster_label": cluster_label,
            "is_anomaly": is_anomaly,
            "category_a_what_detected": {
                "summary": detected_summary,
                "findings": findings_bullets,
                "deviations": deviations,
                "domains": domains_list
            },
            "category_b_why_flagged": {
                "summary": why_flagged_text,
                "reasons": flag_reasons
            },
            "category_c_investigations": recommended_investigations,
            "category_d_specialists": {
                "primary_specialist": suggested_specialist,
                "additional_specialists": additional_specialists
            },
            "category_e_risk_reduction": risk_reduction_guidance,
            "category_f_management_considerations": management_considerations,
            "matched_pattern": matched_pattern,
            "db_records": db_records,
            "disclaimer": self.MANDATORY_DISCLAIMER,
            "emergency_disclaimer": self.EMERGENCY_DISCLAIMER
        }

    def _classify_feature_domain(self, feature_name: str) -> str:
        """Classify clinical feature name into physiological domain."""
        fn = feature_name.lower()
        if any(k in fn for k in ["creatinine", "bun", "egfr", "kidney", "renal", "proteinuria"]):
            return "RENAL"
        if any(k in fn for k in ["systolic", "diastolic", "bp", "heart_rate", "pulse", "cardiac"]):
            return "CARDIOVASCULAR"
        if any(k in fn for k in ["glucose", "sugar", "cholesterol", "hba1c", "bmi", "metabolic"]):
            return "METABOLIC"
        if any(k in fn for k in ["hemoglobin", "wbc", "platelet", "rbc", "hematocrit", "hematological"]):
            return "HEMATOLOGICAL"
        if any(k in fn for k in ["oxygen", "spo2", "respiratory", "pulmonary", "fev1"]):
            return "RESPIRATORY"
        if any(k in fn for k in ["neuro", "seizure", "tremor", "cognitive"]):
            return "NEUROLOGICAL"
        return "GENERAL"

    def _match_pattern(self, domains_detected: List[str], is_anomaly: int) -> Dict[str, Any]:
        """Match detected organ domain combinations with recognizable clinical pattern."""
        d_set = set(domains_detected)
        
        if {"CARDIOVASCULAR", "RENAL"}.issubset(d_set):
            return self.CLINICAL_PATTERNS["CARDIO_RENAL_SYNDROME_LIKE"]
        if {"METABOLIC", "HEMATOLOGICAL"}.issubset(d_set):
            return self.CLINICAL_PATTERNS["METABOLIC_HEMATOLOGICAL_PATTERN"]
        if len(d_set) >= 3:
            return self.CLINICAL_PATTERNS["MULTI_ORGAN_INFLAMMATORY_PATTERN"]
        if "RENAL" in d_set:
            return self.CLINICAL_PATTERNS["ISOLATED_RENAL_OUTLIER"]
        if "CARDIOVASCULAR" in d_set:
            return self.CLINICAL_PATTERNS["ISOLATED_CARDIOVASCULAR_OUTLIER"]
        if "HEMATOLOGICAL" in d_set:
            return self.CLINICAL_PATTERNS["ISOLATED_HEMATOLOGICAL_OUTLIER"]
        
        return self.CLINICAL_PATTERNS["GENERAL_ANOMALOUS_PROFILE"]
