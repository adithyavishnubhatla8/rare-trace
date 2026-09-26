"""
models/clinical_decision_support.py
=============================================================================
RARETRACE — Medical Suggestions, Clinical Features & Specialist Recommendation
=============================================================================
Academic Healthcare Machine-Learning Decision Support Prototype.

STRICT MEDICAL SAFETY NOTICE:
RARETRACE is an academic prototype. This module analyzes detected clinical
abnormalities and provides structured, non-prescriptive suggestions for clinician
review. It does NOT autonomously diagnose diseases, prescribe medications, specify
drug dosages, or schedule surgery. All final medical decisions must be made by
qualified healthcare professionals.
=============================================================================
"""

import json
import numpy as np
import pandas as pd


class ClinicalDecisionSupportEngine:
    """
    Analyzes patient clinical profiles, DBSCAN cluster status, anomaly scores,
    and Z-score feature deviations to generate structured, non-prescriptive
    medical suggestions and specialist recommendations.
    """

    MANDATORY_DISCLAIMER = (
        "This is a clinical decision-support suggestion, not a medical diagnosis or prescription. "
        "Final diagnosis, medication, treatment, and surgical decisions must be made by a qualified healthcare professional."
    )

    EMERGENCY_NOTICE = (
        "URGENT CLINICAL REVIEW MAY BE APPROPRIATE. If the patient has severe, sudden, or worsening symptoms, "
        "seek immediate medical attention or contact local emergency services rather than relying on this application."
    )

    # Standard clinical domain feature keywords
    DOMAIN_KEYWORD_MAP = {
        "RENAL": ["creatinine", "bun", "egfr", "proteinuria", "kidney", "renal", "albumin_creatinine", "cystatin"],
        "CARDIOVASCULAR": ["systolic_bp", "diastolic_bp", "heart_rate", "pulse", "bp_abnormality", "troponin", "cardiac", "blood_pressure", "sbp", "dbp", "hr"],
        "METABOLIC": ["glucose", "cholesterol", "hba1c", "bmi", "triglycerides", "ldl", "hdl", "metabolic_risk", "weight", "sugar"],
        "HEMATOLOGICAL": ["hemoglobin", "wbc", "platelets", "rbc", "hematocrit", "neutrophils", "lymphocytes", "hematological_risk", "mcv", "mch", "plt"],
        "RESPIRATORY": ["oxygen_saturation", "spo2", "respiratory_rate", "body_temp", "temperature", "fev1", "o2_sat", "resp_rate"],
        "NEUROLOGICAL": ["neurological", "seizure", "tremor", "cognitive", "neuro"],
    }

    # Specialist mapping rules
    SPECIALIST_MAP = {
        "RENAL": "Nephrologist",
        "CARDIOVASCULAR": "Cardiologist",
        "HEMATOLOGICAL": "Hematologist",
        "METABOLIC": "Endocrinologist",
        "RESPIRATORY": "Pulmonologist",
        "NEUROLOGICAL": "Neurologist",
        "GENERAL": "General Physician / Internal Medicine Specialist"
    }

    # Clinical feature explanation knowledge base (MedlinePlus / NIH / WHO aligned)
    FEATURE_EXPLANATION_LIBRARY = {
        "creatinine": {
            "what_it_means": "Creatinine is a waste byproduct of muscle breakdown cleared exclusively by the kidneys. It serves as a primary biomarker of renal filtration efficiency.",
            "clinical_area": "Renal",
            "suggested_action": "Consider clinical review and kidney-function assessment (serum creatinine, BUN, eGFR, and urinalysis)."
        },
        "bun": {
            "what_it_means": "Blood Urea Nitrogen (BUN) measures urea nitrogen concentration in blood, reflecting renal clearance and protein catabolism balance.",
            "clinical_area": "Renal",
            "suggested_action": "Consider evaluating renal perfusion, hydration status, and comprehensive renal metabolic panel."
        },
        "hemoglobin": {
            "what_it_means": "Hemoglobin is an iron-containing protein in red blood cells essential for transporting oxygen from the lungs to peripheral tissues.",
            "clinical_area": "Hematological",
            "suggested_action": "Consider medical evaluation of the abnormal blood count, peripheral blood smear, and hematinic indices."
        },
        "wbc": {
            "what_it_means": "White Blood Cells (leukocytes) are core cellular mediators of the immune system responding to infection, inflammation, or bone marrow shifts.",
            "clinical_area": "Hematological",
            "suggested_action": "Consider differential leukocyte count, inflammatory marker evaluation, and confirmatory hematological review."
        },
        "platelets": {
            "what_it_means": "Platelets (thrombocytes) are specialized cellular fragments vital for primary hemostasis, coagulation initiation, and vascular repair.",
            "clinical_area": "Hematological",
            "suggested_action": "Consider repeat platelet count with manual smear inspection and clinical bleeding/thrombosis risk evaluation."
        },
        "systolic_bp": {
            "what_it_means": "Systolic blood pressure quantifies the peak hydrostatic pressure exerted against arterial walls during ventricular myocardial contraction.",
            "clinical_area": "Cardiovascular",
            "suggested_action": "Consider repeated blood-pressure measurement with validated equipment and clinical cardiovascular risk assessment."
        },
        "diastolic_bp": {
            "what_it_means": "Diastolic blood pressure represents baseline resting arterial vascular resistance between cardiac beats during ventricular refill.",
            "clinical_area": "Cardiovascular",
            "suggested_action": "Consider evaluating vascular resistance, lifestyle factors, and repeat hemodynamic measurements."
        },
        "heart_rate": {
            "what_it_means": "Heart rate (pulse) measures the frequency of cardiac ventricular contractions per minute.",
            "clinical_area": "Cardiovascular",
            "suggested_action": "Consider 12-lead resting electrocardiogram (ECG) and rhythm assessment if persistent deviations exist."
        },
        "glucose": {
            "what_it_means": "Circulating blood glucose serves as the essential cellular carbohydrate energy source, regulated by pancreatic insulin and glucagon secretion.",
            "clinical_area": "Metabolic",
            "suggested_action": "Consider fasting plasma glucose, HbA1c testing, and comprehensive metabolic clinical assessment."
        },
        "cholesterol": {
            "what_it_means": "Cholesterol is an essential structural sterol lipid; abnormal circulating fractions correlate with vascular and cardiovascular risk.",
            "clinical_area": "Metabolic",
            "suggested_action": "Consider fasting lipid subfraction profile (LDL-C, HDL-C, triglycerides) and lifestyle cardiovascular review."
        },
        "oxygen_saturation": {
            "what_it_means": "Oxygen saturation (SpO2) measures the percentage of hemoglobin binding sites occupied by oxygen in circulating blood.",
            "clinical_area": "Respiratory",
            "suggested_action": "Consider continuous pulse oximetry monitoring, evaluation of oxygenation on exertion, and pulmonary assessment."
        },
        "body_temp": {
            "what_it_means": "Core body temperature reflects internal thermoregulatory equilibrium between metabolic heat generation and environmental dissipation.",
            "clinical_area": "Respiratory / Systemic",
            "suggested_action": "Consider clinical evaluation for underlying inflammatory, infectious, or thermoregulatory etiologies."
        },
        "symptom_count": {
            "what_it_means": "Total cumulative burden of active symptoms reported or documented for the patient.",
            "clinical_area": "General Medicine",
            "suggested_action": "Consider comprehensive holistic review of multi-system symptom progression and historical baseline records."
        },
        "hospital_visits": {
            "what_it_means": "Frequency of acute medical encounters, indicating chronic disease instability or clinical management complexity.",
            "clinical_area": "General Medicine",
            "suggested_action": "Consider structured care coordination, medication reconciliation, and primary care follow-up."
        }
    }

    # Authoritative reference library
    AUTHORITATIVE_REFERENCES = {
        "RENAL": {
            "title": "KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease",
            "organization": "KDIGO / Kidney International",
            "summary": "Evidence-based guidelines recommending serum creatinine, eGFR staging, and albuminuria evaluation for renal abnormalities.",
            "url": "https://kdigo.org/guidelines/ckd-evaluation-and-management/"
        },
        "CARDIOVASCULAR": {
            "title": "2023 ACC/AHA Guideline for the Prevention and Management of High Blood Pressure",
            "organization": "American College of Cardiology / American Heart Association",
            "summary": "Standardized clinical protocols for multi-measurement BP confirmation and cardiovascular risk stratification.",
            "url": "https://www.acc.org/guidelines"
        },
        "METABOLIC": {
            "title": "American Diabetes Association Standards of Care in Diabetes — 2024",
            "organization": "American Diabetes Association (ADA)",
            "summary": "Clinical criteria for fasting plasma glucose, HbA1c assessment, and comprehensive metabolic risk evaluation.",
            "url": "https://diabetesjournals.org/care"
        },
        "HEMATOLOGICAL": {
            "title": "American Society of Hematology (ASH) Clinical Practice Guidelines",
            "organization": "American Society of Hematology",
            "summary": "Diagnostic approach to cytopenias, leukocytosis, and abnormal hematological indices requiring confirmatory smear.",
            "url": "https://www.hematology.org/education/clinicians/guidelines-and-quality-care"
        },
        "RESPIRATORY": {
            "title": "WHO Clinical Management Guidelines for Respiratory and Oxygenation Impairments",
            "organization": "World Health Organization (WHO)",
            "summary": "Clinical protocols for continuous pulse oximetry monitoring and hypoxemia clinical escalation.",
            "url": "https://www.who.int/publications"
        },
        "GENERAL": {
            "title": "NIH Clinical Center Diagnostic and Patient Care Guidelines",
            "organization": "National Institutes of Health (NIH)",
            "summary": "Systematic approach to evaluating complex clinical outliers and multi-system clinical presentations.",
            "url": "https://clinicalcenter.nih.gov/"
        }
    }

    # Safe general lifestyle and health considerations
    LIFESTYLE_CONSIDERATIONS = [
        {
            "category": "Hydration & Fluid Intake",
            "recommendation": "Maintain appropriate daily fluid intake unless clinically advised otherwise.",
            "rationale": "Supports renal perfusion, hemodynamic regulation, and electrolyte homeostasis."
        },
        {
            "category": "Nutrition & Dietary Balance",
            "recommendation": "Follow balanced nutritional guidelines and discuss dietary factors with a clinician.",
            "rationale": "Optimizing sodium, potassium, and glycemic intake benefits cardiovascular and metabolic health."
        },
        {
            "category": "Vital Sign Self-Monitoring",
            "recommendation": "Periodically record resting blood pressure, pulse, and temperature as recommended.",
            "rationale": "Establishing reliable longitudinal trends helps physicians evaluate clinical stability."
        },
        {
            "category": "Symptom Logging & Tracking",
            "recommendation": "Maintain a systematic diary of fatigue, exertion tolerance, or acute symptom changes.",
            "rationale": "Timely documentation of multi-system changes accelerates accurate clinical assessment."
        }
    ]

    def __init__(self, z_threshold_moderate=2.0, z_threshold_extreme=3.0):
        self.z_moderate = z_threshold_moderate
        self.z_extreme = z_threshold_extreme

    def identify_feature_domain(self, feature_name):
        """Map an arbitrary feature name to a clinical organ domain."""
        clean_name = feature_name.lower().replace(" ", "_")
        for domain, keywords in self.DOMAIN_KEYWORD_MAP.items():
            for kw in keywords:
                if kw in clean_name:
                    return domain
        return "GENERAL"

    def get_feature_explanation(self, feature_name, z_score):
        """Retrieve plain-English clinical explanation, rationale, and suggested action for a feature."""
        clean_name = feature_name.lower().replace(" ", "_")
        found_info = None
        for key, info in self.FEATURE_EXPLANATION_LIBRARY.items():
            if key in clean_name:
                found_info = info
                break

        formatted_name = feature_name.replace("_", " ").title()
        if found_info:
            what_it_means = found_info["what_it_means"]
            clinical_area = found_info["clinical_area"]
            suggested_action = found_info["suggested_action"]
        else:
            what_it_means = f"{formatted_name} is a quantitative physiological or laboratory parameter recorded in the clinical cohort."
            clinical_area = self.identify_feature_domain(feature_name).title()
            suggested_action = f"Consider clinical review of abnormal {formatted_name.lower()} in context with the patient's complete history."

        severity = "Severe Deviation" if abs(z_score) >= self.z_extreme else "Moderate Shift"
        direction = "higher than" if z_score > 0 else "lower than"
        why_flagged = f"The patient's value deviates by {abs(z_score):.1f} standard deviations (Z-score: {z_score:+.2f}) from the reference cohort baseline ({direction} expected mean)."

        return {
            "feature": feature_name,
            "feature_label": formatted_name,
            "what_it_means": what_it_means,
            "why_flagged": why_flagged,
            "suggested_action": suggested_action,
            "clinical_area": clinical_area,
            "severity": severity,
            "z_score": round(z_score, 2)
        }

    def detect_emergency_vitals(self, row):
        """Check for critical vital signs that warrant immediate clinical attention."""
        emergency_triggers = []
        for col, val in row.items():
            if not isinstance(val, (int, float, np.number)) or pd.isna(val):
                continue
            clean = col.lower().replace(" ", "_")
            val_num = float(val)

            if any(k in clean for k in ["oxygen_saturation", "spo2", "o2_sat"]) and val_num < 90.0:
                emergency_triggers.append(f"Low Oxygen Saturation ({val_num:.1f}%)")
            elif any(k in clean for k in ["systolic_bp", "sbp"]) and val_num >= 180.0:
                emergency_triggers.append(f"Markedly Elevated Systolic BP ({val_num:.0f} mmHg)")
            elif any(k in clean for k in ["diastolic_bp", "dbp"]) and val_num >= 120.0:
                emergency_triggers.append(f"Markedly Elevated Diastolic BP ({val_num:.0f} mmHg)")
            elif any(k in clean for k in ["heart_rate", "pulse", "hr"]) and (val_num >= 135.0 or val_num <= 40.0):
                emergency_triggers.append(f"Extreme Heart Rate ({val_num:.0f} bpm)")
            elif any(k in clean for k in ["body_temp", "temperature"]) and (val_num >= 39.5 or val_num <= 35.0):
                emergency_triggers.append(f"Extreme Body Temperature ({val_num:.1f} C)")

        is_emg = len(emergency_triggers) > 0
        crit_vitals = []
        for t in emergency_triggers:
            try:
                f_name = t.split(" (")[0]
                f_val = t.split(" (")[1].rstrip(")")
            except Exception:
                f_name = t
                f_val = "Critical"
            crit_vitals.append({
                "feature": f_name,
                "value": f_val,
                "threshold": "Critical Range"
            })

        return {
            "is_emergency": is_emg,
            "has_emergency_signals": is_emg,
            "triggers": emergency_triggers,
            "critical_vitals": crit_vitals,
            "emergency_notice": self.EMERGENCY_NOTICE
        }

    def determine_specialist_hierarchy(self, all_domains, multi_system_signal=0, multi_system=None):
        """
        Determine hierarchical doctor recommendation:
        Multi-system -> General Physician / Internal Medicine first, then specific subspecialists.
        Single domain -> Targeted subspecialist first.
        """
        is_multi = (multi_system is True) or (multi_system_signal == 1 or multi_system_signal is True)
        if is_multi:
            primary = "General Physician / Internal Medicine Specialist"
            hierarchy_type = "MULTI_SYSTEM_GENERAL_FIRST"
            raw_specs = []
            for d in all_domains:
                spec = self.SPECIALIST_MAP.get(d)
                if spec and spec not in raw_specs and spec != primary:
                    raw_specs.append(spec)
            guidance = (
                "Because abnormalities span multiple distinct organ systems, comprehensive evaluation by a "
                "General Physician / Internal Medicine Specialist is recommended first, with targeted subspecialty referrals "
                "coordinated based on clinical assessment."
            )
        elif len(all_domains) == 1:
            dom = all_domains[0]
            primary = self.SPECIALIST_MAP.get(dom, "General Physician / Internal Medicine Specialist")
            hierarchy_type = "TARGETED_ORGAN_SPECIALIST"
            raw_specs = ["General Physician / Internal Medicine Specialist"] if primary != "General Physician / Internal Medicine Specialist" else []
            guidance = f"Findings are predominantly localized to the {dom.title()} domain; evaluation by a {primary} may be considered."
        else:
            primary = "General Physician / Internal Medicine Specialist"
            hierarchy_type = "GENERAL_PRIMARY_CARE"
            raw_specs = []
            guidance = "Routine evaluation by a General Physician or primary healthcare provider is recommended."

        add_objs = [{"specialty": s, "rationale": "Subspecialty review for observed clinical deviations"} for s in raw_specs]
        return {
            "primary_specialist": primary,
            "hierarchy_type": hierarchy_type,
            "additional_specialists": add_objs,
            "additional_specialists_list": raw_specs,
            "specialist_guidance": guidance
        }

    def analyze_patient(self, row, pop_means=None, pop_stds=None, dataset_id="ds_default", run_id=None):
        """
        Generate structured medical suggestions, abnormal feature explanations,
        and specialist recommendations for a single patient record.
        """
        patient_id = str(row.get("patient_id", "Unknown"))
        raw_score = row.get("anomaly_score")
        anomaly_score = float(raw_score) if raw_score is not None and not pd.isna(raw_score) else 0.0
        raw_cluster = row.get("cluster_label")
        cluster_label = int(raw_cluster) if raw_cluster is not None and not pd.isna(raw_cluster) else 0
        raw_anomaly = row.get("is_anomaly")
        is_anomaly = int(raw_anomaly) if raw_anomaly is not None and not pd.isna(raw_anomaly) else 0

        # 1. Detect Z-score deviations across available numeric features
        z_drivers = []
        domain_counts = {}
        domain_features = {}
        abnormal_feature_details = []

        for col, val in row.items():
            if col in ["patient_id", "cluster_label", "is_anomaly", "anomaly_score", "run_id", "dataset_id"]:
                continue
            if isinstance(val, (int, float, np.number)) and not pd.isna(val):
                if pop_means is not None and col in pop_means and pop_stds is not None and col in pop_stds:
                    mean_val = pop_means[col]
                    std_val = pop_stds[col]
                    mean = float(mean_val) if not pd.isna(mean_val) else float(val)
                    std = float(std_val) if not pd.isna(std_val) else 0.0
                    z = (float(val) - mean) / std if std > 1e-6 else 0.0
                else:
                    z = 0.0

                abs_z = abs(z)
                if abs_z >= self.z_moderate:
                    domain = self.identify_feature_domain(col)
                    direction = "Significantly Elevated" if z > 0 else "Significantly Depressed"
                    severity = "Highly unusual" if abs_z >= self.z_extreme else "Unusual"
                    
                    explanation_info = self.get_feature_explanation(col, z)
                    explanation_info["patient_value"] = round(float(val), 2)
                    explanation_info["pop_mean"] = round(mean, 2) if pop_means is not None and col in pop_means else None
                    explanation_info["z_score"] = round(z, 2)
                    explanation_info["direction"] = direction
                    explanation_info["domain"] = domain
                    abnormal_feature_details.append(explanation_info)

                    z_drivers.append({
                        "feature": col,
                        "feature_label": col.replace("_", " ").title(),
                        "patient_value": round(float(val), 2),
                        "pop_mean": round(mean, 2) if pop_means is not None and col in pop_means else None,
                        "z_score": round(z, 2),
                        "direction": direction,
                        "severity": severity,
                        "domain": domain
                    })

                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    domain_features.setdefault(domain, []).append(col.replace("_", " ").title())

        # Sort Z-score drivers and feature details by absolute magnitude
        z_drivers.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        abnormal_feature_details.sort(key=lambda x: abs(x["z_score"]), reverse=True)

        # 2. Determine involved clinical domains
        all_domains = list(domain_counts.keys())
        if not all_domains:
            primary_domain = "GENERAL"
        else:
            primary_domain = max(domain_counts, key=domain_counts.get)

        # 3. Multi-system signal detection (>= 2 organ systems with |Z| >= 2.0)
        multi_system_signal = 1 if len(all_domains) >= 2 else 0

        # 4. Clinical Priority Level (Academic Review Urgency)
        extreme_drivers_count = sum(1 for d in z_drivers if abs(d["z_score"]) >= self.z_extreme)
        
        if anomaly_score >= 75.0 or extreme_drivers_count >= 2 or (multi_system_signal == 1 and is_anomaly == 1):
            clinical_priority = "HIGH"
        elif anomaly_score >= 40.0 or len(z_drivers) >= 2 or is_anomaly == 1:
            clinical_priority = "MEDIUM"
        else:
            clinical_priority = "LOW"

        # 5. Emergency Vitals Detection
        emergency_data = self.detect_emergency_vitals(row)

        # 6. Specialist Hierarchy
        specialist_info = self.determine_specialist_hierarchy(all_domains, multi_system_signal)

        # 7. Key Clinical Findings Summary
        key_findings = [f"{d['direction'].replace('Significantly ', '')} {d['feature_label']}" for d in z_drivers[:5]]
        if not key_findings:
            key_findings = ["Measurements within expected reference cohort range"]

        # 8. Medical Suggestions (Phrased with 'Consider', 'Discuss with clinician')
        medical_suggestions = []
        if "RENAL" in all_domains:
            medical_suggestions.append("Review the abnormal renal-related laboratory findings with a qualified physician.")
            medical_suggestions.append("Consider repeat or confirmatory renal function testing (e.g. serum creatinine, BUN, eGFR).")
        if "CARDIOVASCULAR" in all_domains:
            medical_suggestions.append("Review blood-pressure and vital-sign measurements across multiple resting intervals.")
        if "HEMATOLOGICAL" in all_domains:
            medical_suggestions.append("Review abnormal blood-count parameters and consider complete blood count with manual differential.")
        if "METABOLIC" in all_domains:
            medical_suggestions.append("Review metabolic/glucose parameters and consider fasting glycemic assessment.")
        if "RESPIRATORY" in all_domains:
            medical_suggestions.append("Evaluate respiratory status and oxygen saturation response to baseline activities.")
        if multi_system_signal == 1:
            medical_suggestions.append("Discuss the combined multi-system findings with a General Physician or Internal Medicine specialist.")
        if not medical_suggestions:
            medical_suggestions.append("Discuss routine wellness parameters and healthy lifestyle maintenance with a healthcare provider.")

        # 9. Investigation Suggestions
        investigation_suggestions = []
        if "RENAL" in all_domains:
            investigation_suggestions.append("Consider clinician-directed evaluation of kidney function, relevant renal laboratory parameters, blood pressure, and urinalysis.")
        if "HEMATOLOGICAL" in all_domains:
            investigation_suggestions.append("Consider repeat confirmatory complete blood count, evaluation of red/white cell indices, and peripheral blood smear if indicated.")
        if "CARDIOVASCULAR" in all_domains:
            investigation_suggestions.append("Consider repeat blood-pressure measurement with validated equipment, 12-lead ECG, and cardiovascular risk assessment.")
        if "METABOLIC" in all_domains:
            investigation_suggestions.append("Consider fasting glucose assessment, HbA1c, comprehensive lipid panel, and review of relevant metabolic risk factors.")
        if "RESPIRATORY" in all_domains:
            investigation_suggestions.append("Consider continuous pulse oximetry monitoring and formal pulmonary clinical evaluation.")
        if not investigation_suggestions:
            investigation_suggestions.append("Consider periodic routine preventive clinical screening.")

        # 10. Medication Review Considerations
        med_count = None
        for k, v in row.items():
            if "medication" in k.lower() and "count" in k.lower():
                try:
                    med_count = int(v)
                except Exception:
                    pass
                break

        medication_notes = (
            "A qualified clinician should review the patient's current medications, possible medication-related effects, "
            "drug-drug interactions, dose appropriateness, and kidney/liver clearance considerations. "
            "Do not start, stop, or alter prescription medications based solely on RARETRACE results."
        )
        if med_count is not None and med_count >= 5:
            medication_notes += f" Note: Patient has {med_count} recorded medications (polypharmacy threshold reached)."

        # 11. Procedure / Surgical Evaluation Considerations
        procedure_notes = (
            "No automatic surgical or invasive procedure recommendation is made. "
            "If clinically indicated, a qualified medical specialist must evaluate the patient's complete clinical trajectory "
            "before determining whether additional diagnostic procedures or interventional evaluation is appropriate."
        )

        # 12. Possible Clinical Areas for Discussion (Carefully Phrased)
        possible_areas = []
        for dom in all_domains:
            if dom == "RENAL":
                possible_areas.append("The observed pattern may warrant evaluation for conditions affecting renal clearance or kidney function.")
            elif dom == "HEMATOLOGICAL":
                possible_areas.append("The observed findings may warrant evaluation for hematological balance or cell production shifts.")
            elif dom == "CARDIOVASCULAR":
                possible_areas.append("The observed pattern may warrant assessment for hemodynamic or vascular regulation factors.")
            elif dom == "METABOLIC":
                possible_areas.append("The observed findings may warrant discussion regarding glycemic regulation and metabolic homeostasis.")
            elif dom == "RESPIRATORY":
                possible_areas.append("The observed parameters may warrant discussion regarding respiratory oxygenation stability.")

        # 13. Medical References
        medical_references = []
        for dom in all_domains:
            if dom in self.AUTHORITATIVE_REFERENCES:
                medical_references.append(self.AUTHORITATIVE_REFERENCES[dom])
        if not medical_references:
            medical_references.append(self.AUTHORITATIVE_REFERENCES["GENERAL"])

        return {
            "dataset_id": dataset_id,
            "patient_id": patient_id,
            "analysis_run_id": run_id,
            "clinical_priority": clinical_priority,
            "primary_domain": primary_domain,
            "all_domains": all_domains,
            "multi_system_signal": multi_system_signal,
            "key_findings": key_findings,
            "abnormal_feature_details": abnormal_feature_details,
            "z_score_drivers": z_drivers,
            "medical_suggestions": medical_suggestions,
            "investigation_suggestions": investigation_suggestions,
            "primary_specialist": specialist_info["primary_specialist"],
            "additional_specialists": specialist_info["additional_specialists"],
            "specialist_guidance": specialist_info["specialist_guidance"],
            "medication_review_notes": medication_notes,
            "procedure_review_notes": procedure_notes,
            "lifestyle_considerations": self.LIFESTYLE_CONSIDERATIONS,
            "possible_clinical_areas": possible_areas,
            "emergency_data": emergency_data,
            "medical_references": medical_references,
            "anomaly_score": anomaly_score,
            "cluster_label": cluster_label,
            "is_anomaly": is_anomaly,
            "disclaimer": self.MANDATORY_DISCLAIMER,
            "emergency_notice": self.EMERGENCY_NOTICE
        }

    def generate_cohort_suggestions(self, df_analyzed, candidate_details=None, pop_means=None, pop_stds=None, dataset_id="ds_default", run_id=None):
        """
        Generate structured clinical decision support suggestions for an entire dataset cohort.
        """
        suggestions = []
        for _, row in df_analyzed.iterrows():
            suggestion = self.analyze_patient(
                row=row,
                pop_means=pop_means,
                pop_stds=pop_stds,
                dataset_id=dataset_id,
                run_id=run_id
            )
            suggestions.append(suggestion)
        return suggestions

    def generate_detailed_recommendations(self, df_analyzed, candidate_details=None, pop_means=None, pop_stds=None, dataset_id="ds_default", run_id=None):
        """
        Generate flat records for the clinical_recommendations table.
        One record per abnormal feature per patient (or a summary record if no features abnormal).
        """
        recommendations = []
        for _, row in df_analyzed.iterrows():
            p_sug = self.analyze_patient(
                row=row,
                pop_means=pop_means,
                pop_stds=pop_stds,
                dataset_id=dataset_id,
                run_id=run_id
            )
            patient_id = p_sug["patient_id"]
            priority = p_sug["clinical_priority"]
            primary_spec = p_sug["primary_specialist"]
            add_specs = p_sug["additional_specialists"]
            med_review = p_sug["medication_review_notes"]
            proc_review = p_sug["procedure_review_notes"]
            first_ref = p_sug["medical_references"][0]["title"] if p_sug["medical_references"] else "NIH Clinical Guidelines"

            if p_sug["abnormal_feature_details"]:
                for feat_info in p_sug["abnormal_feature_details"]:
                    recommendations.append({
                        "dataset_id": dataset_id,
                        "patient_id": patient_id,
                        "analysis_run_id": run_id,
                        "clinical_domain": feat_info["clinical_area"],
                        "abnormal_feature": feat_info["feature_label"],
                        "z_score": feat_info["z_score"],
                        "severity": feat_info["severity"],
                        "clinical_explanation": f"{feat_info['what_it_means']} {feat_info['why_flagged']}",
                        "medical_suggestion": feat_info["suggested_action"],
                        "investigation_suggestion": feat_info["suggested_action"],
                        "primary_specialist": primary_spec,
                        "additional_specialists": add_specs,
                        "medication_review": med_review,
                        "procedure_review": proc_review,
                        "priority": priority,
                        "evidence_reference": first_ref
                    })
            else:
                # Standard profile record
                recommendations.append({
                    "dataset_id": dataset_id,
                    "patient_id": patient_id,
                    "analysis_run_id": run_id,
                    "clinical_domain": "GENERAL",
                    "abnormal_feature": "Expected Variation",
                    "z_score": 0.0,
                    "severity": "Expected Profile",
                    "clinical_explanation": "All quantitative features align within reference population bounds.",
                    "medical_suggestion": "Continue routine clinical monitoring and preventive healthcare.",
                    "investigation_suggestion": "Standard periodic wellness assessment.",
                    "primary_specialist": primary_spec,
                    "additional_specialists": add_specs,
                    "medication_review": med_review,
                    "procedure_review": proc_review,
                    "priority": priority,
                    "evidence_reference": first_ref
                })

        return recommendations
