import os
import io
import re
import pandas as pd
import numpy as np
from pathlib import Path
from fpdf import FPDF
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config
from database.db import db_manager

class AnalysisReportGenerator:
    """
    Generates downloadable analysis reports in CSV and PDF formats strictly scoped
    to a single independent dataset, featuring the dataset name, source, and cohort metrics.
    """
    @staticmethod
    def generate_csv_report(dataset_id="ds_default"):
        """Export complete analyzed dataset as a CSV file buffer strictly for dataset_id."""
        df_all = db_manager.get_full_patient_analysis(dataset_id)
        
        if "anomaly_score" in df_all.columns:
            df_all = df_all.sort_values("anomaly_score", ascending=False)

        buffer = io.BytesIO()
        df_all.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    @staticmethod
    def generate_pdf_report(dataset_id="ds_default"):
        """Generate a comprehensive 2-page PDF report strictly for dataset_id."""
        dataset = db_manager.get_dataset(dataset_id) or {
            "dataset_id": dataset_id,
            "dataset_name": "Clinical Dataset",
            "source": "Standard Cohort",
            "filename": "patients.csv"
        }
        df_all = db_manager.get_full_patient_analysis(dataset_id)
        latest_run = db_manager.get_latest_model_run(dataset_id)

        total_patients = len(df_all)
        candidate_cases = int(np.sum(df_all["is_anomaly"] == 1)) if "is_anomaly" in df_all.columns else 0
        anomaly_pct = round((candidate_cases / max(1, total_patients)) * 100.0, 1)

        high_cnt = int(np.sum(df_all["review_priority"] == "High Priority")) if "review_priority" in df_all.columns else 0
        med_cnt = int(np.sum(df_all["review_priority"] == "Medium Priority")) if "review_priority" in df_all.columns else 0
        low_cnt = int(np.sum(df_all["review_priority"] == "Low Priority")) if "review_priority" in df_all.columns else 0

        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # ==========================================
        # PAGE 1: EXECUTIVE SUMMARY & TOP CANDIDATES
        # ==========================================
        pdf.add_page()

        # Insert RARETRACE Logo if file exists
        logo_path = Config.BASE_DIR / "static" / "images" / "logo.png"
        if logo_path.exists():
            pdf.image(str(logo_path), x=14, y=10, w=18)

        # Document Header
        pdf.set_font("Helvetica", "B", 17)
        pdf.set_text_color(2, 132, 199) # Medical Blue accent
        pdf.cell(0, 8, "RARETRACE CLINICAL ANALYSIS REPORT", ln=True, align="C")
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 6, f"Dataset: {dataset.get('dataset_name', 'Cohort')} (Source: {dataset.get('source', 'Unknown')})", ln=True, align="C")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, f"Source File: {dataset.get('filename', 'dataset.csv')} | Dataset ID: {dataset_id}", ln=True, align="C")
        pdf.ln(5)

        # Executive Summary Box
        pdf.set_fill_color(248, 250, 252)
        pdf.rect(10, pdf.get_y(), 190, 28, style="F")
        pdf.set_xy(12, pdf.get_y() + 2)
        
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(90, 6, f"Total Cohort Records: {total_patients}", ln=False)
        pdf.cell(90, 6, f"Candidate Rare Cases (Noise): {candidate_cases} ({anomaly_pct}%)", ln=True)
        
        pdf.set_x(12)
        pdf.cell(90, 6, f"DBSCAN Epsilon (eps): {latest_run.get('eps', 'N/A')}", ln=False)
        pdf.cell(90, 6, f"MinPts (min_samples): {latest_run.get('min_samples', 'N/A')}", ln=True)

        pdf.set_x(12)
        pdf.cell(90, 6, f"Clusters Discovered: {latest_run.get('number_of_clusters', 'N/A')}", ln=False)
        pdf.cell(90, 6, f"Silhouette Score: {latest_run.get('silhouette_score', 'N/A')}", ln=True)
        pdf.ln(10)

        # Review Priority Breakdown Table
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 7, f"Candidate Review Priority Distribution ({dataset.get('dataset_name', 'Cohort')})", ln=True)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(2, 132, 199)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(63, 7, "High Priority", border=1, align="C", fill=True)
        pdf.cell(63, 7, "Medium Priority", border=1, align="C", fill=True)
        pdf.cell(64, 7, "Low Priority", border=1, align="C", fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(63, 7, f"{high_cnt} cases", border=1, align="C")
        pdf.cell(63, 7, f"{med_cnt} cases", border=1, align="C")
        pdf.cell(64, 7, f"{low_cnt} cases", border=1, align="C")
        pdf.ln(9)

        # Top Candidate Rare Cases Table
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Identified Candidate Rare Cases (Top Density Outliers)", ln=True)

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(30, 41, 59)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(25, 7, "Patient ID", border=1, fill=True)
        pdf.cell(20, 7, "Score", border=1, fill=True, align="C")
        pdf.cell(30, 7, "Priority", border=1, fill=True, align="C")
        pdf.cell(50, 7, "Key Abnormal Features", border=1, fill=True)
        pdf.cell(65, 7, "Clinical Interpretation", border=1, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(15, 23, 42)

        df_anomalies = df_all[df_all["is_anomaly"] == 1].sort_values("anomaly_score", ascending=False).head(15)

        for idx, row in df_anomalies.iterrows():
            pid = str(row.get("patient_id", "N/A"))
            score = str(row.get("anomaly_score", "N/A"))
            prio = str(row.get("review_priority", "N/A"))
            feats = str(row.get("key_abnormal_features", "N/A"))[:30]
            interp = str(row.get("interpretation", "N/A"))[:42]

            pdf.cell(25, 6, pid, border=1)
            pdf.cell(20, 6, score, border=1, align="C")
            pdf.cell(30, 6, prio, border=1, align="C")
            pdf.cell(50, 6, feats, border=1)
            pdf.cell(65, 6, interp, border=1)
            pdf.ln()

        pdf.ln(5)
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(100, 116, 139)
        pdf.multi_cell(0, 4, "ACADEMIC DISCLAIMER: RARETRACE identifies statistically unusual patient profiles using unsupervised DBSCAN anomaly detection relative to this reference dataset and does not provide a medical diagnosis.")

        # ==========================================
        # PAGE 2: FEATURE SHIFT & DATASET SCHEMA
        # ==========================================
        pdf.add_page()

        if logo_path.exists():
            pdf.image(str(logo_path), x=14, y=10, w=18)

        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(2, 132, 199)
        pdf.cell(0, 8, "Feature Shift Analysis (Normal vs Candidate Cases)", ln=True, align="C")
        
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, f"Dataset: {dataset.get('dataset_name')} | Source: {dataset.get('source')}", ln=True, align="C")
        pdf.ln(5)

        numeric_cols = [
            c for c in df_all.columns 
            if pd.api.types.is_numeric_dtype(df_all[c]) and c not in ["cluster_label", "is_anomaly", "anomaly_score", "run_id"] and "id" not in c.lower()
        ]

        normal_mask = df_all["is_anomaly"] == 0
        anomaly_mask = df_all["is_anomaly"] == 1

        if not np.any(normal_mask):
            normal_mask = np.ones(len(df_all), dtype=bool)

        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 7, f"Cohort Statistical Baseline Profile ({len(df_all.columns)} Columns, {total_patients} Records)", ln=True)

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(2, 132, 199)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(55, 7, "Feature Name", border=1, fill=True)
        pdf.cell(32, 7, "Cohort Mean", border=1, align="C", fill=True)
        pdf.cell(32, 7, "Cohort StdDev", border=1, align="C", fill=True)
        pdf.cell(35, 7, "Anomaly Mean", border=1, align="C", fill=True)
        pdf.cell(36, 7, "Mean Shift (Delta)", border=1, align="C", fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(15, 23, 42)

        for col in numeric_cols[:18]:
            c_mean = float(df_all.loc[normal_mask, col].mean()) if len(df_all.loc[normal_mask, col]) > 0 else 0.0
            c_std = float(df_all.loc[normal_mask, col].std()) if len(df_all.loc[normal_mask, col]) > 0 else 1.0
            a_mean = float(df_all.loc[anomaly_mask, col].mean()) if np.any(anomaly_mask) else c_mean
            shift = a_mean - c_mean

            col_label = col.replace('_', ' ').title()[:24]
            pdf.cell(55, 6, col_label, border=1)
            pdf.cell(32, 6, f"{c_mean:.2f}", border=1, align="C")
            pdf.cell(32, 6, f"{c_std:.2f}", border=1, align="C")
            pdf.cell(35, 6, f"{a_mean:.2f}", border=1, align="C")
            pdf.cell(36, 6, f"{'+' if shift > 0 else ''}{shift:.2f}", border=1, align="C")
            pdf.ln()

        pdf.ln(8)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Independent Pipeline Integrity Assurance", ln=True)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.multi_cell(0, 4.5, f"All statistics, cluster assignments, anomaly scores, and feature shifts in this document were computed strictly relative to dataset '{dataset.get('dataset_name')}' (Source: {dataset.get('source')}). No records from other datasets were aggregated, merged, or referenced.")

        # ==========================================
        # PAGE 3: CLINICAL DECISION SUPPORT & MEDICAL RECOMMENDATIONS
        # ==========================================
        pdf.add_page()

        if logo_path.exists():
            pdf.image(str(logo_path), x=14, y=10, w=18)

        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(2, 132, 199)
        pdf.cell(0, 8, "Clinical Decision Support & Review Considerations", ln=True, align="C")

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, f"Dataset: {dataset.get('dataset_name')} | Source: {dataset.get('source')}", ln=True, align="C")
        pdf.ln(3)

        # Prominent Emergency & Medical Safety Disclaimer
        pdf.set_fill_color(255, 241, 242)
        pdf.set_text_color(159, 18, 57)
        pdf.set_font("Helvetica", "B", 7)
        pdf.multi_cell(0, 4, "EMERGENCY SAFETY NOTICE: URGENT CLINICAL REVIEW MAY BE APPROPRIATE. If the patient has severe, sudden, or worsening symptoms, seek immediate medical attention or contact local emergency services rather than relying on this application.", border=1, fill=True)
        pdf.ln(2)

        pdf.set_fill_color(254, 243, 199)
        pdf.set_text_color(146, 64, 14)
        pdf.set_font("Helvetica", "B", 7)
        pdf.multi_cell(0, 4, "MANDATORY MEDICAL DISCLAIMER: This is a clinical decision-support suggestion, not a medical diagnosis or prescription. Final diagnosis, medication, treatment, and surgical decisions must be made by a qualified healthcare professional.", border=1, fill=True)
        pdf.ln(4)

        suggestions = db_manager.get_clinical_suggestions(dataset_id=dataset_id)
        cds_summary = db_manager.get_clinical_suggestions_summary(dataset_id=dataset_id)

        # Summary KPIs
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(45, 6, f"Total Cases Reviewed: {cds_summary['total_reviewed']}", border=1)
        pdf.cell(45, 6, f"High Priority Reviews: {cds_summary['high_priority_count']}", border=1)
        pdf.cell(50, 6, f"Multi-System Involvements: {cds_summary['multi_system_count']}", border=1)
        pdf.cell(50, 6, f"Moderate Reviews: {cds_summary['medium_priority_count']}", border=1)
        pdf.ln(9)

        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 7, "Priority Candidate Clinical Dossiers & Investigation Suggestions", ln=True)

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(2, 132, 199)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(24, 7, "Patient ID", border=1, fill=True)
        pdf.cell(24, 7, "Priority", border=1, align="C", fill=True)
        pdf.cell(32, 7, "Domain(s)", border=1, fill=True)
        pdf.cell(60, 7, "Investigation Considerations", border=1, fill=True)
        pdf.cell(50, 7, "Specialist Referral", border=1, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(15, 23, 42)

        for s in suggestions[:8]:
            p_id = str(s.get("patient_id", ""))[:12]
            prio = str(s.get("clinical_priority", "LOW"))
            domains = ", ".join(s.get("all_domains", ["GENERAL"]))[:20]
            
            invs = s.get("investigation_suggestions", [])
            inv_text = invs[0] if invs else "Routine monitoring"
            inv_text = inv_text[:50] + ("..." if len(inv_text) > 50 else "")

            specs = s.get("specialist_referrals", [])
            spec_text = specs[0].get("specialty", "General Medicine") if (specs and isinstance(specs[0], dict)) else "General Medicine"
            spec_text = str(spec_text)[:30]

            pdf.cell(24, 6, p_id, border=1)
            pdf.cell(24, 6, prio, border=1, align="C")
            pdf.cell(32, 6, domains, border=1)
            pdf.cell(60, 6, inv_text, border=1)
            pdf.cell(50, 6, spec_text, border=1)
            pdf.ln()

        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, "Medication & Procedural Evaluation Policies", ln=True)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(100, 116, 139)
        pdf.multi_cell(0, 4, "- Medication Review: Qualified clinician reconciliation required. Assess renal/hepatic clearance, polypharmacy risks, and potential contraindications. No autonomous prescriptions are made.\n- Procedure/Surgery: No automatic surgical recommendation is made. Specialist clinical trajectory review must precede any invasive diagnostic or therapeutic procedures.")

        buffer = io.BytesIO()
        pdf.output(buffer)
        buffer.seek(0)
        return buffer

    @staticmethod
    def generate_clinical_suggestions_csv(dataset_id="ds_default"):
        """Export dataset-isolated clinical decision support suggestions as a CSV file buffer."""
        suggestions = db_manager.get_clinical_suggestions(dataset_id=dataset_id)
        rows = []
        for s in suggestions:
            invs = "; ".join(s.get("investigation_suggestions", []))
            specs = "; ".join([sp.get("specialty", "") if isinstance(sp, dict) else str(sp) for sp in s.get("specialist_referrals", [])])
            drivers = "; ".join([f"{d.get('feature_label')}: Z={d.get('z_score')}" for d in s.get("z_score_drivers", [])])
            rows.append({
                "dataset_id": s.get("dataset_id"),
                "patient_id": s.get("patient_id"),
                "analysis_run_id": s.get("analysis_run_id"),
                "clinical_priority": s.get("clinical_priority"),
                "primary_domain": s.get("primary_domain"),
                "all_domains": ", ".join(s.get("all_domains", [])),
                "multi_system_signal": s.get("multi_system_signal"),
                "z_score_drivers": drivers,
                "investigation_suggestions": invs,
                "specialist_referrals": specs,
                "medication_review_notes": s.get("medication_review_notes"),
                "procedure_review_notes": s.get("procedure_review_notes"),
                "disclaimer": "Academic decision support suggestion only. Not a medical diagnosis or prescription."
            })
        
        df_export = pd.DataFrame(rows)
        buffer = io.BytesIO()
        df_export.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    @staticmethod
    def generate_patient_pdf_report(dataset_id="ds_default", patient_id=None):
        """Generate an individual patient audit and evidence-based clinical recommendation PDF report."""
        dataset = db_manager.get_dataset(dataset_id) or {
            "dataset_id": dataset_id,
            "dataset_name": "Clinical Cohort",
            "source": "Clinical Benchmark"
        }
        patient = db_manager.get_patient_by_id(dataset_id, patient_id) or {}
        df_all = db_manager.get_full_patient_analysis(dataset_id)

        numeric_cols = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c]) and c not in ["cluster_label", "is_anomaly", "anomaly_score", "run_id"] and "id" not in c.lower()]
        normal_mask = df_all["is_anomaly"] == 0 if "is_anomaly" in df_all.columns else np.ones(len(df_all), dtype=bool)
        if not np.any(normal_mask):
            normal_mask = np.ones(len(df_all), dtype=bool)
        pop_means = df_all.loc[normal_mask, numeric_cols].mean() if numeric_cols else None
        pop_stds = df_all.loc[normal_mask, numeric_cols].std().replace(0, 1.0) if numeric_cols else None

        from models.clinical_recommendation_engine import ClinicalRecommendationEngine
        rec_engine = ClinicalRecommendationEngine()
        rec = rec_engine.generate_recommendations_for_patient(
            patient_row=patient,
            pop_means=pop_means,
            pop_stds=pop_stds,
            dataset_id=dataset_id,
            analysis_run_id=patient.get("run_id")
        )

        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        logo_path = Config.BASE_DIR / "static" / "images" / "logo.png"
        if logo_path.exists():
            pdf.image(str(logo_path), x=14, y=10, w=18)

        # Header
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_text_color(2, 132, 199)
        pdf.cell(0, 8, "RARETRACE PATIENT CLINICAL AUDIT REPORT", ln=True, align="C")

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 6, f"Patient Profile: {patient_id} | Dataset: {dataset.get('dataset_name')}", ln=True, align="C")

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 4.5, f"Source: {dataset.get('source')} | Cluster Assignment: {rec['cluster_label']} | Anomaly Score: {rec['anomaly_score']}/100", ln=True, align="C")
        pdf.ln(4)

        # Mandatory Safety Disclaimer Box
        pdf.set_fill_color(254, 243, 199)
        pdf.set_text_color(146, 64, 14)
        pdf.set_font("Helvetica", "B", 7)
        pdf.multi_cell(0, 4, f"ACADEMIC & MEDICAL NOTICE: {rec['disclaimer']}", border=1, fill=True)
        pdf.ln(4)

        # Recommendation Priority & Status Banner
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(95, 7, f"Recommendation Priority: {rec['priority']}", border=1)
        pdf.cell(95, 7, f"Density Noise Status: {'Isolated Candidate Case' if rec['is_anomaly'] == 1 else 'Normative Cohort'}", border=1)
        pdf.ln(10)

        # Category A: What Was Detected
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(2, 132, 199)
        pdf.cell(0, 6, "Category A: What Was Detected?", ln=True)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(30, 41, 59)
        pdf.multi_cell(0, 4.5, rec["category_a_what_detected"]["summary"])
        pdf.ln(2)

        for finding in rec["category_a_what_detected"]["findings"][:5]:
            pdf.cell(5)
            pdf.cell(0, 4.5, f"- {finding}", ln=True)
        pdf.ln(4)

        # Category B: Why Flagged
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(2, 132, 199)
        pdf.cell(0, 6, "Category B: Why Was This Profile Flagged?", ln=True)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(30, 41, 59)
        pdf.multi_cell(0, 4.5, rec["category_b_why_flagged"]["summary"])
        pdf.ln(4)

        # Category C: Recommended Next Investigations
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(2, 132, 199)
        pdf.cell(0, 6, "Category C: Recommended Next Investigations", ln=True)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(30, 41, 59)
        for inv in rec["category_c_investigations"][:4]:
            pdf.cell(5)
            pdf.cell(0, 4.5, f"- {inv['test_name']} ({inv['domain']} evaluation)", ln=True)
        pdf.ln(4)

        # Category D: Specialist Referral
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(2, 132, 199)
        pdf.cell(0, 6, "Category D: Recommended Specialist Referral", ln=True)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 5, f"Primary Referral: {rec['category_d_specialists']['primary_specialist']}", ln=True)
        if rec['category_d_specialists']['additional_specialists']:
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(0, 4.5, f"Multidisciplinary Team: {', '.join(rec['category_d_specialists']['additional_specialists'])}", ln=True)
        pdf.ln(4)

        # Category E & F: Risk-Reduction Guidance & Management Considerations
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(2, 132, 199)
        pdf.cell(0, 6, "Categories E & F: Risk-Reduction & Clinical Management Considerations", ln=True)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(30, 41, 59)
        for g in rec["category_e_risk_reduction"][:3]:
            pdf.cell(5)
            pdf.cell(0, 4.5, f"- {g['topic']}: {g['guidance']}", ln=True)
        pdf.ln(2)
        for m in rec["category_f_management_considerations"][:2]:
            pdf.cell(5)
            pdf.cell(0, 4.5, f"- Management Note: {m}", ln=True)

        buffer = io.BytesIO()
        pdf.output(buffer)
        buffer.seek(0)
        return buffer

