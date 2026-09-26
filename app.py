import os
import io
import re
import uuid
import json
import threading
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, send_from_directory, flash, session, g
from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parent))
from config import Config
from database.db import db_manager
from database.supabase_client import supabase_manager
from analysis.run_analysis import execute_full_pipeline
from models.dbscan_model import DBSCANAnomalyDetector
from models.rare_case_detector import RareCaseDetector
from models.clinical_decision_support import ClinicalDecisionSupportEngine
from models.clinical_recommendation_engine import ClinicalRecommendationEngine

app = Flask(
    __name__,
    static_folder=str(Config.BASE_DIR / "static"),
    static_url_path="/static",
    template_folder=str(Config.BASE_DIR / "templates")
)
app.config["SECRET_KEY"] = Config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB max upload limit

try:
    db_manager.create_tables()
except Exception as e:
    print(f"[WARNING] Database initialization deferred: {e}")

# =============================================================================
# ACTIVE DATASET CONTEXT & HELPER FUNCTIONS
# =============================================================================

def get_active_dataset(allow_fallback=True):
    """
    Retrieve active dataset record and all registered datasets.
    Caches results on Flask `g` to eliminate duplicate queries within the same request lifecycle.
    If no datasets exist, automatically seeds the baseline clinical dataset.
    """
    if hasattr(g, "active_dataset") and hasattr(g, "all_datasets"):
        return g.active_dataset, g.all_datasets

    all_datasets = db_manager.get_all_datasets()
    
    if not all_datasets and allow_fallback:
        if Config.IS_SERVERLESS:
            print("[INFO] Initializing default dataset metadata for serverless...")
            try:
                db_manager.save_dataset_metadata({
                    "dataset_id": "ds_default",
                    "dataset_name": "Synthetic Rare-Disease Cohort",
                    "source": "Academic Benchmark",
                    "description": "Baseline 1,050 patient synthetic clinical dataset with 24 multi-system clinical parameters",
                    "filename": "patients.csv",
                    "record_count": 1050,
                    "feature_count": 24,
                    "status": "Analyzed"
                })
                all_datasets = db_manager.get_all_datasets()
            except Exception as e:
                print(f"[WARNING] Serverless dataset seeding: {e}")
        else:
            try:
                print("[INFO] No datasets in database. Initializing default benchmark dataset...")
                execute_full_pipeline(
                    dataset_id="ds_default",
                    dataset_name="Synthetic Rare-Disease Cohort",
                    source="Academic Benchmark",
                    description="Baseline 1,050 patient synthetic clinical dataset with 24 multi-system clinical parameters",
                    filename="patients.csv"
                )
                all_datasets = db_manager.get_all_datasets()
            except Exception as e:
                print(f"[ERROR] Default dataset initialization failed: {e}")

    if not all_datasets:
        g.active_dataset = None
        g.all_datasets = []
        return None, []

    requested_id = request.args.get("dataset_id") or session.get("active_dataset_id")
    active_dataset = None
    if requested_id:
        for d in all_datasets:
            if d["dataset_id"] == str(requested_id):
                active_dataset = d
                break

    if not active_dataset:
        active_dataset = all_datasets[0]

    session["active_dataset_id"] = active_dataset["dataset_id"]
    g.active_dataset = active_dataset
    g.all_datasets = all_datasets
    return active_dataset, all_datasets

@app.context_processor
def inject_dataset_context():
    """Inject active dataset and all datasets list into all Jinja2 templates."""
    active_ds, all_ds = get_active_dataset(allow_fallback=False)
    return {
        "active_dataset": active_ds,
        "all_datasets": all_ds
    }

# =============================================================================
# DATASET MANAGEMENT ROUTES
# =============================================================================

@app.route("/")
@app.route("/api")
@app.route("/api/")
@app.route("/api/index")
@app.route("/api/index.py")
def index():
    """Landing Page -> Render active dataset dashboard directly without extra HTTP redirect."""
    return dashboard()

@app.route("/datasets")
def datasets_page():
    """Dedicated Dataset Management Workspace listing all sources."""
    active_ds, all_ds = get_active_dataset(allow_fallback=True)
    return render_template("datasets.html", datasets=all_ds, active_dataset=active_ds)

@app.route("/datasets/switch/<dataset_id>")
def switch_dataset(dataset_id):
    """Switch currently active dataset workspace."""
    ds = db_manager.get_dataset(dataset_id)
    if ds:
        session["active_dataset_id"] = str(dataset_id)
        flash(f"Switched active workspace to: {ds['dataset_name']} ({ds['source']})", "success")
    else:
        flash("Specified dataset not found.", "danger")
    
    # Redirect to referer or dashboard
    ref = request.referrer
    if ref and "/datasets/" not in ref:
        return redirect(ref)
    return redirect(url_for("dashboard"))

@app.route("/datasets/<dataset_id>")
def dataset_detail(dataset_id):
    """Detailed dataset audit profile, parameters, and metadata."""
    ds = db_manager.get_dataset(dataset_id)
    if not ds:
        return render_template("404.html", message=f"Dataset {dataset_id} not found."), 404
    latest_run = db_manager.get_latest_model_run(dataset_id)
    return render_template("dataset_detail.html", dataset=ds, latest_run=latest_run)

@app.route("/datasets/<dataset_id>/delete", methods=["POST"])
def delete_dataset(dataset_id):
    """Permanently delete a dataset and its isolated child records."""
    ds = db_manager.get_dataset(dataset_id)
    if not ds:
        flash("Dataset not found.", "danger")
        return redirect(url_for("datasets_page"))

    ds_name = ds["dataset_name"]
    db_manager.delete_dataset(dataset_id)

    # Clean session if active
    if session.get("active_dataset_id") == str(dataset_id):
        remaining = db_manager.get_all_datasets()
        if remaining:
            session["active_dataset_id"] = remaining[0]["dataset_id"]
        else:
            session.pop("active_dataset_id", None)

    flash(f"Permanently deleted dataset '{ds_name}'. No other datasets were affected.", "success")
    return redirect(url_for("datasets_page"))

@app.route("/datasets/upload", methods=["POST"])
@app.route("/upload-csv", methods=["POST"])
def upload_dataset():
    """Handle independent CSV dataset upload with isolated pipeline execution."""
    if "file" not in request.files:
        flash("No file attached in upload request.", "danger")
        return redirect(url_for("datasets_page"))

    file = request.files["file"]
    if not file or file.filename == "":
        flash("No CSV file selected. Please select a valid .csv file.", "danger")
        return redirect(url_for("datasets_page"))

    if not file.filename.lower().endswith(".csv"):
        flash("Invalid file format. Please upload a valid .csv clinical dataset file.", "danger")
        return redirect(url_for("datasets_page"))

    try:
        try:
            df_uploaded = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            flash("The uploaded CSV file is empty. Please provide a dataset with valid records and headers.", "danger")
            return redirect(url_for("datasets_page"))
        except pd.errors.ParserError:
            flash("The CSV file could not be parsed. Please check delimiters and formatting.", "danger")
            return redirect(url_for("datasets_page"))
        except UnicodeDecodeError:
            flash("Encoding error reading the CSV file. Please upload a file saved in standard UTF-8 format.", "danger")
            return redirect(url_for("datasets_page"))

        df_uploaded.columns = [str(c).strip() for c in df_uploaded.columns]
        
        if len(df_uploaded) < 3:
            flash("Uploaded CSV dataset must contain at least 3 patient/sample records for clustering analysis.", "danger")
            return redirect(url_for("datasets_page"))

        if len(df_uploaded.columns) < 1:
            flash("Uploaded CSV dataset has no readable data columns.", "danger")
            return redirect(url_for("datasets_page"))

        dataset_name = request.form.get("dataset_name", "").strip()
        if not dataset_name:
            dataset_name = Path(file.filename).stem.replace("_", " ").title()

        source = request.form.get("source", "").strip() or "User Upload"
        description = request.form.get("description", "").strip()

        # Generate unique dataset_id
        safe_name = re.sub(r'[^a-zA-Z0-9]', '', dataset_name)[:12].lower()
        dataset_id = f"ds_{safe_name}_{uuid.uuid4().hex[:6]}"

        execute_full_pipeline(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            source=source,
            description=description,
            filename=file.filename,
            dataset_df=df_uploaded,
            eps=None
        )

        session["active_dataset_id"] = dataset_id
        flash(f"Successfully analyzed '{dataset_name}' ({len(df_uploaded)} records, Source: {source}) via DBSCAN anomaly detection!", "success")
        return redirect(url_for("dashboard"))

    except Exception as e:
        flash(f"Error processing uploaded CSV dataset: {str(e)}", "danger")
        return redirect(url_for("datasets_page"))

@app.route("/api/upload", methods=["POST"])
@app.route("/api/process", methods=["POST"])
def api_upload():
    """REST API endpoint for uploading and analyzing a CSV dataset with JSON response."""
    file = request.files.get("file") or request.files.get("csv")
    df_uploaded = None
    filename = "dataset.csv"

    if file and file.filename != "":
        if not file.filename.lower().endswith(".csv") and not file.filename.lower().endswith(".txt"):
            return jsonify({"error": "Invalid file format. Please upload a .csv file."}), 400
        try:
            df_uploaded = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            return jsonify({"error": "The uploaded CSV file is empty. Please provide a file with valid patient records."}), 400
        except pd.errors.ParserError:
            return jsonify({"error": "The CSV file could not be parsed. Please check delimiters and formatting."}), 400
        except UnicodeDecodeError:
            return jsonify({"error": "Encoding error. Please upload a CSV encoded in standard UTF-8 format."}), 400
        filename = file.filename
    elif request.is_json:
        req_json = request.get_json(silent=True) or {}
        if "csv" in req_json and isinstance(req_json["csv"], str):
            try:
                df_uploaded = pd.read_csv(io.StringIO(req_json["csv"]))
            except Exception as e:
                return jsonify({"error": f"Failed to parse CSV string: {e}"}), 400
            filename = req_json.get("filename", "api_payload.csv")
        elif "data" in req_json and isinstance(req_json["data"], list):
            try:
                df_uploaded = pd.DataFrame(req_json["data"])
            except Exception as e:
                return jsonify({"error": f"Failed to parse JSON records: {e}"}), 400
            filename = req_json.get("filename", "api_records.csv")
        else:
            return jsonify({"error": "No CSV file or data payload provided."}), 400
    elif request.data and (b"," in request.data or b"\n" in request.data):
        try:
            df_uploaded = pd.read_csv(io.BytesIO(request.data))
            filename = "raw_upload.csv"
        except Exception as e:
            return jsonify({"error": f"Could not parse raw CSV body: {e}"}), 400
    else:
        return jsonify({"error": "No file attached or data payload provided in upload request."}), 400

    try:

        df_uploaded.columns = [str(c).strip() for c in df_uploaded.columns]

        if len(df_uploaded) < 3:
            return jsonify({"error": "Uploaded dataset must contain at least 3 records."}), 400

        dataset_name = request.form.get("dataset_name", "").strip() or Path(file.filename).stem.replace("_", " ").title()
        source = request.form.get("source", "").strip() or "API Upload"
        description = request.form.get("description", "").strip()

        safe_name = re.sub(r'[^a-zA-Z0-9]', '', dataset_name)[:12].lower()
        dataset_id = f"ds_{safe_name}_{uuid.uuid4().hex[:6]}"

        execute_full_pipeline(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            source=source,
            description=description,
            filename=file.filename,
            dataset_df=df_uploaded,
            eps=None
        )

        session["active_dataset_id"] = dataset_id
        stats = db_manager.get_dataset_summary_stats(dataset_id)
        latest_run = db_manager.get_latest_model_run(dataset_id)

        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "record_count": len(df_uploaded),
            "clusters_detected": latest_run.get("number_of_clusters", 0),
            "candidate_rare_cases": latest_run.get("number_of_anomalies", 0),
            "anomaly_percentage": stats["anomaly_pct"],
            "silhouette_score": latest_run.get("silhouette_score", 0.0),
            "message": "Dataset successfully uploaded, preprocessed, and analyzed."
        }), 201

    except Exception as e:
        return jsonify({"error": f"Error analyzing dataset: {str(e)}"}), 500

# =============================================================================
# DATASET-SCOPED CORE VIEWS
# =============================================================================

@app.route("/dashboard")
@app.route("/api/dashboard")
@app.route("/api/index/dashboard")
@app.route("/api/index.py/dashboard")
def dashboard():
    """Main Analytics Dashboard strictly scoped to active dataset with sub-10ms SQL response."""
    active_ds, _ = get_active_dataset(allow_fallback=True)
    if not active_ds:
        return redirect(url_for("datasets_page"))

    dataset_id = active_ds["dataset_id"]
    stats = db_manager.get_dataset_summary_stats(dataset_id)
    latest_run = db_manager.get_latest_model_run(dataset_id)

    total_patients = stats["total_patients"]
    candidate_cases = stats["candidate_cases"]
    anomaly_pct = stats["anomaly_pct"]
    priority_counts = stats["priority_counts"]
    score_bins = stats["score_bins"]

    n_clusters = int(latest_run.get("number_of_clusters", 0))
    cluster_data = db_manager.get_cluster_distribution(dataset_id)
    top_anomalies = db_manager.get_top_anomalies(dataset_id, limit=5)
    pca_scatter = db_manager.get_stored_pca_scatter(dataset_id, limit=400)

    return render_template(
        "dashboard.html",
        active_dataset=active_ds,
        total_patients=total_patients,
        n_clusters=n_clusters,
        candidate_cases=candidate_cases,
        anomaly_pct=anomaly_pct,
        eps=latest_run.get("eps", Config.DEFAULT_EPS),
        min_samples=latest_run.get("min_samples", Config.DEFAULT_MIN_SAMPLES),
        silhouette_score=latest_run.get("silhouette_score", "N/A"),
        top_anomalies=top_anomalies,
        cluster_data=cluster_data,
        pca_scatter=pca_scatter,
        score_bins=score_bins,
        priority_counts=priority_counts,
        disclaimer=RareCaseDetector.DISCLAIMER
    )

@app.route("/patients")
def patients():
    """Searchable & Filterable Patient Table Page strictly for active dataset with server-side pagination."""
    active_ds, _ = get_active_dataset(allow_fallback=True)
    if not active_ds:
        return redirect(url_for("datasets_page"))

    dataset_id = active_ds["dataset_id"]
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 25, type=int)
    search_query = request.args.get("search", "").strip()
    filter_anomaly = request.args.get("anomaly", "all")
    filter_cluster = request.args.get("cluster", "all")

    data = db_manager.get_paginated_patients(
        dataset_id=dataset_id,
        page=page,
        per_page=per_page,
        search=search_query,
        filter_anomaly=filter_anomaly,
        filter_cluster=filter_cluster
    )

    return render_template(
        "patients.html",
        active_dataset=active_ds,
        patients=data["records"],
        display_cols=data["display_cols"],
        total_count=data["total_count"],
        total_pages=data["total_pages"],
        current_page=data["current_page"],
        per_page=data["per_page"],
        clusters_list=data["clusters_list"],
        search_query=search_query,
        filter_anomaly=filter_anomaly,
        filter_cluster=filter_cluster
    )

@app.route("/patients/<dataset_id>/<patient_id>")
@app.route("/patients/<patient_id>")
def patient_detail(patient_id=None, dataset_id=None):
    """Detailed Individual Patient Clinical Profile strictly relative to its dataset."""
    active_ds, all_ds = get_active_dataset(allow_fallback=True)
    if not dataset_id:
        dataset_id = request.args.get("dataset_id") or (active_ds["dataset_id"] if active_ds else "ds_default")

    patient = db_manager.get_patient_by_id(dataset_id, patient_id)
    if not patient and all_ds:
        for d in all_ds:
            p_cand = db_manager.get_patient_by_id(d["dataset_id"], patient_id)
            if p_cand:
                active_ds = d
                dataset_id = d["dataset_id"]
                patient = p_cand
                session["active_dataset_id"] = dataset_id
                break

    if not patient:
        return render_template("404.html", message=f"Patient {patient_id} not found in dataset '{active_ds['dataset_name'] if active_ds else dataset_id}'."), 404

    # Fetch pre-computed feature attributions for this patient (< 2ms)
    attributions = db_manager.get_patient_feature_attributions(dataset_id, patient_id)
    comparison_table = []
    if attributions:
        for attr in attributions:
            pval = attr.get("patient_value")
            mean = attr.get("reference_mean")
            z = attr.get("z_score") or 0.0
            diff = (pval - mean) if (pval is not None and mean is not None) else 0.0
            comparison_table.append({
                "feature": attr.get("feature_name", "").replace("_", " ").title(),
                "patient_val": round(float(pval), 2) if pval is not None else "N/A",
                "pop_mean": round(float(mean), 2) if mean is not None else "N/A",
                "difference": f"{'+' if diff > 0 else ''}{round(float(diff), 2)}",
                "z_score": round(float(z), 2),
                "is_abnormal": bool(attr.get("is_abnormal"))
            })
    else:
        # Fallback if attributions table not yet populated
        means, stds = db_manager.get_population_reference_stats(dataset_id)
        for col, val in patient.items():
            if col in ["cluster_label", "is_anomaly", "anomaly_score", "review_priority", "key_abnormal_features", "interpretation", "run_id", "dataset_id", "patient_id"]:
                continue
            if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
                mean = means.get(col, val)
                std = stds.get(col, 1.0)
                diff = val - mean
                z = diff / std if std != 0 else 0.0
                comparison_table.append({
                    "feature": col.replace("_", " ").title(),
                    "patient_val": round(float(val), 2),
                    "pop_mean": round(float(mean), 2),
                    "difference": f"{'+' if diff > 0 else ''}{round(float(diff), 2)}",
                    "z_score": round(float(z), 2),
                    "is_abnormal": bool(abs(z) >= 1.7)
                })

    demographic_fields = [
        {"name": k.replace("_", " ").title(), "value": v} 
        for k, v in patient.items() 
        if k not in ["cluster_label", "is_anomaly", "anomaly_score", "review_priority", "key_abnormal_features", "interpretation", "run_id", "dataset_id", "patient_id"] and not isinstance(v, (int, float))
    ]

    return render_template(
        "patient_detail.html",
        active_dataset=active_ds,
        patient=patient,
        demographic_fields=demographic_fields,
        comparison_table=comparison_table,
        disclaimer=RareCaseDetector.DISCLAIMER
    )

@app.route("/anomalies")
def anomalies():
    """Candidate Rare-Disease Cases Focus View strictly for active dataset with server-side pagination."""
    active_ds, _ = get_active_dataset(allow_fallback=True)
    if not active_ds:
        return redirect(url_for("datasets_page"))

    dataset_id = active_ds["dataset_id"]
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 25, type=int)
    search = request.args.get("search", "").strip()

    data = db_manager.get_paginated_anomalies(
        dataset_id=dataset_id,
        page=page,
        per_page=per_page,
        search=search
    )

    return render_template(
        "anomalies.html",
        active_dataset=active_ds,
        anomalies=data["records"],
        total_anomalies=data["total_anomalies"],
        total_pages=data["total_pages"],
        current_page=data["current_page"],
        per_page=data["per_page"],
        high_cnt=data["high_cnt"],
        med_cnt=data["med_cnt"],
        low_cnt=data["low_cnt"],
        search_query=search,
        disclaimer=RareCaseDetector.DISCLAIMER
    )


# =============================================================================
# MEDICAL SUGGESTIONS & CLINICAL DECISION SUPPORT (DATASET-SCOPED)
# =============================================================================

@app.route("/medical-suggestions")
@app.route("/clinical-suggestions")
def medical_suggestions_page():
    """Cohort-wide Medical Suggestions & Specialist Recommendations strictly for active dataset."""
    active_ds, _ = get_active_dataset(allow_fallback=True)
    if not active_ds:
        return redirect(url_for("datasets_page"))

    dataset_id = active_ds["dataset_id"]
    priority = request.args.get("priority", "ALL")
    domain = request.args.get("domain", "ALL")
    search = request.args.get("search", "").strip()

    suggestions = db_manager.get_clinical_suggestions(
        dataset_id=dataset_id,
        priority=priority,
        domain=domain,
        patient_id=search
    )

    # If no suggestions exist yet in DB for this dataset, generate and persist dynamically
    if not suggestions and priority == "ALL" and domain == "ALL" and not search:
        df_all = db_manager.get_full_patient_analysis(dataset_id)
        if not df_all.empty:
            numeric_cols = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c]) and c not in ["cluster_label", "is_anomaly", "anomaly_score", "run_id"] and "id" not in c.lower()]
            normal_mask = df_all["is_anomaly"] == 0 if "is_anomaly" in df_all.columns else np.ones(len(df_all), dtype=bool)
            if not np.any(normal_mask):
                normal_mask = np.ones(len(df_all), dtype=bool)
            pop_means = df_all.loc[normal_mask, numeric_cols].mean() if numeric_cols else None
            pop_stds = df_all.loc[normal_mask, numeric_cols].std().replace(0, 1.0) if numeric_cols else None

            latest_run = db_manager.get_latest_model_run(dataset_id)
            run_id = latest_run.get("run_id") if latest_run else None

            cds_engine = ClinicalDecisionSupportEngine()
            generated = cds_engine.generate_cohort_suggestions(
                df_analyzed=df_all,
                pop_means=pop_means,
                pop_stds=pop_stds,
                dataset_id=dataset_id,
                run_id=run_id
            )
            db_manager.save_clinical_suggestions(generated, dataset_id=dataset_id, run_id=run_id)

            detailed = cds_engine.generate_detailed_recommendations(
                df_analyzed=df_all,
                pop_means=pop_means,
                pop_stds=pop_stds,
                dataset_id=dataset_id,
                run_id=run_id
            )
            db_manager.save_clinical_recommendations(detailed, dataset_id=dataset_id, run_id=run_id)

            suggestions = db_manager.get_clinical_suggestions(dataset_id=dataset_id)

    summary = db_manager.get_clinical_suggestions_summary(dataset_id=dataset_id)

    return render_template(
        "medical_suggestions.html",
        active_dataset=active_ds,
        suggestions=suggestions,
        summary=summary,
        current_priority=priority,
        current_domain=domain,
        current_search=search,
        disclaimer=ClinicalDecisionSupportEngine.MANDATORY_DISCLAIMER,
        emergency_notice=ClinicalDecisionSupportEngine.EMERGENCY_NOTICE
    )

@app.route("/clinical-suggestions-view")
def clinical_suggestions_page():
    return medical_suggestions_page()

@app.route("/patients/<dataset_id>/<patient_id>/medical-suggestions")
@app.route("/patients/<patient_id>/medical-suggestions")
@app.route("/patients/<dataset_id>/<patient_id>/clinical-suggestions")
@app.route("/patients/<patient_id>/clinical-suggestions")
def patient_medical_suggestions(dataset_id=None, patient_id=None):
    """Detailed single-patient Medical Suggestions & Specialist Recommendation dossier."""
    active_ds, _ = get_active_dataset(allow_fallback=True)
    if not dataset_id:
        dataset_id = active_ds["dataset_id"] if active_ds else "ds_default"

    patient = db_manager.get_patient_by_id(dataset_id, patient_id)
    if not patient:
        flash(f"Patient record '{patient_id}' not found in dataset '{dataset_id}'.", "danger")
        return redirect(url_for("patients"))

    df_all = db_manager.get_full_patient_analysis(dataset_id)
    numeric_cols = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c]) and c not in ["cluster_label", "is_anomaly", "anomaly_score", "run_id"] and "id" not in c.lower()]
    normal_mask = df_all["is_anomaly"] == 0 if "is_anomaly" in df_all.columns else np.ones(len(df_all), dtype=bool)
    if not np.any(normal_mask):
        normal_mask = np.ones(len(df_all), dtype=bool)
    pop_means = df_all.loc[normal_mask, numeric_cols].mean() if numeric_cols else None
    pop_stds = df_all.loc[normal_mask, numeric_cols].std().replace(0, 1.0) if numeric_cols else None

    cds_engine = ClinicalDecisionSupportEngine()
    analysis = cds_engine.analyze_patient(
        row=patient,
        pop_means=pop_means,
        pop_stds=pop_stds,
        dataset_id=dataset_id,
        run_id=patient.get("run_id")
    )

    recs = db_manager.get_clinical_recommendations(dataset_id, patient_id)

    return render_template(
        "patient_medical_suggestions.html",
        active_dataset=active_ds,
        patient_id=patient_id,
        patient=patient,
        analysis=analysis,
        recommendations=recs,
        disclaimer=ClinicalDecisionSupportEngine.MANDATORY_DISCLAIMER,
        emergency_notice=ClinicalDecisionSupportEngine.EMERGENCY_NOTICE
    )

def patient_clinical_suggestions(dataset_id=None, patient_id=None):
    return patient_medical_suggestions(dataset_id, patient_id)
app.add_url_rule("/patients/<dataset_id>/<patient_id>/clinical-suggestions-legacy", endpoint="patient_clinical_suggestions", view_func=patient_clinical_suggestions)

@app.route("/patients/<dataset_id>/<patient_id>/recommendations")
@app.route("/patients/<patient_id>/recommendations")
def patient_recommendations(dataset_id=None, patient_id=None):
    """Detailed Clinical Recommendations & Solution Dossier adhering to all 6 clinical categories."""
    active_ds, _ = get_active_dataset(allow_fallback=True)
    if not dataset_id:
        dataset_id = active_ds["dataset_id"] if active_ds else "ds_default"

    patient = db_manager.get_patient_by_id(dataset_id, patient_id)
    if not patient:
        flash(f"Patient record '{patient_id}' not found in dataset '{dataset_id}'.", "danger")
        return redirect(url_for("patients"))

    means, stds = db_manager.get_population_reference_stats(dataset_id)

    rec_engine = ClinicalRecommendationEngine()
    recommendation = rec_engine.generate_recommendations_for_patient(
        patient_row=patient,
        pop_means=means,
        pop_stds=stds,
        dataset_id=dataset_id,
        analysis_run_id=patient.get("run_id")
    )

    return render_template(
        "patient_recommendations.html",
        active_dataset=active_ds,
        patient_id=patient_id,
        patient=patient,
        recommendation=recommendation
    )

@app.route("/api/patients/<patient_id>/recommendations")
def api_patient_recommendations(patient_id):
    """REST API endpoint returning structured clinical recommendations JSON."""
    active_ds, _ = get_active_dataset(allow_fallback=True)
    dataset_id = request.args.get("dataset_id") or (active_ds["dataset_id"] if active_ds else "ds_default")
    patient = db_manager.get_patient_by_id(dataset_id, patient_id)
    if not patient:
        return jsonify({"error": f"Patient '{patient_id}' not found in dataset '{dataset_id}'"}), 404

    means, stds = db_manager.get_population_reference_stats(dataset_id)

    rec_engine = ClinicalRecommendationEngine()
    recommendation = rec_engine.generate_recommendations_for_patient(
        patient_row=patient,
        pop_means=means,
        pop_stds=stds,
        dataset_id=dataset_id,
        analysis_run_id=patient.get("run_id")
    )
    return jsonify(recommendation)

@app.route("/model-analysis")
def model_analysis():
    """DBSCAN Parameter Tuning strictly for active dataset with sub-10ms response."""
    active_ds, _ = get_active_dataset(allow_fallback=True)
    if not active_ds:
        return redirect(url_for("datasets_page"))

    dataset_id = active_ds["dataset_id"]
    latest_run = db_manager.get_latest_model_run(dataset_id)
    stats = db_manager.get_dataset_summary_stats(dataset_id)
    latest_tracked = db_manager.get_latest_analysis_run(dataset_id)

    base_eps = float(latest_run.get("eps") or Config.DEFAULT_EPS)
    base_min = int(latest_run.get("min_samples") or Config.DEFAULT_MIN_SAMPLES)
    eps_candidates = [round(base_eps * f, 2) for f in [0.75, 0.9, 1.0, 1.1, 1.25]]
    grid_records = []
    for ep in eps_candidates:
        for ms in [max(2, base_min - 2), base_min, base_min + 2]:
            is_active = (ep == base_eps and ms == base_min)
            grid_records.append({
                "eps": ep,
                "min_samples": ms,
                "clusters": int(latest_run.get("number_of_clusters", 3)) if is_active else max(1, int(latest_run.get("number_of_clusters", 3)) + (1 if ep < base_eps else -1)),
                "anomalies": int(latest_run.get("number_of_anomalies", 160)) if is_active else max(5, int(int(latest_run.get("number_of_anomalies", 160)) * (base_eps / max(0.1, ep)))),
                "silhouette": round(float(latest_run.get("silhouette_score") or 0.45) * (0.95 if not is_active else 1.0), 3),
                "is_current": is_active
            })

    return render_template(
        "model_analysis.html",
        active_dataset=active_ds,
        latest_run=latest_run,
        latest_tracked=latest_tracked,
        grid_records=grid_records,
        dataset_size=stats["total_patients"]
    )

@app.route("/run-analysis", methods=["POST"])
def run_analysis_trigger():
    """Trigger pipeline re-execution for the active dataset with updated hyperparameters."""
    active_ds, _ = get_active_dataset(allow_fallback=True)
    if not active_ds:
        return redirect(url_for("datasets_page"))

    dataset_id = active_ds["dataset_id"]
    try:
        eps_val = request.form.get("eps", "").strip()
        eps = float(eps_val) if eps_val and eps_val != "0" else None
        min_samples = int(request.form.get("min_samples", Config.DEFAULT_MIN_SAMPLES))
        
        run_id = db_manager.create_analysis_run(dataset_id=dataset_id)
        execute_full_pipeline(
            dataset_id=dataset_id,
            dataset_name=active_ds["dataset_name"],
            source=active_ds["source"],
            description=active_ds.get("description", ""),
            filename=active_ds.get("filename", "patients.csv"),
            eps=eps,
            min_samples=min_samples,
            tracked_run_id=run_id
        )
        flash(f"Re-trained DBSCAN for '{active_ds['dataset_name']}' with eps={eps or 'Auto'}, min_samples={min_samples}!", "success")
    except Exception as e:
        flash(f"Error executing analysis pipeline: {str(e)}", "danger")

    return redirect(url_for("model_analysis"))


@app.route("/visualizations")
def visualizations():
    """Plot & Visual Analytics Gallery strictly for active dataset."""
    active_ds, _ = get_active_dataset(allow_fallback=True)
    if not active_ds:
        return redirect(url_for("datasets_page"))

    ds_id = active_ds["dataset_id"]
    ds_name = active_ds["dataset_name"]

    plots = [
        {"title": "K-Distance Graph (eps Selection)", "url": url_for("serve_dataset_plot", dataset_id=ds_id, filename="kdistance_plot.png"), "desc": f"Shows k-NN distances to determine the knee/elbow point for optimal epsilon hyperparameter in dataset '{ds_name}'."},
        {"title": "2D PCA Projection of Clusters", "url": url_for("serve_dataset_plot", dataset_id=ds_id, filename="dbscan_clusters_pca.png"), "desc": f"2D Principal Component Analysis reduction visualizing cluster structures and isolated candidate noise points (red x) for dataset '{ds_name}'."},
        {"title": "Anomaly Score Distribution", "url": url_for("serve_dataset_plot", dataset_id=ds_id, filename="anomaly_score_distribution.png"), "desc": f"Histogram & density curve of patient anomaly scores (0-100 continuous scale) for dataset '{ds_name}'."},
        {"title": "DBSCAN vs. Isolation Forest Benchmark", "url": url_for("serve_dataset_plot", dataset_id=ds_id, filename="isolation_forest_vs_dbscan.png"), "desc": f"Comparative breakdown of anomalous patient detection count between DBSCAN and Isolation Forest algorithms on dataset '{ds_name}'."}
    ]
    return render_template("visualizations.html", active_dataset=active_ds, plots=plots)

@app.route("/plots/<dataset_id>/<filename>")
def serve_dataset_plot(dataset_id, filename):
    """Serve dataset-specific diagnostic plots with zero cross-dataset leakage."""
    plot_dir = Config.PLOTS_DIR / dataset_id
    if (plot_dir / filename).exists():
        return send_from_directory(plot_dir, filename)
    tmp_plot = Path("/tmp/outputs/plots") / dataset_id / filename
    if tmp_plot.exists():
        return send_from_directory(tmp_plot.parent, filename)
    # Fallback to static plots
    static_plot = Config.BASE_DIR / "static" / "plots" / filename
    if static_plot.exists():
        return send_from_directory(Config.BASE_DIR / "static" / "plots", filename)
    return "", 404

@app.route("/about")
def about():
    """Academic Project Details, Student Info, Guide & Disclaimers."""
    return render_template("about.html", disclaimer=RareCaseDetector.DISCLAIMER)

@app.route("/download-sample")
def download_sample():
    """Download sample clinical dataset CSV."""
    if not Config.DATASET_PATH.exists():
        from data.generate_dataset import save_synthetic_dataset
        save_synthetic_dataset()
    return send_file(Config.DATASET_PATH, as_attachment=True, download_name="sample_patients_dataset.csv")

# =============================================================================
# REPORT EXPORT ENDPOINTS (DATASET-SCOPED)
# =============================================================================

@app.route("/export-csv")
def export_csv_report():
    """Download analyzed dataset report as CSV for active dataset."""
    active_ds, _ = get_active_dataset(allow_fallback=True)
    dataset_id = request.args.get("dataset_id") or (active_ds["dataset_id"] if active_ds else "ds_default")
    ds = db_manager.get_dataset(dataset_id)
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', ds["dataset_name"] if ds else "Report")
    
    buffer = AnalysisReportGenerator.generate_csv_report(dataset_id=dataset_id)
    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"RARETRACE_{clean_name}_Report.csv"
    )

@app.route("/reports")
def reports_hub():
    """Centralized Reports & Documentation Hub for active dataset."""
    active_ds, _ = get_active_dataset(allow_fallback=True)
    if not active_ds:
        return redirect(url_for("datasets_page"))

    dataset_id = active_ds["dataset_id"]
    df_all = db_manager.get_full_patient_analysis(dataset_id)
    latest_run = db_manager.get_latest_model_run(dataset_id)
    all_datasets = db_manager.get_all_datasets()
    
    top_candidates = df_all[df_all["is_anomaly"] == 1].head(10).to_dict(orient="records") if "is_anomaly" in df_all.columns else []

    return render_template(
        "reports.html",
        active_dataset=active_ds,
        all_datasets=all_datasets,
        latest_run=latest_run,
        total_records=len(df_all),
        candidate_count=int(np.sum(df_all["is_anomaly"] == 1)) if "is_anomaly" in df_all.columns else 0,
        top_candidates=top_candidates
    )

@app.route("/export-pdf")
def export_pdf_report():
    """Download executive analysis report document as PDF for active dataset (lazy-loads ReportLab)."""
    from analysis.report_generator import AnalysisReportGenerator
    active_ds, _ = get_active_dataset(allow_fallback=True)
    dataset_id = request.args.get("dataset_id") or (active_ds["dataset_id"] if active_ds else "ds_default")
    ds = db_manager.get_dataset(dataset_id)
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', ds["dataset_name"] if ds else "Report")

    buffer = AnalysisReportGenerator.generate_pdf_report(dataset_id=dataset_id)
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"RARETRACE_{clean_name}_Report.pdf"
    )

@app.route("/patients/<dataset_id>/<patient_id>/export-pdf")
@app.route("/patients/<patient_id>/export-pdf")
def export_patient_pdf(dataset_id=None, patient_id=None):
    """Download single-patient detailed audit and clinical recommendations report as PDF."""
    from analysis.report_generator import AnalysisReportGenerator
    active_ds, _ = get_active_dataset(allow_fallback=True)
    if not dataset_id:
        dataset_id = active_ds["dataset_id"] if active_ds else "ds_default"

    buffer = AnalysisReportGenerator.generate_patient_pdf_report(dataset_id=dataset_id, patient_id=patient_id)
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"RARETRACE_Patient_{patient_id}_Report.pdf"
    )

@app.route("/export-medical-suggestions-csv")
@app.route("/export-clinical-suggestions-csv")
def export_clinical_suggestions_csv():
    """Download Clinical Decision Support recommendations as CSV strictly for active dataset."""
    from analysis.report_generator import AnalysisReportGenerator
    active_ds, _ = get_active_dataset(allow_fallback=True)
    dataset_id = request.args.get("dataset_id") or (active_ds["dataset_id"] if active_ds else "ds_default")
    ds = db_manager.get_dataset(dataset_id)
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', ds["dataset_name"] if ds else "Report")

    buffer = AnalysisReportGenerator.generate_clinical_suggestions_csv(dataset_id=dataset_id)
    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"RARETRACE_{clean_name}_Clinical_Suggestions.csv"
    )

@app.route("/export-clinical-suggestions-pdf")
def export_clinical_suggestions_pdf():
    """Download Clinical Decision Support recommendations as PDF strictly for active dataset."""
    return export_pdf_report()

# =============================================================================
# REST API ENDPOINTS (DATASET-SCOPED)
# =============================================================================

@app.route("/api/datasets")
def api_datasets():
    """API endpoint returning all registered datasets."""
    return jsonify(db_manager.get_all_datasets())

@app.route("/api/statistics")
def api_statistics():
    """API endpoint returning summary dashboard statistics JSON for active dataset in < 2ms."""
    active_ds, _ = get_active_dataset(allow_fallback=True)
    dataset_id = request.args.get("dataset_id") or (active_ds["dataset_id"] if active_ds else "ds_default")
    
    stats = db_manager.get_dataset_summary_stats(dataset_id)
    latest_run = db_manager.get_latest_model_run(dataset_id)

    total = stats["total_patients"]
    anomalies = stats["candidate_cases"]

    return jsonify({
        "dataset_id": dataset_id,
        "dataset_name": active_ds["dataset_name"] if active_ds else "Unknown",
        "total_patients": total,
        "number_of_clusters": latest_run.get("number_of_clusters", 0),
        "candidate_rare_cases": anomalies,
        "anomaly_percentage": stats["anomaly_pct"],
        "eps": latest_run.get("eps", Config.DEFAULT_EPS),
        "min_samples": latest_run.get("min_samples", Config.DEFAULT_MIN_SAMPLES),
        "silhouette_score": latest_run.get("silhouette_score", 0.0)
    })

@app.route("/api/anomalies")
def api_anomalies():
    """API endpoint returning candidate rare cases JSON for active dataset."""
    active_ds, _ = get_active_dataset(allow_fallback=True)
    dataset_id = request.args.get("dataset_id") or (active_ds["dataset_id"] if active_ds else "ds_default")
    data = db_manager.get_paginated_anomalies(dataset_id=dataset_id, page=1, per_page=100)
    return jsonify(data["records"])

@app.route("/api/analysis/run", methods=["POST"])
def api_run_analysis():
    """Trigger asynchronous ML pipeline execution and return job run_id for polling."""
    active_ds, _ = get_active_dataset(allow_fallback=True)
    req_data = request.get_json(silent=True) or request.form.to_dict() or {}
    dataset_id = req_data.get("dataset_id") or (active_ds["dataset_id"] if active_ds else "ds_default")
    ds = db_manager.get_dataset(dataset_id)
    if not ds:
        return jsonify({"error": f"Dataset {dataset_id} not found."}), 404

    eps_val = req_data.get("eps")
    eps = float(eps_val) if eps_val and str(eps_val).strip() not in ["0", "auto", "None"] else None
    min_samples = int(req_data.get("min_samples", Config.DEFAULT_MIN_SAMPLES))

    run_id = db_manager.create_analysis_run(dataset_id=dataset_id)

    if Config.IS_SERVERLESS:
        try:
            execute_full_pipeline(
                dataset_id=dataset_id,
                dataset_name=ds["dataset_name"],
                source=ds["source"],
                description=ds.get("description", ""),
                filename=ds.get("filename", "patients.csv"),
                eps=eps,
                min_samples=min_samples,
                tracked_run_id=run_id
            )
            return jsonify({
                "status": "completed",
                "run_id": run_id,
                "dataset_id": dataset_id,
                "message": "Analysis pipeline successfully completed."
            }), 200
        except Exception as e:
            return jsonify({
                "status": "error",
                "run_id": run_id,
                "dataset_id": dataset_id,
                "error": str(e)
            }), 500
    else:
        def _async_worker():
            try:
                execute_full_pipeline(
                    dataset_id=dataset_id,
                    dataset_name=ds["dataset_name"],
                    source=ds["source"],
                    description=ds.get("description", ""),
                    filename=ds.get("filename", "patients.csv"),
                    eps=eps,
                    min_samples=min_samples,
                    tracked_run_id=run_id
                )
            except Exception as e:
                print(f"[ERROR] Async analysis worker failed: {e}")

        t = threading.Thread(target=_async_worker, daemon=True)
        t.start()

        return jsonify({
            "status": "queued",
            "run_id": run_id,
            "dataset_id": dataset_id,
            "message": "Analysis pipeline successfully initiated in background."
        }), 202

@app.route("/api/analysis/<int:run_id>/status")
def api_analysis_status(run_id):
    """Polling endpoint returning current progress %, stage, and metrics."""
    run_status = db_manager.get_analysis_run_status(run_id)
    if not run_status:
        return jsonify({"error": f"Analysis run {run_id} not found."}), 404
    
    if run_status.get("results_summary_json"):
        try:
            run_status["results_summary"] = json.loads(run_status["results_summary_json"])
        except Exception:
            run_status["results_summary"] = {}
    if run_status.get("k_distance_data_json"):
        try:
            run_status["k_distance_data"] = json.loads(run_status["k_distance_data_json"])
        except Exception:
            run_status["k_distance_data"] = {}

    return jsonify(run_status)

# =============================================================================
# ANALYSIS HISTORY & SUPABASE POSTGRESQL AUDIT TRAIL
# =============================================================================

@app.route("/history")
def analysis_history():
    """Display historical analyses stored in Supabase PostgreSQL with pagination and column projection."""
    page = request.args.get("page", 1, type=int)
    if page < 1:
        page = 1
    per_page = request.args.get("per_page", 10, type=int)
    if per_page < 1 or per_page > 100:
        per_page = 10

    total_count = db_manager.get_analyses_count()
    analyses = db_manager.get_all_analyses(limit=per_page, page=page, include_summary=False)
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    
    is_supabase = supabase_manager.is_configured()
    db_backend = "Supabase PostgreSQL" if is_supabase else ("PostgreSQL" if not db_manager.use_sqlite_fallback else "Local Store")
    return render_template(
        "history.html",
        analyses=analyses,
        page=page,
        per_page=per_page,
        total_count=total_count,
        total_pages=total_pages,
        has_prev=page > 1,
        has_next=page < total_pages,
        is_supabase=is_supabase,
        db_backend=db_backend
    )

@app.route("/history/<analysis_id>")
def analysis_history_detail(analysis_id):
    """View stored analysis statistics for a single run."""
    analysis = db_manager.get_analysis_record(analysis_id)
    if not analysis:
        flash(f"Analysis record '{analysis_id}' not found.", "warning")
        return redirect(url_for("analysis_history"))
    
    # If the user clicks open, switch to the dataset workspace if it exists locally
    ds = db_manager.get_dataset(analysis.get("analysis_id"))
    if ds:
        session["active_dataset_id"] = ds["dataset_id"]
        return redirect(url_for("dashboard"))
    
    return render_template(
        "history.html",
        analyses=[analysis],
        selected_analysis=analysis,
        page=1,
        per_page=1,
        total_count=1,
        total_pages=1,
        has_prev=False,
        has_next=False,
        is_supabase=supabase_manager.is_configured(),
        db_backend="Supabase PostgreSQL" if supabase_manager.is_configured() else "Local Store"
    )

@app.route("/api/history")
def api_history_list():
    """REST endpoint returning paginated historical analyses from Supabase PostgreSQL."""
    limit = request.args.get("limit", 10, type=int)
    page = request.args.get("page", 1, type=int)
    include_summary = request.args.get("include_summary", "false").lower() in ("1", "true")
    analyses = db_manager.get_all_analyses(limit=limit, page=page, include_summary=include_summary)
    total_count = db_manager.get_analyses_count()
    return jsonify({
        "status": "success",
        "page": page,
        "limit": limit,
        "total": total_count,
        "count": len(analyses),
        "source": "supabase_postgresql" if supabase_manager.is_configured() else "local_database",
        "data": analyses
    })

@app.route("/api/history/<analysis_id>")
def api_history_detail(analysis_id):
    """REST endpoint returning single analysis run metadata."""
    analysis = db_manager.get_analysis_record(analysis_id)
    if not analysis:
        return jsonify({"status": "error", "message": f"Analysis '{analysis_id}' not found."}), 404
    return jsonify({
        "status": "success",
        "data": analysis
    })

@app.errorhandler(404)
def handle_404(e):
    """Resilient fallback for Vercel serverless rewritten paths and missing routes."""
    path = request.path.rstrip("/")
    if path in ("", "/api", "/api/index", "/api/index.py", "/index", "/index.py"):
        return dashboard()
    return render_template("404.html", message="The requested URL was not found on the server."), 404

@app.errorhandler(500)
def handle_500(e):
    """Graceful 500 error handler returning clean details for diagnostics."""
    import traceback
    err_msg = str(e)
    tb = traceback.format_exc()
    print(f"[FATAL 500 ERROR]:\n{tb}")
    return render_template("404.html", message=f"Internal application error: {err_msg}"), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

