import time
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config
from data.generate_dataset import generate_synthetic_dataset
from models.preprocessing import ClinicalDataPreprocessor
from models.dbscan_model import DBSCANAnomalyDetector
from models.rare_case_detector import RareCaseDetector
from models.clinical_decision_support import ClinicalDecisionSupportEngine
from models.clinical_recommendation_engine import ClinicalRecommendationEngine
from analysis.evaluation import ModelEvaluator
from analysis.visualization import Visualizer
from sklearn.decomposition import PCA
from database.db import db_manager

def execute_full_pipeline(
    dataset_id="ds_default",
    dataset_name="Synthetic Rare-Disease Cohort",
    source="Academic Benchmark",
    description="Baseline 1,050 patient synthetic dataset with 24 clinical parameters",
    filename="patients.csv",
    dataset_df=None,
    eps=None,
    min_samples=Config.DEFAULT_MIN_SAMPLES,
    tracked_run_id=None
):
    """
    Execute complete end-to-end ML pipeline strictly isolated for a single dataset.
    Never mixes data, models, scalers, parameters, or plots across datasets.
    Pre-computes PCA 2D coordinates and feature attributions for sub-50ms query speeds.
    """
    t0 = time.time()
    dataset_id = str(dataset_id)
    print("=" * 75)
    print(f"STARTING INDEPENDENT PIPELINE FOR DATASET: [{dataset_id}] {dataset_name}")
    print(f"Source: {source} | File: {filename}")
    print("=" * 75)

    db_manager.create_tables()
    if tracked_run_id is None:
        try:
            tracked_run_id = db_manager.create_analysis_run(dataset_id=dataset_id)
        except Exception:
            tracked_run_id = None

    def _update_stage(stage, progress, message):
        if tracked_run_id:
            try:
                db_manager.update_analysis_run(
                    tracked_run_id,
                    status="running",
                    stage=stage,
                    progress=progress,
                    message=message
                )
            except Exception as ex:
                print(f"[WARNING] Stage update deferred: {ex}")

    try:
        _update_stage("validating", 10, "Validating cohort schema and patient identifiers...")

        # 1. Load or accept arbitrary dataset
        if dataset_df is None:
            if Config.DATASET_PATH.exists():
                print(f"[1/7] Loading dataset from {Config.DATASET_PATH}...")
                df_raw = pd.read_csv(Config.DATASET_PATH)
            else:
                print("[1/7] Dataset missing. Generating synthetic clinical dataset...")
                df_raw = generate_synthetic_dataset(n_samples=1050, seed=42)
                df_raw.to_csv(Config.DATASET_PATH, index=False)
        else:
            df_raw = dataset_df.copy()

        # Clean whitespace in column names
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        print(f"      Dataset loaded: {len(df_raw)} records, {len(df_raw.columns)} fields.")

        # 2. Register Dataset Metadata & Save Patients with dataset_id
        db_manager.create_tables()
        db_manager.save_dataset_metadata({
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "source": source,
            "description": description,
            "filename": filename,
            "record_count": len(df_raw),
            "feature_count": len(df_raw.columns),
            "status": "Analyzing..."
        })
        db_manager.save_patients(df_raw, dataset_id=dataset_id)

        # 3. Dynamic Preprocessing & Feature Engineering ONLY for this dataset
        _update_stage("preprocessing", 25, f"Dynamic preprocessing and feature standardization for {len(df_raw)} records...")
        print(f"[2/7] Dynamic Data Preprocessing & Feature Engineering for '{dataset_id}'...")
        preprocessor = ClinicalDataPreprocessor()
        df_clean, df_eng, X_scaled = preprocessor.fit_transform(df_raw)
        
        # Save dataset-isolated preprocessor
        try:
            ds_models_dir = Config.MODELS_DIR / dataset_id
            ds_models_dir.mkdir(parents=True, exist_ok=True)
            preprocessor.save(ds_models_dir / "preprocessor.joblib")
        except Exception as ex:
            print(f"[WARNING] Preprocessor model save deferred: {ex}")

        # 4. Adaptive DBSCAN Model Training & Anomaly Scoring ONLY for this dataset
        _update_stage("k_distance", 40, "Computing k-NN distance graph and optimal eps knee point...")
        detector = DBSCANAnomalyDetector(eps=eps, min_samples=min_samples)
        labels = detector.fit_predict(X_scaled)
        chosen_eps = detector.eps
        print(f"[3/7] DBSCAN Training Completed (Chosen eps={chosen_eps}, min_samples={min_samples})...")
        
        anomaly_scores = detector.compute_anomaly_scores(X_scaled, labels)
        try:
            detector.save(ds_models_dir / "dbscan_model.joblib")
        except Exception as ex:
            print(f"[WARNING] DBSCAN model save deferred: {ex}")
        k_distances = detector.compute_k_distances(X_scaled)

        # 5. Rare Case Identification & Feature Attribution relative ONLY to this dataset's population
        _update_stage("dbscan", 55, f"Candidate rare case identification (eps={chosen_eps}, min_samples={min_samples})...")
        print(f"[4/7] Analyzing Candidate Rare-Disease Cases relative to '{dataset_name}' population...")
        df_analyzed, candidate_details, pop_means, pop_stds = RareCaseDetector.analyze_candidates(
            df_clean, df_eng, X_scaled, labels, anomaly_scores
        )

        # Pre-compute 2D PCA coordinates so dashboard GET requests are 0ms
        _update_stage("explainability", 70, "Computing 2D PCA projection coordinates and Z-score feature attributions...")
        try:
            pca_2d = PCA(n_components=2, random_state=42)
            X_pca_2d = pca_2d.fit_transform(X_scaled)
            df_analyzed["pca_x"] = np.round(X_pca_2d[:, 0], 4)
            df_analyzed["pca_y"] = np.round(X_pca_2d[:, 1], 4)
        except Exception as e:
            print(f"[WARNING] 2D PCA pre-computation error: {e}")
            df_analyzed["pca_x"] = 0.0
            df_analyzed["pca_y"] = 0.0

        # Pre-compute population feature attributions & Z-scores for every patient
        attributions_list = []
        for _, p_row in df_analyzed.iterrows():
            p_id = str(p_row.get("patient_id", ""))
            for feat, mean_val in pop_means.items():
                std_val = float(pop_stds.get(feat, 1.0))
                if std_val == 0.0 or np.isnan(std_val):
                    std_val = 1.0
                p_val = p_row.get(feat)
                if pd.notna(p_val) and isinstance(p_val, (int, float, np.number)):
                    p_val = float(p_val)
                    mean_val = float(mean_val)
                    diff = p_val - mean_val
                    z = diff / std_val
                    attributions_list.append({
                        "patient_id": p_id,
                        "feature_name": feat,
                        "patient_value": round(p_val, 4),
                        "reference_mean": round(mean_val, 4),
                        "reference_std": round(std_val, 4),
                        "z_score": round(z, 4),
                        "absolute_deviation": round(abs(diff), 4),
                        "is_abnormal": 1 if abs(z) >= 1.7 else 0
                    })

        # 6. Evaluation & Isolation Forest Comparison
        _update_stage("isolation_forest", 80, "Evaluating clustering and running Isolation Forest benchmark...")
        print(f"[5/7] Evaluating Clustering & Isolation Forest Benchmark for '{dataset_id}'...")
        eval_metrics = ModelEvaluator.evaluate_dbscan(X_scaled, labels, start_time=t0)
        eval_metrics["eps"] = chosen_eps
        eval_metrics["min_samples"] = min_samples

        iso_summary, iso_labels = ModelEvaluator.compare_with_isolation_forest(X_scaled, labels)

        # 7. Generate Dataset-Specific Visualizations
        _update_stage("explainability", 85, "Generating publication-grade diagnostic plots...")
        print(f"[6/7] Generating Isolated Publication Plots for '{dataset_name}'...")
        try:
            viz = Visualizer(dataset_id=dataset_id, dataset_name=dataset_name, source=source)
            viz.plot_k_distance(k_distances, eps=chosen_eps)
            viz.plot_pca_clusters(X_scaled, labels)
            viz.plot_anomaly_score_distribution(anomaly_scores)
            viz.plot_isolation_forest_comparison(iso_summary)
        except Exception as ex:
            print(f"[WARNING] Diagnostic plot generation deferred: {ex}")

        # 8. Persist Run and Results with dataset_id
        _update_stage("saving_results", 90, "Persisting models, analysis results, and feature attributions...")
        print(f"[7/8] Persisting Results for Dataset '{dataset_id}' to Database...")
        run_id = db_manager.save_model_run(eval_metrics, dataset_id=dataset_id)
        db_manager.save_analysis_results(df_analyzed, dataset_id=dataset_id, run_id=run_id)
        if attributions_list:
            db_manager.save_feature_attributions(attributions_list, dataset_id=dataset_id, run_id=run_id)

        # 9. Generate Clinical Decision Support Suggestions & Detailed Recommendations
        _update_stage("recommendations", 95, "Synthesizing structured clinical recommendations and specialist referrals...")
        print(f"[8/8] Generating Clinical Decision Support & Medical Recommendations for '{dataset_name}'...")
        cds_engine = ClinicalDecisionSupportEngine()
        cds_suggestions = cds_engine.generate_cohort_suggestions(
            df_analyzed=df_analyzed,
            candidate_details=candidate_details,
            pop_means=pop_means,
            pop_stds=pop_stds,
            dataset_id=dataset_id,
            run_id=run_id
        )
        db_manager.save_clinical_suggestions(cds_suggestions, dataset_id=dataset_id, run_id=run_id)

        # Generate Structured Evidence-Based Clinical Recommendations & Provenance Records
        rec_engine = ClinicalRecommendationEngine()
        detailed_recs = []
        for _, row in df_analyzed.iterrows():
            p_rec = rec_engine.generate_recommendations_for_patient(
                patient_row=row.to_dict(),
                pop_means=pop_means,
                pop_stds=pop_stds,
                dataset_id=dataset_id,
                analysis_run_id=run_id
            )
            detailed_recs.extend(p_rec.get("db_records", []))
        if detailed_recs:
            db_manager.save_clinical_recommendations(detailed_recs, dataset_id=dataset_id, run_id=run_id)

        # Mark dataset status as Analyzed
        db_manager.save_dataset_metadata({
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "source": source,
            "description": description,
            "filename": filename,
            "record_count": len(df_raw),
            "feature_count": len(df_raw.columns),
            "status": "Analyzed"
        })

        runtime = round(time.time() - t0, 2)
        results_summary = {
            "clusters_detected": eval_metrics["number_of_clusters"],
            "candidate_cases": eval_metrics["number_of_anomalies"],
            "anomaly_percentage": eval_metrics["anomaly_percentage"],
            "silhouette_score": eval_metrics["silhouette_score"],
            "runtime": runtime
        }

        # 10. Persist Analysis Metadata to Supabase PostgreSQL & Local Analysis Store
        try:
            db_manager.save_analysis_record({
                "analysis_id": dataset_id,
                "dataset_name": dataset_name,
                "dataset_filename": filename or f"{dataset_id}.csv",
                "source": source,
                "description": description,
                "total_records": len(df_raw),
                "total_clusters": eval_metrics["number_of_clusters"],
                "anomalies": eval_metrics["number_of_anomalies"],
                "anomaly_percentage": eval_metrics["anomaly_percentage"],
                "eps": chosen_eps,
                "min_samples": min_samples,
                "silhouette_score": eval_metrics.get("silhouette_score"),
                "status": "completed",
                "results_summary": results_summary
            })
        except Exception as ex:
            print(f"[WARNING] Supabase analysis record sync deferred: {ex}")

        if tracked_run_id:
            k_dist_sample = k_distances[::max(1, len(k_distances) // 100)].tolist() if hasattr(k_distances, "tolist") else list(k_distances[:100])
            db_manager.update_analysis_run(
                tracked_run_id,
                status="completed",
                stage="completed",
                progress=100,
                message=f"Analysis completed successfully in {runtime}s. Detected {eval_metrics['number_of_anomalies']} candidate rare cases.",
                results_summary=results_summary,
                k_distance_data={"k_distances": k_dist_sample, "eps": chosen_eps}
            )

        print("=" * 75)
        print(f"DATASET [{dataset_id}] PIPELINE COMPLETED IN {runtime} SECONDS!")
        print(f"Clusters Detected: {eval_metrics['number_of_clusters']}")
        print(f"Candidate Rare Cases (Noise): {eval_metrics['number_of_anomalies']} ({eval_metrics['anomaly_percentage']}%)")
        print(f"Silhouette Score: {eval_metrics['silhouette_score']}")
        print("=" * 75)

        return {
            "dataset_id": dataset_id,
            "df_analyzed": df_analyzed,
            "eval_metrics": eval_metrics,
            "iso_summary": iso_summary,
            "runtime": runtime,
            "run_id": run_id,
            "tracked_run_id": tracked_run_id
        }
    except Exception as e:
        if tracked_run_id:
            try:
                db_manager.update_analysis_run(
                    tracked_run_id,
                    status="failed",
                    stage="failed",
                    error_message=str(e),
                    message=f"Pipeline execution failed: {str(e)}"
                )
            except Exception:
                pass
        raise


if __name__ == "__main__":
    execute_full_pipeline()
