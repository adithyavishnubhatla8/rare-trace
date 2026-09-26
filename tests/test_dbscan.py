import pytest
import numpy as np
from models.dbscan_model import DBSCANAnomalyDetector
from models.rare_case_detector import RareCaseDetector
from data.generate_dataset import generate_synthetic_dataset
from models.preprocessing import ClinicalDataPreprocessor

def test_dbscan_clustering_and_anomaly_scores():
    df = generate_synthetic_dataset(n_samples=250, seed=42)
    preprocessor = ClinicalDataPreprocessor()
    df_clean, df_eng, X_scaled = preprocessor.fit_transform(df)

    detector = DBSCANAnomalyDetector(eps=1.2, min_samples=5)
    labels = detector.fit_predict(X_scaled)

    assert len(labels) == len(X_scaled)
    assert isinstance(detector.n_clusters_, int)
    assert detector.labels_ is not None

    anomaly_scores = detector.compute_anomaly_scores(X_scaled, labels)
    assert len(anomaly_scores) == len(X_scaled)
    assert np.all(anomaly_scores >= 0.0)
    assert np.all(anomaly_scores <= 100.0)

def test_rare_case_detector_attribution():
    df = generate_synthetic_dataset(n_samples=200, seed=42)
    preprocessor = ClinicalDataPreprocessor()
    df_clean, df_eng, X_scaled = preprocessor.fit_transform(df)

    detector = DBSCANAnomalyDetector(eps=1.2, min_samples=5)
    labels = detector.fit_predict(X_scaled)
    anomaly_scores = detector.compute_anomaly_scores(X_scaled, labels)

    df_analyzed, candidate_details, pop_means, pop_stds = RareCaseDetector.analyze_candidates(
        df_clean, df_eng, X_scaled, labels, anomaly_scores
    )

    assert "review_priority" in df_analyzed.columns
    assert "key_abnormal_features" in df_analyzed.columns
    assert "interpretation" in df_analyzed.columns
