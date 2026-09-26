import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, jaccard_score
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config

class ModelEvaluator:
    """
    Evaluates DBSCAN clustering performance metrics and provides comparative 
    benchmark analysis against Isolation Forest anomaly detection algorithm.
    """
    @staticmethod
    def evaluate_dbscan(X, labels, start_time=None):
        """Calculate clustering and anomaly metrics for DBSCAN."""
        if start_time:
            runtime = round(time.time() - start_time, 4)
        else:
            runtime = 0.0

        n_samples = len(X)
        n_anomalies = int(np.sum(labels == -1))
        anomaly_pct = round((n_anomalies / n_samples) * 100.0, 2)
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)

        # Cluster size distribution
        cluster_sizes = {}
        for l in unique_labels:
            if l != -1:
                cluster_sizes[f"Cluster {l}"] = int(np.sum(labels == l))

        # Silhouette score computation
        valid_mask = labels != -1
        if n_clusters >= 2 and np.sum(valid_mask) > n_clusters:
            sil_score = round(float(silhouette_score(X[valid_mask], labels[valid_mask])), 4)
        else:
            sil_score = 0.0

        return {
            "algorithm": "DBSCAN",
            "dataset_size": n_samples,
            "number_of_clusters": n_clusters,
            "number_of_anomalies": n_anomalies,
            "anomaly_percentage": anomaly_pct,
            "silhouette_score": sil_score,
            "cluster_sizes": cluster_sizes,
            "execution_time_sec": runtime
        }

    @staticmethod
    def compare_with_isolation_forest(X, dbscan_labels, contamination=0.10):
        """
        Compare DBSCAN anomaly detection against Isolation Forest on the same dataset.
        Calculates execution time, anomaly counts, and Jaccard overlap ratio.
        """
        # Fit Isolation Forest
        t0 = time.time()
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso_preds = iso.fit_predict(X)
        t_iso = round(time.time() - t0, 4)

        # Convert predictions to binary flags (1 = anomaly, 0 = normal)
        dbscan_binary = (dbscan_labels == -1).astype(int)
        iso_binary = (iso_preds == -1).astype(int)

        # Overlap analysis
        both_anomaly = np.sum((dbscan_binary == 1) & (iso_binary == 1))
        dbscan_only = np.sum((dbscan_binary == 1) & (iso_binary == 0))
        iso_only = np.sum((dbscan_binary == 0) & (iso_binary == 1))
        neither = np.sum((dbscan_binary == 0) & (iso_binary == 0))

        union = np.sum((dbscan_binary == 1) | (iso_binary == 1))
        jaccard_similarity = round(float(both_anomaly / union), 4) if union > 0 else 0.0
        overlap_pct = round((both_anomaly / max(1, np.sum(dbscan_binary == 1))) * 100.0, 2)

        comparison_summary = {
            "dbscan_anomalies": int(np.sum(dbscan_binary == 1)),
            "isolation_forest_anomalies": int(np.sum(iso_binary == 1)),
            "overlapping_anomalies": int(both_anomaly),
            "dbscan_unique_anomalies": int(dbscan_only),
            "isolation_forest_unique_anomalies": int(iso_only),
            "jaccard_similarity": jaccard_similarity,
            "dbscan_overlap_percentage": overlap_pct,
            "isolation_forest_runtime_sec": t_iso
        }

        return comparison_summary, iso_binary
