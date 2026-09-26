import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config

class DBSCANAnomalyDetector:
    """
    Dynamic DBSCAN Anomaly Detector supporting automatic epsilon selection,
    parameter grid search, and guaranteed anomaly candidate isolation
    for ANY uploaded CSV feature matrix.
    """
    def __init__(self, eps=Config.DEFAULT_EPS, min_samples=Config.DEFAULT_MIN_SAMPLES):
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.labels_ = None
        self.n_clusters_ = 0
        self.n_anomalies_ = 0
        self.silhouette_score_ = None
        self.X_train_ = None

    def auto_select_eps(self, X, min_samples=None):
        """
        Dynamically determine optimal epsilon (eps) for ANY feature matrix X
        using k-NN distance percentiles tailored for density partitioning.
        """
        if min_samples is None:
            min_samples = self.min_samples
        k = max(2, min(min_samples, len(X) - 1))
        
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X)
        distances, _ = nbrs.kneighbors(X)
        k_dists = np.sort(distances[:, k - 1])
        
        # 70th percentile provides balanced density boundary
        opt_eps = float(np.percentile(k_dists, 70))
        return max(0.2, round(opt_eps, 2))

    def compute_k_distances(self, X, k=None):
        """Compute sorted k-nearest neighbor distances for the K-distance graph."""
        if k is None:
            k = max(2, min(self.min_samples, len(X) - 1))
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X)
        distances, _ = nbrs.kneighbors(X)
        k_distances = np.sort(distances[:, k - 1])
        return k_distances

    def grid_search_parameters(self, X, eps_values=None, min_samples_values=None):
        """
        Evaluate multiple DBSCAN parameter combinations dynamically adapted to matrix scale.
        """
        base_eps = self.auto_select_eps(X)
        if eps_values is None:
            eps_values = [round(base_eps * factor, 2) for factor in [0.4, 0.7, 1.0, 1.3, 1.6]]
        if min_samples_values is None:
            min_samples_values = [5, 10, 15]

        results = []

        for eps in set(eps_values):
            for min_samples in min_samples_values:
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_anomalies = np.sum(labels == -1)
                anomaly_pct = (n_anomalies / len(X)) * 100.0
                
                if n_clusters >= 2 and len(set(labels[labels != -1])) >= 2:
                    valid_mask = labels != -1
                    if np.sum(valid_mask) > n_clusters:
                        sil_score = float(silhouette_score(X[valid_mask], labels[valid_mask]))
                    else:
                        sil_score = -1.0
                else:
                    sil_score = -1.0

                results.append({
                    "eps": eps,
                    "min_samples": min_samples,
                    "number_of_clusters": n_clusters,
                    "number_of_anomalies": n_anomalies,
                    "anomaly_percentage": round(anomaly_pct, 2),
                    "silhouette_score": round(sil_score, 4) if sil_score != -1.0 else "N/A"
                })

        return pd.DataFrame(results).sort_values("eps").reset_index(drop=True)

    def fit_predict(self, X):
        """
        Fit DBSCAN clustering with adaptive tuning guaranteeing candidate rare case detection
        for ANY uploaded dataset.
        """
        self.X_train_ = X
        
        if self.eps is None or self.eps == 0:
            self.eps = self.auto_select_eps(X)

        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = self.model.fit_predict(X)
        
        unique_labels = set(self.labels_)
        self.n_clusters_ = len(unique_labels) - (1 if -1 in self.labels_ else 0)
        self.n_anomalies_ = int(np.sum(self.labels_ == -1))

        target_min_anomalies = max(1, int(0.04 * len(X)))
        target_max_anomalies = int(0.25 * len(X))

        # Iteratively adapt eps if 0 anomalies are found (eps too large) or >25% anomalies (eps too small)
        attempts = 0
        while (self.n_anomalies_ < target_min_anomalies or self.n_anomalies_ > target_max_anomalies) and attempts < 8:
            attempts += 1
            if self.n_anomalies_ < target_min_anomalies:
                self.eps = round(self.eps * 0.82, 2)
            else:
                self.eps = round(self.eps * 1.25, 2)

            if self.eps <= 0.1:
                self.eps = 0.1
                break

            self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            self.labels_ = self.model.fit_predict(X)
            unique_labels = set(self.labels_)
            self.n_clusters_ = len(unique_labels) - (1 if -1 in self.labels_ else 0)
            self.n_anomalies_ = int(np.sum(self.labels_ == -1))

        valid_mask = self.labels_ != -1
        if self.n_clusters_ >= 2 and np.sum(valid_mask) > self.n_clusters_:
            self.silhouette_score_ = float(silhouette_score(X[valid_mask], self.labels_[valid_mask]))
        else:
            self.silhouette_score_ = 0.0

        return self.labels_

    def compute_anomaly_scores(self, X, labels):
        """Calculate continuous anomaly scores (0-100 scale) dynamically."""
        non_noise_mask = labels != -1
        if not np.any(non_noise_mask):
            centroid = np.mean(X, axis=0)
            dists = np.linalg.norm(X - centroid, axis=1)
            max_d = np.max(dists) if np.max(dists) > 0 else 1.0
            return np.round((dists / max_d) * 100.0, 2)

        k_nbrs = min(10, max(1, np.sum(non_noise_mask)))
        nbrs = NearestNeighbors(n_neighbors=k_nbrs, metric="euclidean").fit(X[non_noise_mask])
        distances, _ = nbrs.kneighbors(X)
        mean_k_dist = np.mean(distances, axis=1)
        
        scores = np.zeros(len(X))
        max_dist = np.max(mean_k_dist) if np.max(mean_k_dist) > 0 else 1.0
        
        for i in range(len(X)):
            raw_score = (mean_k_dist[i] / max_dist) * 80.0
            if labels[i] == -1:
                scores[i] = min(100.0, raw_score + 20.0)
            else:
                scores[i] = max(5.0, raw_score * 0.75)
                
        return np.round(scores, 2)

    def save(self, filepath=None):
        """Save fitted DBSCAN detector object."""
        if filepath is None:
            filepath = Config.MODELS_DIR / "dbscan_model.joblib"
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath=None):
        """Load fitted DBSCAN detector object."""
        if filepath is None:
            filepath = Config.MODELS_DIR / "dbscan_model.joblib"
        return joblib.load(filepath)
