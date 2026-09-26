import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config

class Visualizer:
    """
    Generates academic publication-quality figures and plots for each independent dataset
    using a clean clinical light theme. Never plots multiple datasets together.
    """
    def __init__(self, dataset_id="default", dataset_name="Clinical Dataset", source="Standard Cohort", output_dir=None):
        self.dataset_id = str(dataset_id)
        self.dataset_name = str(dataset_name)
        self.source = str(source)

        if output_dir is None:
            self.output_dir = Path(Config.PLOTS_DIR) / self.dataset_id
        else:
            self.output_dir = Path(output_dir)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            pass
        
        # Also ensure static fallback path exists if needed
        self.static_dir = Config.BASE_DIR / "static" / "plots" / self.dataset_id
        try:
            self.static_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            pass
        
        # Clinical clean light theme settings
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
        plt.rcParams['figure.facecolor'] = '#ffffff'
        plt.rcParams['axes.facecolor'] = '#ffffff'
        plt.rcParams['axes.edgecolor'] = '#cbd5e1'
        plt.rcParams['axes.labelcolor'] = '#334155'
        plt.rcParams['text.color'] = '#0f172a'
        plt.rcParams['xtick.color'] = '#475569'
        plt.rcParams['ytick.color'] = '#475569'
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['grid.color'] = '#f1f5f9'

    def _save_figure(self, fig, filename):
        """Save plot to outputs directory and static web directory safely."""
        path1 = self.output_dir / filename
        try:
            fig.savefig(path1, bbox_inches='tight', facecolor='#ffffff')
        except Exception as e:
            print(f"[WARNING] Could not save plot to output_dir: {e}")

        try:
            path2 = self.static_dir / filename
            fig.savefig(path2, bbox_inches='tight', facecolor='#ffffff')
        except Exception:
            pass  # Expected on read-only serverless filesystems
            
        plt.close(fig)
        return str(path1)

    def plot_k_distance(self, k_distances, eps=Config.DEFAULT_EPS, save_filename="kdistance_plot.png"):
        """Generate and save K-distance elbow curve plot for this dataset only."""
        fig = plt.figure(figsize=(9, 5), dpi=300)
        plt.plot(k_distances, color='#0284c7', linewidth=2.5, label='k-distance curve')
        plt.axhline(y=eps, color='#e11d48', linestyle='--', linewidth=1.8, label=f'Chosen eps = {eps}')
        
        title_str = (
            f"K-Distance Elbow Graph for DBSCAN Epsilon (eps) Selection\n"
            f"[Dataset: {self.dataset_name} | Source: {self.source}]"
        )
        plt.title(title_str, fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Patient Samples Sorted by k-NN Distance', fontsize=11)
        plt.ylabel('k-NN Distance (Standardized Space)', fontsize=11)
        plt.legend(loc='upper left', frameon=True, framealpha=0.9, facecolor='#ffffff', edgecolor='#cbd5e1')
        plt.tight_layout()
        
        return self._save_figure(fig, save_filename)

    def plot_pca_clusters(self, X, labels, save_filename="dbscan_clusters_pca.png"):
        """Generate 2D PCA projection scatter plot of DBSCAN clusters and noise for this dataset only."""
        fig = plt.figure(figsize=(10, 6.5), dpi=300)
        
        if X.shape[1] < 2:
            X_pca = np.column_stack([X[:, 0], np.zeros(len(X))])
            var_explained = 100.0
        else:
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X)
            var_explained = float(np.sum(pca.explained_variance_ratio_) * 100.0)

        unique_labels = sorted(list(set(labels)))
        cmap = plt.get_cmap("tab10")
        palette = [cmap(i % 10) for i in range(max(1, len(unique_labels)))]
        
        for idx, l in enumerate(unique_labels):
            mask = labels == l
            if l == -1:
                plt.scatter(
                    X_pca[mask, 0], X_pca[mask, 1], 
                    c='#e11d48', label='Candidate Rare Cases (Noise)', 
                    s=55, alpha=0.9, marker='x', linewidths=2.0, zorder=5
                )
            else:
                plt.scatter(
                    X_pca[mask, 0], X_pca[mask, 1], 
                    label=f'Cluster {l}', 
                    s=35, alpha=0.7, zorder=3,
                    color=palette[idx % len(palette)]
                )

        title_str = (
            f"2D PCA Projection of Patient Clusters & Anomalies (Var: {var_explained:.1f}%)\n"
            f"[Dataset: {self.dataset_name} | Source: {self.source}]"
        )
        plt.title(title_str, fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Principal Component 1', fontsize=11)
        plt.ylabel('Principal Component 2', fontsize=11)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, framealpha=0.9, facecolor='#ffffff', edgecolor='#cbd5e1')
        plt.tight_layout()

        self._save_figure(fig, save_filename)
        return str(self.output_dir / save_filename), X_pca

    def plot_anomaly_score_distribution(self, anomaly_scores, save_filename="anomaly_score_distribution.png"):
        """Plot histogram distribution of anomaly scores for this dataset only."""
        fig = plt.figure(figsize=(9, 5), dpi=300)
        plt.hist(anomaly_scores, bins=30, color='#0284c7', edgecolor='#ffffff', linewidth=0.8, alpha=0.85)
        plt.axvline(x=75.0, color='#e11d48', linestyle='--', linewidth=1.5, label='High Priority Threshold (75.0)')
        plt.axvline(x=45.0, color='#d97706', linestyle=':', linewidth=1.5, label='Medium Priority Threshold (45.0)')

        title_str = (
            f"Distribution of Patient Anomaly Scores (0 - 100 Scale)\n"
            f"[Dataset: {self.dataset_name} | Source: {self.source}]"
        )
        plt.title(title_str, fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Calculated Anomaly Score', fontsize=11)
        plt.ylabel('Patient Count', fontsize=11)
        plt.legend(loc='upper right', frameon=True, framealpha=0.9, facecolor='#ffffff', edgecolor='#cbd5e1')
        plt.tight_layout()

        return self._save_figure(fig, save_filename)

    def plot_isolation_forest_comparison(self, comp_summary, save_filename="isolation_forest_vs_dbscan.png"):
        """Plot bar comparison between DBSCAN and Isolation Forest for this dataset only."""
        fig = plt.figure(figsize=(8, 5), dpi=300)
        
        categories = ['DBSCAN Anomalies', 'Isolation Forest Anomalies', 'Overlapping Candidates']
        counts = [
            comp_summary.get('dbscan_anomalies', 0), 
            comp_summary.get('isolation_forest_anomalies', 0), 
            comp_summary.get('overlapping_anomalies', 0)
        ]
        colors = ['#0284c7', '#4f46e5', '#0d9488']

        bars = plt.bar(categories, counts, color=colors, width=0.45, edgecolor='#cbd5e1', linewidth=0.8)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{int(yval)}", ha='center', va='bottom', fontweight='bold', color='#0f172a')

        title_str = (
            f"DBSCAN vs. Isolation Forest Anomaly Counts\n"
            f"[Dataset: {self.dataset_name} | Source: {self.source}]"
        )
        plt.title(title_str, fontsize=12, fontweight='bold', pad=15)
        plt.ylabel('Number of Anomalous Patients Detected', fontsize=11)
        max_c = max(counts) if counts and max(counts) > 0 else 10
        plt.ylim(0, max_c * 1.25)
        plt.tight_layout()

        return self._save_figure(fig, save_filename)
