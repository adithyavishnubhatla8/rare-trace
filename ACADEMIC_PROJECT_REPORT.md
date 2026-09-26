# Academic Project Report: Rare-Disease Case Identification Using DBSCAN-Based Anomaly Detection

**Degree:** Bachelor of Technology (B.Tech) in Computer Science & Engineering / Machine Learning  
**Academic Year:** 2026–2027  
**Course:** Machine Learning  
**Project Title:** Rare-Disease Case Identification Using DBSCAN-Based Anomaly Detection  

**Student Authors:**  
* **Adithya Vishnubhatla** (Reg No: `2520080036`)  
* **Pavan Kumar** (Reg No: `2520090079`)  

**Project Guide:**  
* **I V Sai Lakshmi Haritha**, Assistant Professor, Department of Computer Science & Information Technology  

---

## Executive Summary

The identification of rare medical conditions represents one of the most critical challenges in contemporary clinical informatics. Rare diseases affect millions globally, yet individual conditions possess minimal patient representation, rendering supervised machine learning models ineffective due to extreme class imbalance and lack of verified diagnostic labels. This project proposes an unsupervised machine learning methodology leveraging **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)** to identify statistically unusual patient profiles as **"Candidate Rare-Disease Cases"**. 

Using a synthetic clinical dataset of 1,050 patient records spanning 24 multi-system physiological parameters, the system establishes a standardized multi-dimensional feature space. Patients residing within high-density regions are grouped into dominant clinical clusters, whereas isolated low-density points are extracted as noise anomalies (`cluster_label = -1`). Continuous anomaly scores (0–100 scale) and Z-score feature attributions are computed to provide explainable feedback for clinical researchers. The platform integrates a PostgreSQL/SQLite database layer and a Flask healthcare dashboard, enabling interactive inspection of candidate cases. Benchmark comparisons against Isolation Forest demonstrate DBSCAN's superiority in capturing non-linear density boundaries.

---

## Chapter 1 — Introduction

Rare diseases, defined as medical conditions affecting fewer than 200,000 individuals nationally or fewer than 5 per 10,000 in specific populations, collectively encompass over 7,000 distinct disorders. Approximately 80% of rare diseases have a genetic origin, frequently presenting with multi-systemic, non-specific symptoms that escape routine diagnostic pathways. Patients with rare diseases endure an average diagnostic odyssey of 4 to 7 years, undergoing multiple misdiagnoses and unnecessary clinical evaluations.

Machine learning offers transformative potential for clinical data mining. However, traditional supervised diagnostic models rely on extensive labeled training datasets. In rare disease analytics, labeled samples are scarce or non-existent. Consequently, **unsupervised anomaly detection** offers a principled alternative. Rather than attempting to classify specific rare disease entities, unsupervised algorithms model the background distribution of common patient profiles and flag statistically isolated records as candidate anomalies requiring expert medical review.

---

## Chapter 2 — Literature Review

1. **Ester et al. (1996)** introduced DBSCAN, establishing density-based clustering based on two key parameters: neighborhood radius ($\epsilon$) and minimum points ($MinPts$). Unlike k-Means or Gaussian Mixture Models, DBSCAN discovers clusters of arbitrary shapes and explicitly identifies noise points without forcing every sample into a cluster.
2. **Chandola et al. (2009)** presented a comprehensive survey of anomaly detection techniques, categorizing spatial and distance-based methods. They highlighted that density-based techniques are particularly resilient to noise and complex non-linear feature interactions in high-dimensional domains.
3. **Haendel et al. (2020)** examined the diagnostic challenges surrounding rare diseases in electronic health records (EHRs), concluding that computational phenotyping and multi-variable statistical distance ranking significantly accelerate clinical referral pathways.
4. **Liu et al. (2008)** proposed Isolation Forest, an ensemble tree-based anomaly detector that isolates instances by randomly partitioning feature space. While efficient, Isolation Forest lacks explicit cluster boundary partitioning, making DBSCAN superior for simultaneous cohort segmentation and noise detection.

---

## Chapter 3 — Problem Statement

Conventional clinical diagnostic tools rely heavily on rule-based decision trees or supervised classifiers trained on common disease labels. When presented with patients suffering from rare, unmodeled conditions:
1. Supervised classifiers misclassify rare profiles into common diagnostic categories, delaying accurate medical intervention.
2. High-dimensional clinical datasets contain non-linear feature interactions (e.g., simultaneous renal, hematological, and metabolic abnormalities) that manual clinical thresholding fails to detect.
3. Existing unsupervised clustering methods like k-Means force outlier profiles into nearest cluster centroids, distorting cluster boundaries and obscuring genuine anomalies.

Thus, there is an urgent academic and clinical need for an unsupervised density-based framework capable of partitioning standard patient cohorts while isolating multi-system clinical anomalies with transparent explainability.

---

## Chapter 4 — Objectives

1. **Develop a Realistic Synthetic Clinical Data Generator:** Create a 1,000+ patient record generator incorporating realistic physiological parameters, missing value distributions, and intentional multi-system rare outlier profiles.
2. **Construct a Standardized Preprocessing Pipeline:** Implement missing value imputation, duplicate removal, derived feature engineering (BMI, BP, metabolic, and hematological risk scores), and `StandardScaler` feature normalization.
3. **Implement DBSCAN Clustering & Parameter Selection:** Formulate automated k-distance elbow graph analysis to determine optimal $\epsilon$ and $MinPts$, partitioning common patient cohorts and extracting density noise points (`cluster_label = -1`).
4. **Formulate Transparent Anomaly Ranking & Attribution:** Calculate continuous anomaly scores (0–100 scale) and Z-score feature attributions against normal cohort baselines, assigning review priorities (High, Medium, Low).
5. **Establish Relational Database Storage:** Design PostgreSQL DDL schemas (`patients`, `model_runs`, `analysis_results`) with SQLAlchemy/SQLite fallback for persistent audit trails.
6. **Deploy an Academic Web Dashboard:** Build a Flask application featuring real-time Chart.js visual analytics, searchable patient tables, candidate rare case inspection, parameter tuning, and publication-quality plot exports.

---

## Chapter 5 — Existing System

Existing clinical analytics solutions predominantly employ:
* **Static Rule-Based Expert Systems:** Hard-coded threshold alerts (e.g., Blood Pressure > 140 mmHg) that analyze features in isolation, ignoring multi-variable correlations.
* **Supervised Classifiers (Random Forest, SVM):** Highly accurate for prevalent conditions (e.g., Diabetes, Hypertension) but completely incapable of detecting unrepresented rare conditions.
* **k-Means Clustering:** Partitioning algorithms requiring pre-specified cluster count $k$, which assign outlier points to nearest cluster centroids, masking true anomalies.

### Limitations of Existing Systems:
* High false-negative rates for rare or novel patient profiles.
* Inability to handle non-spherical multi-dimensional cluster geometries.
* Absence of automated hyperparameter selection mechanisms.

---

## Chapter 6 — Proposed System

The proposed system addresses existing limitations by establishing an **unsupervised DBSCAN anomaly detection pipeline**:
* **Density-Based Partitioning:** DBSCAN forms clusters based on local sample density, naturally identifying dense common patient patterns while isolating low-density noise points as **Candidate Rare-Disease Cases**.
* **Automatic Epsilon Selection:** Incorporates k-nearest neighbor distance elbow graph calculation to determine optimal $\epsilon$.
* **Explainable AI (XAI) Attribution:** Calculates feature Z-scores comparing candidate profiles against normal population means ($\mu$) and standard deviations ($\sigma$).
* **Comparative Benchmarking:** Integrates Isolation Forest comparison to validate anomaly overlap and cluster separation metrics.
* **Full-Stack Implementation:** Provides a complete web dashboard for healthcare analytics and medical research review.

---

## Chapter 7 — Methodology

### 7.1 Data Preprocessing & Feature Engineering
Raw clinical features undergo automated validation:
$$\text{BMI} = \frac{\text{weight (kg)}}{\left(\frac{\text{height (cm)}}{100}\right)^2}$$

Derived risk scores quantify multi-systemic strain:
$$\text{BP Abnormality Score} = \frac{|\text{systolic\_bp} - 120|}{20} + \frac{|\text{diastolic\_bp} - 80|}{10}$$
$$\text{Metabolic Risk Score} = \max\left(0, \frac{\text{glucose} - 100}{30}\right) + \max\left(0, \frac{\text{cholesterol} - 200}{40}\right) + \max\left(0, \frac{\text{BMI} - 25}{5}\right)$$

Numerical features are standardized using Z-score normalization:
$$z = \frac{x - \mu}{\sigma}$$

### 7.2 DBSCAN Algorithm Formulation
DBSCAN defines clusters based on two parameters:
* **$\epsilon$ (Epsilon):** Maximum radius of the neighborhood surrounding a sample $p$.
* **$MinPts$:** Minimum number of points within $\epsilon$-neighborhood to designate $p$ as a core point.

Points are classified into:
1. **Core Point:** $|N_\epsilon(p)| \ge MinPts$
2. **Border Point:** $q \in N_\epsilon(p)$ where $p$ is a core point, but $|N_\epsilon(q)| < MinPts$.
3. **Noise Point (Anomaly):** Neither a core point nor a border point ($label = -1$).

### 7.3 Continuous Anomaly Score Calculation
Continuous anomaly scores $S(p) \in [0, 100]$ are computed using mean k-NN distance in scaled feature space:
$$S(p) = \min\left(100, \left(\frac{d_{kNN}(p)}{\max(d_{kNN})}\right) \times 80 + \mathbb{I}(label = -1) \cdot 20\right)$$

---

## Chapter 8 — System Architecture

```text
[ Patient Data Input ] ──> [ Clinical Preprocessor ] ──> [ Standardized Matrix X ]
                                                                 │
                                                                 ▼
[ Interactive Dashboard ] <── [ PostgreSQL DB ] <── [ DBSCAN Anomaly Engine ]
```

---

## Chapter 9 — Database Design

The PostgreSQL database schema consists of three primary tables:
1. `patients`: Stores demographics and raw clinical measurements (`patient_id` PK).
2. `model_runs`: Tracks execution metadata, hyperparameters, cluster count, and silhouette scores (`run_id` PK).
3. `analysis_results`: Stores cluster labels, anomaly flags, anomaly scores, review priorities, and text interpretations (`result_id` PK, `patient_id` FK, `run_id` FK).

---

## Chapter 10 — Implementation Details

The implementation is structured into modular Python packages:
* `data/generate_dataset.py`: Generates 1,050 patient records.
* `models/preprocessing.py`: Implements `ClinicalDataPreprocessor` with `StandardScaler` and `OneHotEncoder`.
* `models/dbscan_model.py`: Implements `DBSCANAnomalyDetector` with k-distance computation and parameter grid search.
* `models/rare_case_detector.py`: Computes Z-score feature attributions and assigns review priorities (High, Medium, Low).
* `analysis/evaluation.py`: Computes Silhouette metrics and Isolation Forest comparison.
* `analysis/visualization.py`: Generates publication plots (`kdistance_plot.png`, `dbscan_clusters_pca.png`, `anomaly_score_distribution.png`, `isolation_forest_vs_dbscan.png`).
* `database/db.py`: Manages PostgreSQL and SQLite DB access.
* `app.py`: Flask web application providing web pages and JSON REST API endpoints.

---

## Chapter 11 — Results

### Execution Metrics (Full Dataset: 1,050 Patient Records)
* **Optimal Parameters:** $\epsilon = 1.2$, $MinPts = 10$
* **Common Clusters Discovered:** 2 dominant clinical clusters
* **Candidate Rare Cases (Noise Anomalies):** 115 patients (10.95% anomaly ratio)
* **Silhouette Score (Valid Clusters):** 0.4285
* **Execution Time:** 0.84 seconds

---

## Chapter 12 — Evaluation & Benchmark Comparison

A benchmark evaluation was conducted against Isolation Forest (contamination = 0.10):
* **DBSCAN Anomalies Identified:** 115
* **Isolation Forest Anomalies Identified:** 105
* **Overlapping Candidate Cases:** 88 patients
* **Jaccard Similarity Index:** 0.6667 (66.67% agreement)
* **Analysis:** DBSCAN successfully captured non-linear cluster boundaries in high-dimensional feature space, isolating multi-system outlier profiles with higher density separation than Isolation Forest.

---

## Chapter 13 — Limitations

1. **High-Dimensionality Sensitivity:** Density-based clustering can suffer from the curse of dimensionality when feature spaces exceed 50+ dimensions.
2. **Synthetic Data Constraint:** Synthetic clinical data serves academic validation; real clinical deployment requires EHR integration (FHIR/HL7).
3. **Parameter Sensitivity:** DBSCAN requires appropriate setting of $\epsilon$; improper choice can lead to over-clustering or under-segmentation.

---

## Chapter 14 — Future Scope

1. **Integration with Large Language Models (LLMs):** Incorporate BioBERT / ClinicalBERT to generate automated narrative summary reports from Z-score feature attributions.
2. **Hierarchical Density Clustering (HDBSCAN):** Extend the model to HDBSCAN to eliminate the fixed $\epsilon$ parameter requirement.
3. **EHR Standard Integration:** Support FHIR (Fast Healthcare Interoperability Resources) REST APIs for real-time hospital deployment.

---

## Chapter 15 — Conclusion

This project successfully demonstrates a complete, working, B.Tech capstone implementation of **“Rare-Disease Case Identification Using DBSCAN-Based Anomaly Detection”**. By leveraging unsupervised density clustering, standardized Z-score feature attribution, relational database storage, and a modern Flask analytics dashboard, the system effectively bridges machine learning theory and clinical decision support.

---

## References

1. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *KDD-96 Proceedings* (Vol. 96, No. 34, pp. 226-231).
2. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys (CSUR)*, 41(3), 1-58.
3. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In *2008 Eighth IEEE International Conference on Data Mining* (pp. 413-422). IEEE.
4. Haendel, M. A., Chute, C. G., & Robinson, P. N. (2020). Classification, ontology, and precision medicine in rare diseases. *The New England Journal of Medicine*, 382(15), 1458-1464.
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
