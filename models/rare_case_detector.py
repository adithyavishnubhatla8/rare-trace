import numpy as np
import pandas as pd

class RareCaseDetector:
    """
    Module responsible for post-DBSCAN anomaly analysis, candidate ranking, 
    clinical feature attribution, and explainable summary generation for ANY uploaded CSV dataset.
    """
    DISCLAIMER = (
        "This system identifies statistically unusual patient profiles using unsupervised "
        "DBSCAN anomaly detection and does not provide a medical diagnosis. Candidate "
        "cases require clinical evaluation by qualified healthcare professionals."
    )

    @staticmethod
    def analyze_candidates(df_raw, df_eng, X_scaled, labels, anomaly_scores):
        """
        Analyze patient records dynamically across all numerical features present in the CSV dataset.
        """
        df_result = df_raw.copy()
        
        # Ensure patient_id column exists
        if "patient_id" not in df_result.columns:
            found_id = None
            for col in df_result.columns:
                c_norm = col.lower().replace("_", "").replace(" ", "")
                if c_norm in ["patientid", "patient", "id", "recordid", "subjectid", "pid", "subjectcode", "caseid", "subject"]:
                    found_id = col
                    break
            if found_id:
                df_result["patient_id"] = df_result[found_id].astype(str)
            else:
                df_result["patient_id"] = [f"PAT-{10001 + i}" for i in range(len(df_result))]
        else:
            df_result["patient_id"] = df_result["patient_id"].astype(str)

        df_result["cluster_label"] = labels
        df_result["is_anomaly"] = (labels == -1).astype(int)
        df_result["anomaly_score"] = anomaly_scores

        # Identify all numerical columns in df_raw (excluding ID column)
        numeric_cols = [
            c for c in df_raw.columns 
            if pd.api.types.is_numeric_dtype(df_raw[c]) and "id" not in c.lower()
        ]

        normal_mask = labels != -1
        if not np.any(normal_mask):
            normal_mask = np.ones(len(df_raw), dtype=bool)

        if numeric_cols:
            pop_means = df_raw.loc[normal_mask, numeric_cols].mean()
            pop_stds = df_raw.loc[normal_mask, numeric_cols].std().replace(0, 1.0)
        else:
            pop_means = pd.Series(dtype=float)
            pop_stds = pd.Series(dtype=float)

        candidate_details = []
        priorities = []
        key_deviations_list = []
        interpretations = []

        for idx, row in df_result.iterrows():
            if row["is_anomaly"] == 1:
                devs = []
                dev_labels = []
                
                for col in numeric_cols:
                    val = row[col]
                    if pd.notna(val) and col in pop_means:
                        mean = pop_means[col]
                        std = pop_stds[col]
                        z = (val - mean) / std if std != 0 else 0.0
                        if abs(z) >= 1.7:
                            direction = "Elevated" if z > 0 else "Depressed"
                            col_formatted = col.replace("_", " ").title()
                            devs.append({
                                "feature": col,
                                "feature_label": col_formatted,
                                "patient_value": round(val, 2),
                                "pop_mean": round(mean, 2),
                                "z_score": round(z, 2),
                                "direction": direction
                            })
                            dev_labels.append(f"{direction} {col_formatted}")

                devs.sort(key=lambda x: abs(x["z_score"]), reverse=True)
                top_dev_labels = [d["feature_label"] for d in devs[:4]]

                score = row["anomaly_score"]
                n_devs = len(devs)
                
                if score >= 75.0 or n_devs >= 4:
                    priority = "High Priority"
                elif score >= 45.0 or n_devs >= 2:
                    priority = "Medium Priority"
                else:
                    priority = "Low Priority"

                if top_dev_labels:
                    feat_str = ", ".join(top_dev_labels[:3])
                    interp = (
                        f"Patient profile differs substantially from dominant population patterns "
                        f"with notable variations in {feat_str}."
                    )
                else:
                    interp = (
                        "Patient profile exhibits multi-variable non-linear feature isolation "
                        "differing from major cluster density centers."
                    )

                priorities.append(priority)
                key_deviations_list.append(", ".join(top_dev_labels) if top_dev_labels else "Multi-feature isolation")
                interpretations.append(interp)
                candidate_details.append(devs)

            else:
                priorities.append("Normal Profile")
                key_deviations_list.append("None")
                interpretations.append("Patient profile aligns with dominant clinical cluster pattern.")
                candidate_details.append([])

        df_result["review_priority"] = priorities
        df_result["key_abnormal_features"] = key_deviations_list
        df_result["interpretation"] = interpretations

        return df_result, candidate_details, pop_means.to_dict(), pop_stds.to_dict()
