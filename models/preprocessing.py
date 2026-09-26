import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config

class ClinicalDataPreprocessor:
    """
    Fully dynamic data preprocessor capable of handling ANY uploaded CSV dataset.
    Automatically detects ID columns, numerical features, categorical features,
    imputes missing values, engineers dynamic risk scores, and scales features.
    """
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        
        self.id_col = "patient_id"
        self.numerical_cols = []
        self.categorical_cols = []
        self.derived_score_cols = []
        self.feature_names = []
        self.is_fitted = False

    def _detect_columns(self, df):
        """Dynamically identify ID column, numerical columns, and categorical columns."""
        cols = list(df.columns)
        
        # 1. Identify or create ID column
        found_id = None
        for col in cols:
            c_norm = col.lower().replace("_", "").replace(" ", "")
            if c_norm in ["patientid", "patient", "id", "recordid", "subjectid", "pid", "subjectcode", "caseid", "subject", "sampleid"]:
                found_id = col
                break
                
        if not found_id:
            for col in cols:
                if "id" in col.lower() or "patient" in col.lower() or "subject" in col.lower():
                    found_id = col
                    break

        if found_id:
            self.id_col = found_id
        else:
            self.id_col = "patient_id"

        # Exclude ID column from features
        feature_cols = [c for c in cols if c != found_id]
        
        num_cols = []
        cat_cols = []
        
        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                if not found_id and df[c].nunique() == len(df) and pd.api.types.is_integer_dtype(df[c]):
                    found_id = c
                    self.id_col = c
                else:
                    num_cols.append(c)
            else:
                cat_cols.append(c)
                
        self.numerical_cols = num_cols
        self.categorical_cols = cat_cols

    def _engineer_features(self, df):
        """
        Engineers derived features:
        - Specific physiological scores if standard healthcare columns are present.
        - Generalized multi-feature abnormality score for any arbitrary numerical columns.
        """
        df_eng = df.copy()
        cols = [c.lower() for c in df_eng.columns]
        col_map = {c.lower(): c for c in df_eng.columns}

        # 1. BMI Category derivation if height & weight or bmi exist
        if "bmi" in cols:
            bmi_col = col_map["bmi"]
            bmi_vals = pd.to_numeric(df_eng[bmi_col], errors='coerce').fillna(22.5)
            categories = ["Underweight" if b < 18.5 else ("Normal" if b < 25.0 else ("Overweight" if b < 30.0 else "Obese")) for b in bmi_vals]
            df_eng["bmi_category"] = categories
            if "bmi_category" not in self.categorical_cols:
                self.categorical_cols.append("bmi_category")

        # 2. Blood Pressure Abnormality Score if BP columns exist
        if "systolic_bp" in cols and "diastolic_bp" in cols:
            sys_col = col_map["systolic_bp"]
            dia_col = col_map["diastolic_bp"]
            sys_bp = pd.to_numeric(df_eng[sys_col], errors='coerce').fillna(120.0)
            dia_bp = pd.to_numeric(df_eng[dia_col], errors='coerce').fillna(80.0)
            df_eng["bp_abnormality_score"] = np.abs(sys_bp - 120.0) / 20.0 + np.abs(dia_bp - 80.0) / 10.0
            if "bp_abnormality_score" not in self.derived_score_cols:
                self.derived_score_cols.append("bp_abnormality_score")

        # 3. Metabolic Risk Score if glucose & cholesterol exist
        if "glucose" in cols and "cholesterol" in cols:
            gluc_col = col_map["glucose"]
            chol_col = col_map["cholesterol"]
            gluc = pd.to_numeric(df_eng[gluc_col], errors='coerce').fillna(95.0)
            chol = pd.to_numeric(df_eng[chol_col], errors='coerce').fillna(190.0)
            df_eng["metabolic_risk_score"] = np.maximum(0, (gluc - 100.0) / 30.0) + np.maximum(0, (chol - 200.0) / 40.0)
            if "metabolic_risk_score" not in self.derived_score_cols:
                self.derived_score_cols.append("metabolic_risk_score")

        # 4. Generalized Multi-Feature Abnormality Score for ANY arbitrary numerical features
        if self.numerical_cols:
            num_data = df_eng[self.numerical_cols].apply(pd.to_numeric, errors='coerce')
            means = num_data.mean()
            stds = num_data.std().replace(0, 1.0).fillna(1.0)
            z_scores = np.abs((num_data - means) / stds)
            df_eng["overall_abnormality_index"] = z_scores.mean(axis=1).fillna(0.0)
            if "overall_abnormality_index" not in self.derived_score_cols:
                self.derived_score_cols.append("overall_abnormality_index")

        return df_eng

    def fit_transform(self, df):
        """Fit preprocessing pipeline on ANY dataset and return clean df, engineered df, and X matrix."""
        df_clean = df.copy()
        df_clean.columns = [str(c).strip() for c in df_clean.columns]
        
        # Detect or generate ID column
        self._detect_columns(df_clean)
        
        # Guarantee patient_id column exists
        if "patient_id" not in df_clean.columns:
            if self.id_col and self.id_col in df_clean.columns:
                df_clean["patient_id"] = df_clean[self.id_col].astype(str)
            else:
                df_clean["patient_id"] = [f"PAT-{10001 + i}" for i in range(len(df_clean))]
        else:
            df_clean["patient_id"] = df_clean["patient_id"].astype(str)
            
        # Drop exact duplicate rows based on feature columns
        feat_subset = self.numerical_cols + self.categorical_cols
        if feat_subset:
            df_clean = df_clean.drop_duplicates(subset=feat_subset).reset_index(drop=True)
            
        # Engineer dynamic features
        df_eng = self._engineer_features(df_clean)
        
        # Impute missing numerical features
        all_num_cols = self.numerical_cols + self.derived_score_cols
        if all_num_cols:
            df_eng[all_num_cols] = df_eng[all_num_cols].apply(pd.to_numeric, errors='coerce')
            df_eng[all_num_cols] = self.num_imputer.fit_transform(df_eng[all_num_cols])
            num_scaled = self.scaler.fit_transform(df_eng[all_num_cols])
        else:
            num_scaled = np.empty((len(df_eng), 0))
            
        # Impute and encode categorical features
        if self.categorical_cols:
            df_eng[self.categorical_cols] = self.cat_imputer.fit_transform(df_eng[self.categorical_cols].astype(str))
            cat_encoded = self.encoder.fit_transform(df_eng[self.categorical_cols])
            encoded_cat_names = self.encoder.get_feature_names_out(self.categorical_cols).tolist()
        else:
            cat_encoded = np.empty((len(df_eng), 0))
            encoded_cat_names = []

        # Combine matrices
        if num_scaled.shape[1] > 0 and cat_encoded.shape[1] > 0:
            X_processed = np.hstack([num_scaled, cat_encoded])
        elif num_scaled.shape[1] > 0:
            X_processed = num_scaled
        elif cat_encoded.shape[1] > 0:
            X_processed = cat_encoded
        else:
            # Fallback single feature if dataset has no features
            X_processed = np.arange(len(df_eng)).reshape(-1, 1).astype(float)
            all_num_cols = ["index_feature"]

        self.feature_names = all_num_cols + encoded_cat_names
        self.is_fitted = True

        return df_clean, df_eng, X_processed

    def transform(self, df):
        """Transform new patient data using already fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before calling transform().")
            
        df_clean = df.copy()
        df_clean.columns = [str(c).strip() for c in df_clean.columns]
        df_eng = self._engineer_features(df_clean)
        all_num_cols = self.numerical_cols + self.derived_score_cols
        
        if all_num_cols:
            df_eng[all_num_cols] = df_eng[all_num_cols].apply(pd.to_numeric, errors='coerce')
            df_eng[all_num_cols] = self.num_imputer.transform(df_eng[all_num_cols])
            num_scaled = self.scaler.transform(df_eng[all_num_cols])
        else:
            num_scaled = np.empty((len(df_eng), 0))
            
        if self.categorical_cols:
            df_eng[self.categorical_cols] = self.cat_imputer.transform(df_eng[self.categorical_cols].astype(str))
            cat_encoded = self.encoder.transform(df_eng[self.categorical_cols])
        else:
            cat_encoded = np.empty((len(df_eng), 0))
            
        if num_scaled.shape[1] > 0 and cat_encoded.shape[1] > 0:
            X_processed = np.hstack([num_scaled, cat_encoded])
        elif num_scaled.shape[1] > 0:
            X_processed = num_scaled
        elif cat_encoded.shape[1] > 0:
            X_processed = cat_encoded
        else:
            X_processed = np.arange(len(df_eng)).reshape(-1, 1).astype(float)
            
        return df_eng, X_processed

    def save(self, filepath=None):
        """Save fitted preprocessor pipeline to disk."""
        if filepath is None:
            filepath = Config.MODELS_DIR / "preprocessor.joblib"
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath=None):
        """Load fitted preprocessor pipeline from disk."""
        if filepath is None:
            filepath = Config.MODELS_DIR / "preprocessor.joblib"
        return joblib.load(filepath)
