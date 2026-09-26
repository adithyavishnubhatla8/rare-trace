import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure parent directory is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config

def generate_synthetic_dataset(n_samples=1000, seed=42):
    """
    Generate realistic synthetic clinical dataset containing normal population clusters
    and intentional rare outlier cases for DBSCAN anomaly detection academic demonstration.
    """
    np.random.seed(seed)
    
    # 88% common population patterns, 12% rare outlier patterns
    n_common = int(n_samples * 0.88)
    n_rare = n_samples - n_common

    # Common cluster 1: General Healthy/Mild Symptoms (50%)
    n_c1 = int(n_common * 0.6)
    # Common cluster 2: Metabolic/Cardiovascular Common Risk (40%)
    n_c2 = n_common - n_c1

    # Cluster 1 features
    age_c1 = np.random.normal(42, 12, n_c1).clip(18, 75)
    gender_c1 = np.random.choice(["Male", "Female"], size=n_c1, p=[0.49, 0.51])
    height_c1 = np.random.normal(168, 9, n_c1).clip(145, 195)
    weight_c1 = np.random.normal(70, 12, n_c1).clip(45, 110)
    heart_rate_c1 = np.random.normal(74, 8, n_c1).clip(58, 95)
    systolic_c1 = np.random.normal(120, 10, n_c1).clip(95, 142)
    diastolic_c1 = np.random.normal(78, 6, n_c1).clip(60, 92)
    temp_c1 = np.random.normal(36.6, 0.3, n_c1).clip(35.8, 37.4)
    hemo_c1 = np.random.normal(14.2, 1.1, n_c1).clip(11.5, 17.2)
    glucose_c1 = np.random.normal(95, 12, n_c1).clip(70, 135)
    chol_c1 = np.random.normal(190, 25, n_c1).clip(140, 245)
    creat_c1 = np.random.normal(0.9, 0.15, n_c1).clip(0.6, 1.3)
    wbc_c1 = np.random.normal(6800, 1200, n_c1).clip(4200, 10500)
    platelet_c1 = np.random.normal(260, 45, n_c1).clip(150, 390)
    o2_c1 = np.random.normal(98.2, 0.8, n_c1).clip(95.0, 100.0)
    symptom_cnt_c1 = np.random.poisson(1.2, n_c1).clip(0, 4)
    med_cnt_c1 = np.random.poisson(1.0, n_c1).clip(0, 4)
    chronic_cnt_c1 = np.random.poisson(0.5, n_c1).clip(0, 2)
    hosp_vis_c1 = np.random.poisson(0.8, n_c1).clip(0, 3)
    fam_score_c1 = np.random.beta(2, 5, n_c1)
    duration_c1 = np.random.exponential(14, n_c1).clip(1, 60).astype(int)
    diag_c1 = np.random.choice(["General", "Cardiovascular", "Metabolic"], size=n_c1, p=[0.6, 0.2, 0.2])

    # Cluster 2 features (Metabolic/Cardio profile)
    age_c2 = np.random.normal(61, 10, n_c2).clip(35, 88)
    gender_c2 = np.random.choice(["Male", "Female"], size=n_c2, p=[0.52, 0.48])
    height_c2 = np.random.normal(166, 8, n_c2).clip(142, 190)
    weight_c2 = np.random.normal(84, 14, n_c2).clip(55, 125)
    heart_rate_c2 = np.random.normal(82, 10, n_c2).clip(62, 108)
    systolic_c2 = np.random.normal(144, 14, n_c2).clip(115, 175)
    diastolic_c2 = np.random.normal(91, 8, n_c2).clip(72, 110)
    temp_c2 = np.random.normal(36.7, 0.3, n_c2).clip(35.9, 37.6)
    hemo_c2 = np.random.normal(13.4, 1.4, n_c2).clip(10.2, 16.5)
    glucose_c2 = np.random.normal(148, 35, n_c2).clip(90, 240)
    chol_c2 = np.random.normal(235, 35, n_c2).clip(170, 320)
    creat_c2 = np.random.normal(1.15, 0.25, n_c2).clip(0.7, 1.7)
    wbc_c2 = np.random.normal(7600, 1500, n_c2).clip(4500, 11500)
    platelet_c2 = np.random.normal(240, 50, n_c2).clip(140, 380)
    o2_c2 = np.random.normal(96.8, 1.2, n_c2).clip(93.0, 99.5)
    symptom_cnt_c2 = np.random.poisson(2.5, n_c2).clip(1, 6)
    med_cnt_c2 = np.random.poisson(3.2, n_c2).clip(1, 7)
    chronic_cnt_c2 = np.random.poisson(1.8, n_c2).clip(0, 4)
    hosp_vis_c2 = np.random.poisson(2.1, n_c2).clip(0, 6)
    fam_score_c2 = np.random.beta(3, 4, n_c2)
    duration_c2 = np.random.exponential(45, n_c2).clip(5, 180).astype(int)
    diag_c2 = np.random.choice(["Metabolic", "Cardiovascular", "Respiratory"], size=n_c2, p=[0.45, 0.45, 0.10])

    # Rare Outlier Cases (12%) with intentional multi-system abnormal profiles
    age_r = np.random.uniform(19, 84, n_rare)
    gender_r = np.random.choice(["Male", "Female", "Other"], size=n_rare, p=[0.48, 0.48, 0.04])
    height_r = np.random.normal(167, 10, n_rare).clip(140, 195)
    weight_r = np.random.uniform(42, 135, n_rare)
    
    # Rare profiles: combinations of extreme blood parameters, unusual vitals, or rare multisystem symptoms
    half_r = n_rare // 2
    rem_r = n_rare - half_r

    heart_rate_r = np.concatenate([np.random.uniform(38, 52, half_r), np.random.uniform(125, 160, rem_r)])
    systolic_r = np.concatenate([np.random.uniform(75, 92, half_r), np.random.uniform(185, 225, rem_r)])
    diastolic_r = np.concatenate([np.random.uniform(45, 55, half_r), np.random.uniform(115, 135, rem_r)])
    temp_r = np.random.uniform(38.6, 40.8, n_rare)
    hemo_r = np.concatenate([np.random.uniform(3.8, 7.5, half_r), np.random.uniform(18.5, 22.0, rem_r)])
    glucose_r = np.random.uniform(320, 510, n_rare)
    chol_r = np.random.uniform(360, 580, n_rare)
    creat_r = np.random.uniform(3.5, 9.8, n_rare) # Extreme renal failure / rare metabolic crisis
    wbc_r = np.concatenate([np.random.uniform(1200, 2800, half_r), np.random.uniform(28000, 52000, rem_r)])
    platelet_r = np.concatenate([np.random.uniform(15, 55, half_r), np.random.uniform(650, 950, rem_r)])
    o2_r = np.random.uniform(78.0, 90.5, n_rare)
    symptom_cnt_r = np.random.randint(6, 15, n_rare)
    med_cnt_r = np.random.randint(5, 12, n_rare)
    chronic_cnt_r = np.random.randint(3, 7, n_rare)
    hosp_vis_r = np.random.randint(5, 14, n_rare)
    fam_score_r = np.random.uniform(0.75, 1.0, n_rare)
    duration_r = np.random.randint(120, 450, n_rare)
    diag_r = np.random.choice(["Rare Candidate", "Renal", "Hematological", "Respiratory"], size=n_rare, p=[0.4, 0.2, 0.2, 0.2])

    # Combine all arrays
    age = np.concatenate([age_c1, age_c2, age_r])
    gender = np.concatenate([gender_c1, gender_c2, gender_r])
    height = np.concatenate([height_c1, height_c2, height_r])
    weight = np.concatenate([weight_c1, weight_c2, weight_r])
    heart_rate = np.concatenate([heart_rate_c1, heart_rate_c2, heart_rate_r])
    systolic_bp = np.concatenate([systolic_c1, systolic_c2, systolic_r])
    diastolic_bp = np.concatenate([diastolic_c1, diastolic_c2, diastolic_r])
    temperature = np.concatenate([temp_c1, temp_c2, temp_r])
    hemoglobin = np.concatenate([hemo_c1, hemo_c2, hemo_r])
    glucose = np.concatenate([glucose_c1, glucose_c2, glucose_r])
    cholesterol = np.concatenate([chol_c1, chol_c2, chol_r])
    creatinine = np.concatenate([creat_c1, creat_c2, creat_r])
    wbc_count = np.concatenate([wbc_c1, wbc_c2, wbc_r])
    platelet_count = np.concatenate([platelet_c1, platelet_c2, platelet_r])
    oxygen_saturation = np.concatenate([o2_c1, o2_c2, o2_r])
    symptom_count = np.concatenate([symptom_cnt_c1, symptom_cnt_c2, symptom_cnt_r])
    medication_count = np.concatenate([med_cnt_c1, med_cnt_c2, med_cnt_r])
    chronic_condition_count = np.concatenate([chronic_cnt_c1, chronic_cnt_c2, chronic_cnt_r])
    hospital_visits = np.concatenate([hosp_vis_c1, hosp_vis_c2, hosp_vis_r])
    family_history_score = np.concatenate([fam_score_c1, fam_score_c2, fam_score_r])
    symptom_duration_days = np.concatenate([duration_c1, duration_c2, duration_r])
    diagnosis_category = np.concatenate([diag_c1, diag_c2, diag_r])

    # Calculate exact BMI: weight (kg) / (height (m) ^ 2)
    bmi = weight / ((height / 100.0) ** 2)

    # Generate patient IDs PAT-10001, PAT-10002...
    patient_ids = [f"PAT-{10001 + i}" for i in range(n_samples)]

    df = pd.DataFrame({
        "patient_id": patient_ids,
        "age": np.round(age, 1),
        "gender": gender,
        "height": np.round(height, 1),
        "weight": np.round(weight, 1),
        "bmi": np.round(bmi, 2),
        "heart_rate": np.round(heart_rate, 1),
        "systolic_bp": np.round(systolic_bp, 1),
        "diastolic_bp": np.round(diastolic_bp, 1),
        "temperature": np.round(temperature, 2),
        "hemoglobin": np.round(hemoglobin, 2),
        "glucose": np.round(glucose, 1),
        "cholesterol": np.round(cholesterol, 1),
        "creatinine": np.round(creatinine, 2),
        "wbc_count": np.round(wbc_count, 1),
        "platelet_count": np.round(platelet_count, 1),
        "oxygen_saturation": np.round(oxygen_saturation, 1),
        "symptom_count": symptom_count,
        "medication_count": medication_count,
        "chronic_condition_count": chronic_condition_count,
        "hospital_visits": hospital_visits,
        "family_history_score": np.round(family_history_score, 3),
        "symptom_duration_days": symptom_duration_days,
        "diagnosis_category": diagnosis_category
    })

    # Introduce a small number of realistic missing values (~1-2%) to test missing value imputation pipeline
    mask_missing_glucose = np.random.choice([True, False], size=n_samples, p=[0.015, 0.985])
    mask_missing_chol = np.random.choice([True, False], size=n_samples, p=[0.015, 0.985])
    mask_missing_diag = np.random.choice([True, False], size=n_samples, p=[0.01, 0.99])
    
    df.loc[mask_missing_glucose, "glucose"] = np.nan
    df.loc[mask_missing_chol, "cholesterol"] = np.nan
    df.loc[mask_missing_diag, "diagnosis_category"] = np.nan

    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

def save_synthetic_dataset():
    """Generate and save the synthetic clinical dataset."""
    df = generate_synthetic_dataset(n_samples=1050, seed=42)
    output_path = Config.DATASET_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Synthetic clinical dataset saved to {output_path} ({len(df)} records).")
    return df

if __name__ == "__main__":
    save_synthetic_dataset()
