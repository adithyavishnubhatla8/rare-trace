-- PostgreSQL Database Schema for RARETRACE Multi-Dataset Anomaly Detection Project

CREATE TABLE IF NOT EXISTS analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id TEXT UNIQUE NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_filename TEXT NOT NULL,
    source TEXT DEFAULT 'User Upload',
    description TEXT,
    total_records INTEGER NOT NULL,
    total_clusters INTEGER NOT NULL,
    anomalies INTEGER NOT NULL,
    anomaly_percentage DOUBLE PRECISION NOT NULL,
    eps DOUBLE PRECISION,
    min_samples INTEGER,
    silhouette_score DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'completed',
    results_summary JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_id ON analyses(analysis_id);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id VARCHAR(64) PRIMARY KEY,
    dataset_name VARCHAR(128) NOT NULL,
    source VARCHAR(128) NOT NULL,
    description TEXT,
    filename VARCHAR(256),
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    record_count INT DEFAULT 0,
    feature_count INT DEFAULT 0,
    status VARCHAR(32) DEFAULT 'Analyzed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patients (
    dataset_id VARCHAR(64) NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    patient_id VARCHAR(64) NOT NULL,
    data_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (dataset_id, patient_id)
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id SERIAL PRIMARY KEY,
    dataset_id VARCHAR(64) NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    algorithm VARCHAR(64) NOT NULL DEFAULT 'DBSCAN',
    eps NUMERIC(6,3) NOT NULL,
    min_samples INT NOT NULL,
    number_of_clusters INT NOT NULL,
    number_of_anomalies INT NOT NULL,
    dataset_size INT NOT NULL,
    silhouette_score NUMERIC(6,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analysis_results (
    result_id SERIAL PRIMARY KEY,
    dataset_id VARCHAR(64) NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    patient_id VARCHAR(64) NOT NULL,
    cluster_label INT NOT NULL,
    is_anomaly INT NOT NULL DEFAULT 0,
    anomaly_score NUMERIC(5,2) NOT NULL DEFAULT 0.0,
    review_priority VARCHAR(32) DEFAULT 'Normal Profile',
    key_abnormal_features TEXT,
    interpretation TEXT,
    pca_x NUMERIC(8,4) DEFAULT 0.0,
    pca_y NUMERIC(8,4) DEFAULT 0.0,
    run_id INT REFERENCES model_runs(run_id) ON DELETE SET NULL,
    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id, patient_id) REFERENCES patients(dataset_id, patient_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_patients_dataset ON patients(dataset_id);
CREATE INDEX IF NOT EXISTS idx_analysis_dataset ON analysis_results(dataset_id);
CREATE INDEX IF NOT EXISTS idx_analysis_anomaly ON analysis_results(dataset_id, is_anomaly);
CREATE INDEX IF NOT EXISTS idx_analysis_patient ON analysis_results(dataset_id, patient_id);
CREATE INDEX IF NOT EXISTS idx_analysis_score ON analysis_results(dataset_id, anomaly_score DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_priority ON analysis_results(dataset_id, review_priority);
CREATE INDEX IF NOT EXISTS idx_model_runs_dataset ON model_runs(dataset_id);

CREATE TABLE IF NOT EXISTS clinical_suggestions (
    suggestion_id SERIAL PRIMARY KEY,
    dataset_id VARCHAR(64) NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    patient_id VARCHAR(64) NOT NULL,
    analysis_run_id INT REFERENCES model_runs(run_id) ON DELETE CASCADE,
    clinical_priority VARCHAR(32) NOT NULL,
    primary_domain VARCHAR(64) NOT NULL,
    all_domains_json TEXT NOT NULL,
    multi_system_signal INT DEFAULT 0,
    key_signals_json TEXT NOT NULL,
    z_score_drivers_json TEXT NOT NULL,
    investigation_suggestions_json TEXT,
    specialist_referrals_json TEXT,
    medication_review_notes TEXT,
    procedure_review_notes TEXT,
    medical_references_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id, patient_id) REFERENCES patients(dataset_id, patient_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cds_dataset ON clinical_suggestions(dataset_id);
CREATE INDEX IF NOT EXISTS idx_cds_patient ON clinical_suggestions(dataset_id, patient_id);
CREATE INDEX IF NOT EXISTS idx_cds_run ON clinical_suggestions(analysis_run_id);

CREATE TABLE IF NOT EXISTS clinical_recommendations (
    recommendation_id SERIAL PRIMARY KEY,
    dataset_id VARCHAR(64) NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    patient_id VARCHAR(64) NOT NULL,
    analysis_run_id INT REFERENCES model_runs(run_id) ON DELETE CASCADE,
    clinical_domain VARCHAR(64) NOT NULL,
    abnormal_feature VARCHAR(128) NOT NULL,
    z_score NUMERIC(6,2),
    severity VARCHAR(32) NOT NULL,
    clinical_explanation TEXT NOT NULL,
    medical_suggestion TEXT NOT NULL,
    investigation_suggestion TEXT NOT NULL,
    primary_specialist VARCHAR(128) NOT NULL,
    additional_specialists_json TEXT NOT NULL,
    medication_review TEXT NOT NULL,
    procedure_review TEXT NOT NULL,
    priority VARCHAR(32) NOT NULL,
    evidence_reference TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id, patient_id) REFERENCES patients(dataset_id, patient_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rec_dataset ON clinical_recommendations(dataset_id);
CREATE INDEX IF NOT EXISTS idx_rec_patient ON clinical_recommendations(dataset_id, patient_id);
CREATE INDEX IF NOT EXISTS idx_rec_run ON clinical_recommendations(analysis_run_id);

CREATE TABLE IF NOT EXISTS feature_attributions (
    id SERIAL PRIMARY KEY,
    dataset_id VARCHAR(64) NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    analysis_run_id INT REFERENCES model_runs(run_id) ON DELETE CASCADE,
    patient_id VARCHAR(64) NOT NULL,
    feature_name VARCHAR(128) NOT NULL,
    patient_value NUMERIC(10,4),
    reference_mean NUMERIC(10,4),
    reference_std NUMERIC(10,4),
    z_score NUMERIC(8,4),
    absolute_deviation NUMERIC(10,4),
    is_abnormal INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id, patient_id) REFERENCES patients(dataset_id, patient_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_attr_patient ON feature_attributions(dataset_id, patient_id);
CREATE INDEX IF NOT EXISTS idx_attr_run ON feature_attributions(analysis_run_id);
CREATE INDEX IF NOT EXISTS idx_attr_zscore ON feature_attributions(dataset_id, z_score DESC);
CREATE INDEX IF NOT EXISTS idx_attr_abnormal ON feature_attributions(dataset_id, is_abnormal);

CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id SERIAL PRIMARY KEY,
    dataset_id VARCHAR(64) NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    status VARCHAR(32) DEFAULT 'queued',
    stage VARCHAR(64) DEFAULT 'queued',
    progress INT DEFAULT 0,
    message TEXT,
    k_distance_data_json TEXT,
    results_summary_json TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_aruns_ds ON analysis_runs(dataset_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_aruns_status ON analysis_runs(status);



