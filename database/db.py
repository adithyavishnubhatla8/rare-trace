import os
import uuid
import sqlite3
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config
from database.supabase_client import supabase_manager

class DatabaseManager:
    """
    Unified multi-tenant database access layer supporting source-separated datasets.
    Guarantees 100% strict dataset isolation across patients, models, runs, and results.
    Supports PostgreSQL with automatic SQLite fallback.
    """
    def __init__(self, db_url=Config.DATABASE_URL):
        self.db_url = db_url
        self.engine = None
        self.use_sqlite_fallback = False
        self._init_connection()

    def _init_connection(self):
        """Attempt to establish PostgreSQL connection with pooling, fallback to SQLite if needed."""
        try:
            if self.db_url.startswith("postgresql"):
                self.engine = create_engine(
                    self.db_url,
                    pool_size=5,
                    max_overflow=10,
                    pool_recycle=300,
                    pool_pre_ping=True,
                    connect_args={"connect_timeout": 3}
                )
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                print(f"[SUCCESS] Connected to PostgreSQL Database: {self.db_url}")
            else:
                self.use_sqlite_fallback = True
                print(f"[INFO] Using SQLite Database fallback ({Config.SQLITE_DB_PATH})")
        except Exception as e:
            self.use_sqlite_fallback = True
            print(f"[INFO] PostgreSQL connection failed ({e}). Using SQLite Database fallback ({Config.SQLITE_DB_PATH})")

    def _get_raw_sqlite_connection(self):
        """Return a fresh sqlite3 connection with foreign keys and WAL mode enabled."""
        try:
            Config.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        if Config.IS_SERVERLESS and not Config.SQLITE_DB_PATH.exists():
            candidates = [
                Config.BASE_DIR / "rare_disease.db",
                Path(__file__).resolve().parent.parent / "rare_disease.db",
                Path("/var/task") / "rare_disease.db",
                Path.cwd() / "rare_disease.db",
            ]
            for seed_db in candidates:
                if seed_db.exists():
                    import shutil
                    try:
                        shutil.copy2(seed_db, Config.SQLITE_DB_PATH)
                        print(f"[SUCCESS] Copied seed database from {seed_db} to {Config.SQLITE_DB_PATH}")
                        break
                    except Exception as e:
                        print(f"[WARNING] Could not copy seed database: {e}")

        conn = sqlite3.connect(str(Config.SQLITE_DB_PATH), timeout=30.0)
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            if Config.IS_SERVERLESS:
                conn.execute("PRAGMA journal_mode = DELETE;")
            else:
                conn.execute("PRAGMA journal_mode = WAL;")
        except sqlite3.OperationalError:
            pass
        conn.execute("PRAGMA busy_timeout = 30000;")
        return conn

    def create_tables(self):
        """Create multi-dataset tables in SQLite or PostgreSQL."""
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            cursor = conn.cursor()
            
            # Check if legacy patients table exists without dataset_id
            cursor.execute("PRAGMA table_info(patients);")
            cols = [row[1] for row in cursor.fetchall()]
            if cols and "dataset_id" not in cols:
                print("[INFO] Migrating legacy SQLite schema to source-separated multi-dataset schema...")
                cursor.executescript("""
                    DROP TABLE IF EXISTS feature_attributions;
                    DROP TABLE IF EXISTS analysis_runs;
                    DROP TABLE IF EXISTS clinical_recommendations;
                    DROP TABLE IF EXISTS clinical_suggestions;
                    DROP TABLE IF EXISTS analysis_results;
                    DROP TABLE IF EXISTS model_runs;
                    DROP TABLE IF EXISTS patients;
                """)


            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    analysis_id TEXT UNIQUE NOT NULL,
                    dataset_name TEXT NOT NULL,
                    dataset_filename TEXT NOT NULL,
                    source TEXT DEFAULT 'User Upload',
                    description TEXT,
                    total_records INTEGER NOT NULL,
                    total_clusters INTEGER NOT NULL,
                    anomalies INTEGER NOT NULL,
                    anomaly_percentage REAL NOT NULL,
                    eps REAL,
                    min_samples INTEGER,
                    silhouette_score REAL,
                    status TEXT NOT NULL DEFAULT 'completed',
                    results_summary TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_analyses_id ON analyses(analysis_id);

                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    dataset_name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    description TEXT,
                    filename TEXT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    record_count INTEGER DEFAULT 0,
                    feature_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'Analyzed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS patients (
                    dataset_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (dataset_id, patient_id),
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS model_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    algorithm TEXT DEFAULT 'DBSCAN',
                    eps REAL,
                    min_samples INTEGER,
                    number_of_clusters INTEGER,
                    number_of_anomalies INTEGER,
                    dataset_size INTEGER,
                    silhouette_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS analysis_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    cluster_label INTEGER,
                    is_anomaly INTEGER DEFAULT 0,
                    anomaly_score REAL DEFAULT 0.0,
                    review_priority TEXT DEFAULT 'Normal Profile',
                    key_abnormal_features TEXT,
                    interpretation TEXT,
                    pca_x REAL DEFAULT 0.0,
                    pca_y REAL DEFAULT 0.0,
                    run_id INTEGER,
                    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE,
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
                    suggestion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    analysis_run_id INTEGER,
                    clinical_priority TEXT NOT NULL,
                    primary_domain TEXT NOT NULL,
                    all_domains_json TEXT NOT NULL,
                    multi_system_signal INTEGER DEFAULT 0,
                    key_signals_json TEXT NOT NULL,
                    z_score_drivers_json TEXT NOT NULL,
                    investigation_suggestions_json TEXT,
                    specialist_referrals_json TEXT,
                    medication_review_notes TEXT,
                    procedure_review_notes TEXT,
                    medical_references_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE,
                    FOREIGN KEY (dataset_id, patient_id) REFERENCES patients(dataset_id, patient_id) ON DELETE CASCADE,
                    FOREIGN KEY (analysis_run_id) REFERENCES model_runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_cds_dataset ON clinical_suggestions(dataset_id);
                CREATE INDEX IF NOT EXISTS idx_cds_patient ON clinical_suggestions(dataset_id, patient_id);
                CREATE INDEX IF NOT EXISTS idx_cds_run ON clinical_suggestions(analysis_run_id);

                CREATE TABLE IF NOT EXISTS clinical_recommendations (
                    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    analysis_run_id INTEGER,
                    clinical_domain TEXT NOT NULL,
                    abnormal_feature TEXT NOT NULL,
                    z_score REAL,
                    severity TEXT NOT NULL,
                    clinical_explanation TEXT NOT NULL,
                    medical_suggestion TEXT NOT NULL,
                    investigation_suggestion TEXT NOT NULL,
                    primary_specialist TEXT NOT NULL,
                    additional_specialists_json TEXT NOT NULL,
                    medication_review TEXT NOT NULL,
                    procedure_review TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    evidence_reference TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE,
                    FOREIGN KEY (dataset_id, patient_id) REFERENCES patients(dataset_id, patient_id) ON DELETE CASCADE,
                    FOREIGN KEY (analysis_run_id) REFERENCES model_runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_rec_dataset ON clinical_recommendations(dataset_id);
                CREATE INDEX IF NOT EXISTS idx_rec_patient ON clinical_recommendations(dataset_id, patient_id);
                CREATE INDEX IF NOT EXISTS idx_rec_run ON clinical_recommendations(analysis_run_id);

                CREATE TABLE IF NOT EXISTS feature_attributions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    analysis_run_id INTEGER,
                    patient_id TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    patient_value REAL,
                    reference_mean REAL,
                    reference_std REAL,
                    z_score REAL,
                    absolute_deviation REAL,
                    is_abnormal INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE,
                    FOREIGN KEY (dataset_id, patient_id) REFERENCES patients(dataset_id, patient_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_attr_patient ON feature_attributions(dataset_id, patient_id);
                CREATE INDEX IF NOT EXISTS idx_attr_run ON feature_attributions(analysis_run_id);
                CREATE INDEX IF NOT EXISTS idx_attr_zscore ON feature_attributions(dataset_id, z_score DESC);
                CREATE INDEX IF NOT EXISTS idx_attr_abnormal ON feature_attributions(dataset_id, is_abnormal);

                CREATE TABLE IF NOT EXISTS analysis_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    status TEXT DEFAULT 'queued',
                    stage TEXT DEFAULT 'queued',
                    progress INTEGER DEFAULT 0,
                    message TEXT,
                    k_distance_data_json TEXT,
                    results_summary_json TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_aruns_ds ON analysis_runs(dataset_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_aruns_status ON analysis_runs(status);
            """)

            # Ensure pca_x and pca_y columns exist in analysis_results if table already existed
            cursor.execute("PRAGMA table_info(analysis_results);")
            ar_cols = [r[1] for r in cursor.fetchall()]
            if ar_cols and "pca_x" not in ar_cols:
                cursor.execute("ALTER TABLE analysis_results ADD COLUMN pca_x REAL DEFAULT 0.0;")
            if ar_cols and "pca_y" not in ar_cols:
                cursor.execute("ALTER TABLE analysis_results ADD COLUMN pca_y REAL DEFAULT 0.0;")

            conn.commit()
            conn.close()
        else:
            schema_path = Config.BASE_DIR / "database" / "schema.sql"
            if schema_path.exists():
                with open(schema_path, "r") as f:
                    sql_script = f.read()
                with self.engine.connect() as conn:
                    for statement in sql_script.split(";"):
                        if statement.strip():
                            try:
                                conn.execute(text(statement))
                            except Exception:
                                pass
                    try:
                        conn.execute(text("ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS pca_x NUMERIC(8,4) DEFAULT 0.0;"))
                        conn.execute(text("ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS pca_y NUMERIC(8,4) DEFAULT 0.0;"))
                    except Exception:
                        pass
                    conn.commit()


    # =========================================================================
    # DATASET METADATA OPERATIONS
    # =========================================================================

    def save_dataset_metadata(self, dataset_info):
        """Save or update dataset registration metadata."""
        dataset_id = str(dataset_info["dataset_id"])
        dataset_name = str(dataset_info.get("dataset_name", "Untitled Dataset"))
        source = str(dataset_info.get("source", "User Upload"))
        description = str(dataset_info.get("description", ""))
        filename = str(dataset_info.get("filename", "patients.csv"))
        record_count = int(dataset_info.get("record_count", 0))
        feature_count = int(dataset_info.get("feature_count", 0))
        status = str(dataset_info.get("status", "Analyzed"))

        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO datasets (dataset_id, dataset_name, source, description, filename, record_count, feature_count, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    dataset_name=excluded.dataset_name,
                    source=excluded.source,
                    description=excluded.description,
                    filename=excluded.filename,
                    record_count=excluded.record_count,
                    feature_count=excluded.feature_count,
                    status=excluded.status
            """, (dataset_id, dataset_name, source, description, filename, record_count, feature_count, status))
            conn.commit()
            conn.close()
        else:
            with self.engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO datasets (dataset_id, dataset_name, source, description, filename, record_count, feature_count, status)
                    VALUES (:dataset_id, :dataset_name, :source, :description, :filename, :record_count, :feature_count, :status)
                    ON CONFLICT (dataset_id) DO UPDATE SET
                        dataset_name=EXCLUDED.dataset_name,
                        source=EXCLUDED.source,
                        description=EXCLUDED.description,
                        filename=EXCLUDED.filename,
                        record_count=EXCLUDED.record_count,
                        feature_count=EXCLUDED.feature_count,
                        status=EXCLUDED.status
                """), {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "source": source,
                    "description": description,
                    "filename": filename,
                    "record_count": record_count,
                    "feature_count": feature_count,
                    "status": status
                })

    def get_all_datasets(self):
        """Retrieve all registered datasets sorted by upload date descending."""
        query = "SELECT * FROM datasets ORDER BY upload_date DESC"
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            rows = [dict(r) for r in cursor.fetchall()]
            conn.close()
            return rows
        else:
            with self.engine.connect() as conn:
                res = conn.execute(text(query))
                return [dict(r._mapping) for r in res.fetchall()]

    def get_dataset(self, dataset_id):
        """Retrieve a specific dataset metadata record by dataset_id."""
        query = "SELECT * FROM datasets WHERE dataset_id = ?"
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (str(dataset_id),))
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else None
        else:
            with self.engine.connect() as conn:
                res = conn.execute(text("SELECT * FROM datasets WHERE dataset_id = :dataset_id"), {"dataset_id": str(dataset_id)})
                row = res.fetchone()
                return dict(row._mapping) if row else None

    def delete_dataset(self, dataset_id):
        """Delete a dataset and all associated patients, runs, and results via CASCADE."""
        dataset_id = str(dataset_id)
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM feature_attributions WHERE dataset_id = ?", (dataset_id,))
            cursor.execute("DELETE FROM analysis_runs WHERE dataset_id = ?", (dataset_id,))
            cursor.execute("DELETE FROM clinical_recommendations WHERE dataset_id = ?", (dataset_id,))
            cursor.execute("DELETE FROM clinical_suggestions WHERE dataset_id = ?", (dataset_id,))
            cursor.execute("DELETE FROM analysis_results WHERE dataset_id = ?", (dataset_id,))
            cursor.execute("DELETE FROM model_runs WHERE dataset_id = ?", (dataset_id,))
            cursor.execute("DELETE FROM patients WHERE dataset_id = ?", (dataset_id,))
            cursor.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
            conn.commit()
            conn.close()
        else:
            with self.engine.begin() as conn:
                conn.execute(text("DELETE FROM datasets WHERE dataset_id = :dataset_id"), {"dataset_id": dataset_id})


    # =========================================================================
    # PATIENT RECORDS (DATASET-SCOPED)
    # =========================================================================

    def save_patients(self, df_patients, dataset_id="ds_default"):
        """Save patient records strictly partitioned by dataset_id, serializing raw features as JSON."""
        dataset_id = str(dataset_id)
        df_save = df_patients.copy()

        # Guarantee dataset record exists to satisfy foreign key constraints
        if not self.get_dataset(dataset_id):
            self.save_dataset_metadata({
                "dataset_id": dataset_id,
                "dataset_name": "Synthetic Rare-Disease Cohort" if dataset_id == "ds_default" else f"Dataset {dataset_id}",
                "source": "Academic Benchmark" if dataset_id == "ds_default" else "User Upload",
                "description": "Clinical patient cohort records",
                "filename": "patients.csv",
                "record_count": len(df_save),
                "feature_count": len(df_save.columns),
                "status": "Analyzed"
            })

        # Guarantee patient_id column exists
        if "patient_id" not in df_save.columns:
            found_id = None
            for col in df_save.columns:
                c_norm = col.lower().replace("_", "").replace(" ", "")
                if c_norm in ["patientid", "patient", "id", "recordid", "subjectid", "pid", "subjectcode", "caseid", "subject"]:
                    found_id = col
                    break
            if found_id:
                df_save["patient_id"] = df_save[found_id].astype(str)
            else:
                df_save["patient_id"] = [f"PAT-{10001 + i}" for i in range(len(df_save))]
        else:
            df_save["patient_id"] = df_save["patient_id"].astype(str)

        rows = []
        for _, row in df_save.iterrows():
            p_id = str(row["patient_id"])
            row_dict = row.to_dict()
            # Clean non-serializable elements
            clean_dict = {
                k: (None if pd.isna(v) else v) for k, v in row_dict.items()
                if k not in ["cluster_label", "is_anomaly", "anomaly_score", "review_priority", "key_abnormal_features", "interpretation", "run_id"]
            }
            rows.append((dataset_id, p_id, json.dumps(clean_dict)))

        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM patients WHERE dataset_id = ?", (dataset_id,))
            cursor.executemany("INSERT INTO patients (dataset_id, patient_id, data_json) VALUES (?, ?, ?)", rows)
            conn.commit()
            conn.close()
        else:
            with self.engine.begin() as conn:
                conn.execute(text("DELETE FROM patients WHERE dataset_id = :dataset_id"), {"dataset_id": dataset_id})
                for r in rows:
                    conn.execute(text("INSERT INTO patients (dataset_id, patient_id, data_json) VALUES (:ds, :pid, :data)"),
                                 {"ds": r[0], "pid": r[1], "data": r[2]})

        print(f"[SUCCESS] Saved {len(rows)} patient records for dataset '{dataset_id}' to DB.")

    # =========================================================================
    # MODEL RUNS (DATASET-SCOPED)
    # =========================================================================

    def save_model_run(self, run_metrics, dataset_id="ds_default"):
        """Save a new model run execution record for a specific dataset and return run_id."""
        dataset_id = str(dataset_id)
        algo = str(run_metrics.get("algorithm", "DBSCAN"))
        eps = float(run_metrics.get("eps", Config.DEFAULT_EPS))
        min_samples = int(run_metrics.get("min_samples", Config.DEFAULT_MIN_SAMPLES))
        n_clusters = int(run_metrics.get("number_of_clusters", 0))
        n_anomalies = int(run_metrics.get("number_of_anomalies", 0))
        d_size = int(run_metrics.get("dataset_size", 0))
        sil_score = float(run_metrics.get("silhouette_score", 0.0))

        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_runs (dataset_id, algorithm, eps, min_samples, number_of_clusters, number_of_anomalies, dataset_size, silhouette_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (dataset_id, algo, eps, min_samples, n_clusters, n_anomalies, d_size, sil_score))
            run_id = cursor.lastrowid
            conn.commit()
            conn.close()
        else:
            with self.engine.begin() as conn:
                res = conn.execute(text("""
                    INSERT INTO model_runs (dataset_id, algorithm, eps, min_samples, number_of_clusters, number_of_anomalies, dataset_size, silhouette_score)
                    VALUES (:ds, :algo, :eps, :ms, :nc, :na, :dsize, :sil)
                    RETURNING run_id
                """), {
                    "ds": dataset_id, "algo": algo, "eps": eps, "ms": min_samples,
                    "nc": n_clusters, "na": n_anomalies, "dsize": d_size, "sil": sil_score
                })
                run_id = res.scalar()

        return run_id or 1

    def get_latest_model_run(self, dataset_id="ds_default"):
        """Fetch the most recent model execution run record for a specific dataset."""
        dataset_id = str(dataset_id)
        query = "SELECT * FROM model_runs WHERE dataset_id = ? ORDER BY run_id DESC LIMIT 1"
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (dataset_id,))
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else {}
        else:
            with self.engine.connect() as conn:
                res = conn.execute(text("SELECT * FROM model_runs WHERE dataset_id = :ds ORDER BY run_id DESC LIMIT 1"), {"ds": dataset_id})
                row = res.fetchone()
                return dict(row._mapping) if row else {}

    # =========================================================================
    # ANALYSIS RESULTS (DATASET-SCOPED)
    # =========================================================================

    def save_analysis_results(self, df_results, dataset_id="ds_default", run_id=1):
        """Save analysis results strictly partitioned by dataset_id."""
        dataset_id = str(dataset_id)
        df_save = df_results.copy()

        if "patient_id" not in df_save.columns:
            df_save["patient_id"] = [f"PAT-{10001 + i}" for i in range(len(df_save))]
        else:
            df_save["patient_id"] = df_save["patient_id"].astype(str)

        df_save["dataset_id"] = dataset_id
        if "pca_x" not in df_save.columns:
            df_save["pca_x"] = 0.0
        if "pca_y" not in df_save.columns:
            df_save["pca_y"] = 0.0

        rows = []
        for _, r in df_save.iterrows():
            rows.append((
                dataset_id,
                str(r["patient_id"]),
                int(r["cluster_label"]) if pd.notna(r.get("cluster_label")) else 0,
                int(r.get("is_anomaly", 0)),
                float(r.get("anomaly_score", 0.0)),
                str(r.get("review_priority", "Normal Profile")),
                str(r.get("key_abnormal_features", "")),
                str(r.get("interpretation", "")),
                float(r.get("pca_x", 0.0)),
                float(r.get("pca_y", 0.0)),
                run_id
            ))

        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM analysis_results WHERE dataset_id = ?", (dataset_id,))
            cursor.executemany("""
                INSERT INTO analysis_results (
                    dataset_id, patient_id, cluster_label, is_anomaly, anomaly_score,
                    review_priority, key_abnormal_features, interpretation, pca_x, pca_y, run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            conn.close()
        else:
            with self.engine.begin() as conn:
                conn.execute(text("DELETE FROM analysis_results WHERE dataset_id = :ds"), {"ds": dataset_id})
                for r in rows:
                    conn.execute(text("""
                        INSERT INTO analysis_results (
                            dataset_id, patient_id, cluster_label, is_anomaly, anomaly_score,
                            review_priority, key_abnormal_features, interpretation, pca_x, pca_y, run_id
                        ) VALUES (
                            :ds, :pid, :cl, :ia, :sc, :prio, :feat, :interp, :px, :py, :rid
                        )
                    """), {
                        "ds": r[0], "pid": r[1], "cl": r[2], "ia": r[3], "sc": r[4],
                        "prio": r[5], "feat": r[6], "interp": r[7], "px": r[8], "py": r[9], "rid": r[10]
                    })

        print(f"[SUCCESS] Saved {len(rows)} analysis results for dataset '{dataset_id}' to DB.")


    def get_full_patient_analysis(self, dataset_id="ds_default"):
        """
        Join patients table with analysis_results strictly for dataset_id.
        Deserializes original dynamic patient features and combines with ML analysis output.
        """
        dataset_id = str(dataset_id)
        query = """
            SELECT p.patient_id, p.data_json, 
                   a.cluster_label, a.is_anomaly, a.anomaly_score, 
                   a.review_priority, a.key_abnormal_features, a.interpretation, a.run_id
            FROM patients p
            LEFT JOIN analysis_results a 
                ON p.dataset_id = a.dataset_id AND p.patient_id = a.patient_id
            WHERE p.dataset_id = ?
            ORDER BY a.anomaly_score DESC
        """
        records = []
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (dataset_id,))
            rows = cursor.fetchall()
            for r in rows:
                p_dict = json.loads(r["data_json"]) if r["data_json"] else {}
                p_dict["patient_id"] = r["patient_id"]
                p_dict["cluster_label"] = r["cluster_label"]
                p_dict["is_anomaly"] = r["is_anomaly"]
                p_dict["anomaly_score"] = r["anomaly_score"]
                p_dict["review_priority"] = r["review_priority"]
                p_dict["key_abnormal_features"] = r["key_abnormal_features"]
                p_dict["interpretation"] = r["interpretation"]
                p_dict["run_id"] = r["run_id"]
                records.append(p_dict)
            conn.close()
        else:
            with self.engine.connect() as conn:
                res = conn.execute(text("""
                    SELECT p.patient_id, p.data_json, 
                           a.cluster_label, a.is_anomaly, a.anomaly_score, 
                           a.review_priority, a.key_abnormal_features, a.interpretation, a.run_id
                    FROM patients p
                    LEFT JOIN analysis_results a 
                        ON p.dataset_id = a.dataset_id AND p.patient_id = a.patient_id
                    WHERE p.dataset_id = :ds
                    ORDER BY a.anomaly_score DESC
                """), {"ds": dataset_id})
                for r in res.fetchall():
                    mapping = r._mapping
                    p_dict = json.loads(mapping["data_json"]) if mapping["data_json"] else {}
                    p_dict["patient_id"] = mapping["patient_id"]
                    p_dict["cluster_label"] = mapping["cluster_label"]
                    p_dict["is_anomaly"] = mapping["is_anomaly"]
                    p_dict["anomaly_score"] = mapping["anomaly_score"]
                    p_dict["review_priority"] = mapping["review_priority"]
                    p_dict["key_abnormal_features"] = mapping["key_abnormal_features"]
                    p_dict["interpretation"] = mapping["interpretation"]
                    p_dict["run_id"] = mapping["run_id"]
                    records.append(p_dict)

        return pd.DataFrame(records) if records else pd.DataFrame()

    def get_patient_by_id(self, dataset_id, patient_id=None):
        """Fetch individual patient profile and ML analysis for a specific dataset."""
        if patient_id is None:
            patient_id = str(dataset_id)
            dataset_id = "ds_default"
        else:
            dataset_id = str(dataset_id)
            patient_id = str(patient_id)
        query = """
            SELECT p.patient_id, p.data_json, 
                   a.cluster_label, a.is_anomaly, a.anomaly_score, 
                   a.review_priority, a.key_abnormal_features, a.interpretation, a.run_id
            FROM patients p
            LEFT JOIN analysis_results a 
                ON p.dataset_id = a.dataset_id AND p.patient_id = a.patient_id
            WHERE p.dataset_id = ? AND p.patient_id = ?
        """
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (dataset_id, patient_id))
            r = cursor.fetchone()
            conn.close()
            if not r:
                return None
            p_dict = json.loads(r["data_json"]) if r["data_json"] else {}
            p_dict["patient_id"] = r["patient_id"]
            p_dict["cluster_label"] = r["cluster_label"]
            p_dict["is_anomaly"] = r["is_anomaly"]
            p_dict["anomaly_score"] = r["anomaly_score"]
            p_dict["review_priority"] = r["review_priority"]
            p_dict["key_abnormal_features"] = r["key_abnormal_features"]
            p_dict["interpretation"] = r["interpretation"]
            p_dict["run_id"] = r["run_id"]
            return p_dict
        else:
            with self.engine.connect() as conn:
                res = conn.execute(text("""
                    SELECT p.patient_id, p.data_json, 
                           a.cluster_label, a.is_anomaly, a.anomaly_score, 
                           a.review_priority, a.key_abnormal_features, a.interpretation, a.run_id
                    FROM patients p
                    LEFT JOIN analysis_results a 
                        ON p.dataset_id = a.dataset_id AND p.patient_id = a.patient_id
                    WHERE p.dataset_id = :ds AND p.patient_id = :pid
                """), {"ds": dataset_id, "pid": patient_id})
                r = res.fetchone()
                if not r:
                    return None
                mapping = r._mapping
                p_dict = json.loads(mapping["data_json"]) if mapping["data_json"] else {}
                p_dict["patient_id"] = mapping["patient_id"]
                p_dict["cluster_label"] = mapping["cluster_label"]
                p_dict["is_anomaly"] = mapping["is_anomaly"]
                p_dict["anomaly_score"] = mapping["anomaly_score"]
                p_dict["review_priority"] = mapping["review_priority"]
                p_dict["key_abnormal_features"] = mapping["key_abnormal_features"]
                p_dict["interpretation"] = mapping["interpretation"]
                p_dict["run_id"] = mapping["run_id"]
                return p_dict

    # =========================================================================
    # CLINICAL DECISION SUPPORT & MEDICAL RECOMMENDATIONS (DATASET-SCOPED)
    # =========================================================================

    def save_clinical_suggestions(self, suggestions, dataset_id="ds_default", run_id=None):
        """
        Persist structured clinical decision support suggestions partitioned by dataset_id and run_id.
        Replaces any prior suggestions for this dataset_id (or analysis_run_id).
        """
        dataset_id = str(dataset_id)
        if not suggestions:
            return

        # Validate candidate run_ids in a single query to prevent per-row SQLite lock contention
        candidate_rids = set()
        if run_id is not None:
            try:
                candidate_rids.add(int(run_id))
            except Exception:
                pass
        for s in suggestions:
            c_rid = s.get("analysis_run_id")
            if c_rid is not None:
                try:
                    candidate_rids.add(int(c_rid))
                except Exception:
                    pass

        valid_rids = set()
        if candidate_rids:
            try:
                if self.use_sqlite_fallback:
                    c_chk = self._get_raw_sqlite_connection()
                    cur_chk = c_chk.cursor()
                    placeholders = ",".join("?" for _ in candidate_rids)
                    cur_chk.execute(f"SELECT run_id FROM model_runs WHERE run_id IN ({placeholders})", list(candidate_rids))
                    valid_rids = {row[0] for row in cur_chk.fetchall()}
                    c_chk.close()
                else:
                    with self.engine.connect() as c_chk:
                        res = c_chk.execute(text(f"SELECT run_id FROM model_runs WHERE run_id IN :rids"), {"rids": tuple(candidate_rids)}).fetchall()
                        valid_rids = {row[0] for row in res}
            except Exception:
                valid_rids = set()

        default_run_id = int(run_id) if run_id is not None and int(run_id) in valid_rids else None

        rows = []
        for s in suggestions:
            s_run_id = default_run_id
            if s_run_id is None and s.get("analysis_run_id") is not None:
                try:
                    cand = int(s.get("analysis_run_id"))
                    if cand in valid_rids:
                        s_run_id = cand
                except Exception:
                    s_run_id = None

            rows.append((
                dataset_id,
                str(s.get("patient_id")),
                s_run_id,
                str(s.get("clinical_priority", "LOW")),
                str(s.get("primary_domain", "GENERAL")),
                json.dumps(s.get("all_domains", [])),
                int(s.get("multi_system_signal", 0)),
                json.dumps(s.get("key_signals", [])),
                json.dumps(s.get("z_score_drivers", [])),
                json.dumps(s.get("investigation_suggestions", [])),
                json.dumps(s.get("specialist_referrals", [])),
                str(s.get("medication_review_notes", "")),
                str(s.get("procedure_review_notes", "")),
                json.dumps(s.get("medical_references", []))
            ))

        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            cursor = conn.cursor()
            if run_id is not None:
                cursor.execute("DELETE FROM clinical_suggestions WHERE dataset_id = ? AND analysis_run_id = ?", (dataset_id, run_id))
            else:
                cursor.execute("DELETE FROM clinical_suggestions WHERE dataset_id = ?", (dataset_id,))
            cursor.executemany("""
                INSERT INTO clinical_suggestions (
                    dataset_id, patient_id, analysis_run_id, clinical_priority,
                    primary_domain, all_domains_json, multi_system_signal,
                    key_signals_json, z_score_drivers_json, investigation_suggestions_json,
                    specialist_referrals_json, medication_review_notes, procedure_review_notes,
                    medical_references_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            conn.close()
        else:
            with self.engine.begin() as conn:
                if run_id is not None:
                    conn.execute(text("DELETE FROM clinical_suggestions WHERE dataset_id = :ds AND analysis_run_id = :rid"),
                                 {"ds": dataset_id, "rid": run_id})
                else:
                    conn.execute(text("DELETE FROM clinical_suggestions WHERE dataset_id = :ds"), {"ds": dataset_id})
                for r in rows:
                    conn.execute(text("""
                        INSERT INTO clinical_suggestions (
                            dataset_id, patient_id, analysis_run_id, clinical_priority,
                            primary_domain, all_domains_json, multi_system_signal,
                            key_signals_json, z_score_drivers_json, investigation_suggestions_json,
                            specialist_referrals_json, medication_review_notes, procedure_review_notes,
                            medical_references_json
                        ) VALUES (
                            :ds, :pid, :rid, :prio, :pdom, :adom, :ms, :sig, :zd, :inv, :spec, :med, :proc, :ref
                        )
                    """), {
                        "ds": r[0], "pid": r[1], "rid": r[2], "prio": r[3],
                        "pdom": r[4], "adom": r[5], "ms": r[6], "sig": r[7],
                        "zd": r[8], "inv": r[9], "spec": r[10], "med": r[11],
                        "proc": r[12], "ref": r[13]
                    })

        print(f"[SUCCESS] Saved {len(rows)} clinical suggestions for dataset '{dataset_id}' to DB.")

    def _unpack_suggestion_row(self, r):
        """Helper to unpack JSON fields in a clinical suggestion record."""
        row_dict = dict(r)
        for json_field in ["all_domains_json", "key_signals_json", "z_score_drivers_json",
                           "investigation_suggestions_json", "specialist_referrals_json",
                           "medical_references_json"]:
            clean_key = json_field.replace("_json", "")
            try:
                row_dict[clean_key] = json.loads(row_dict[json_field]) if row_dict.get(json_field) else []
            except Exception:
                row_dict[clean_key] = []
        return row_dict

    def get_clinical_suggestions(self, dataset_id="ds_default", priority=None, domain=None, patient_id=None, run_id=None):
        """
        Query clinical suggestions strictly scoped to dataset_id with optional filters.
        Orders by priority (HIGH, MEDIUM, LOW) and patient_id.
        """
        dataset_id = str(dataset_id)
        query = "SELECT * FROM clinical_suggestions WHERE dataset_id = ?"
        params = [dataset_id]

        if run_id is not None:
            query += " AND analysis_run_id = ?"
            params.append(int(run_id))
        if priority and priority.upper() != "ALL":
            query += " AND UPPER(clinical_priority) = ?"
            params.append(priority.upper())
        if domain and domain.upper() != "ALL":
            query += " AND (UPPER(primary_domain) = ? OR all_domains_json LIKE ?)"
            params.append(domain.upper())
            params.append(f"%{domain.upper()}%")
        if patient_id:
            query += " AND patient_id LIKE ?"
            params.append(f"%{patient_id}%")

        query += """
            ORDER BY 
                CASE clinical_priority 
                    WHEN 'HIGH' THEN 1 
                    WHEN 'MEDIUM' THEN 2 
                    WHEN 'LOW' THEN 3 
                    ELSE 4 
                END, 
                multi_system_signal DESC,
                patient_id ASC
        """

        results = []
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            for r in cursor.fetchall():
                results.append(self._unpack_suggestion_row(r))
            conn.close()
        else:
            with self.engine.connect() as conn:
                # Convert SQLite ? parameters to text
                pg_query = query.replace("?", ":p")
                param_dict = {f"p{i}": v for i, v in enumerate(params)}
                res = conn.execute(text(pg_query), param_dict)
                for r in res.fetchall():
                    results.append(self._unpack_suggestion_row(r._mapping))

        return results

    def get_patient_clinical_suggestion(self, dataset_id, patient_id):
        """Retrieve the latest clinical suggestion for a specific patient within a dataset."""
        dataset_id = str(dataset_id)
        patient_id = str(patient_id)
        query = """
            SELECT * FROM clinical_suggestions 
            WHERE dataset_id = ? AND patient_id = ? 
            ORDER BY suggestion_id DESC LIMIT 1
        """
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (dataset_id, patient_id))
            row = cursor.fetchone()
            conn.close()
            return self._unpack_suggestion_row(row) if row else None
        else:
            with self.engine.connect() as conn:
                res = conn.execute(text("""
                    SELECT * FROM clinical_suggestions 
                    WHERE dataset_id = :ds AND patient_id = :pid 
                    ORDER BY suggestion_id DESC LIMIT 1
                """), {"ds": dataset_id, "pid": patient_id})
                row = res.fetchone()
                return self._unpack_suggestion_row(row._mapping) if row else None

    def get_clinical_suggestions_summary(self, dataset_id="ds_default"):
        """Get high-level summary metrics of clinical suggestions for the active dataset."""
        dataset_id = str(dataset_id)
        suggestions = self.get_clinical_suggestions(dataset_id=dataset_id)
        
        total = len(suggestions)
        high = sum(1 for s in suggestions if s.get("clinical_priority") == "HIGH")
        medium = sum(1 for s in suggestions if s.get("clinical_priority") == "MEDIUM")
        low = sum(1 for s in suggestions if s.get("clinical_priority") == "LOW")
        multi_system = sum(1 for s in suggestions if s.get("multi_system_signal") == 1)

        domain_counts = {}
        for s in suggestions:
            dom = s.get("primary_domain", "GENERAL")
            domain_counts[dom] = domain_counts.get(dom, 0) + 1

        return {
            "total_reviewed": total,
            "high_priority_count": high,
            "medium_priority_count": medium,
            "low_priority_count": low,
            "multi_system_count": multi_system,
            "domain_breakdown": domain_counts
        }

    # =========================================================================
    # DETAILED MEDICAL SUGGESTIONS & SPECIALIST RECOMMENDATIONS (DATASET-SCOPED)
    # =========================================================================

    def save_clinical_recommendations(self, recommendations, dataset_id="ds_default", run_id=None):
        """
        Persist detailed per-feature clinical recommendations partitioned strictly by dataset_id and run_id.
        """
        dataset_id = str(dataset_id)
        if not recommendations:
            return

        # Validate candidate run_ids in a single query to prevent per-row SQLite lock contention
        candidate_rids = set()
        if run_id is not None:
            try:
                candidate_rids.add(int(run_id))
            except Exception:
                pass
        for r in recommendations:
            c_rid = r.get("analysis_run_id")
            if c_rid is not None:
                try:
                    candidate_rids.add(int(c_rid))
                except Exception:
                    pass

        valid_rids = set()
        if candidate_rids:
            try:
                if self.use_sqlite_fallback:
                    c_chk = self._get_raw_sqlite_connection()
                    cur_chk = c_chk.cursor()
                    placeholders = ",".join("?" for _ in candidate_rids)
                    cur_chk.execute(f"SELECT run_id FROM model_runs WHERE run_id IN ({placeholders})", list(candidate_rids))
                    valid_rids = {row[0] for row in cur_chk.fetchall()}
                    c_chk.close()
                else:
                    with self.engine.connect() as c_chk:
                        res = c_chk.execute(text(f"SELECT run_id FROM model_runs WHERE run_id IN :rids"), {"rids": tuple(candidate_rids)}).fetchall()
                        valid_rids = {row[0] for row in res}
            except Exception:
                valid_rids = set()

        default_run_id = int(run_id) if run_id is not None and int(run_id) in valid_rids else None

        rows = []
        for r in recommendations:
            r_run_id = default_run_id
            if r_run_id is None and r.get("analysis_run_id") is not None:
                try:
                    cand = int(r.get("analysis_run_id"))
                    if cand in valid_rids:
                        r_run_id = cand
                except Exception:
                    r_run_id = None
            rows.append((
                dataset_id,
                str(r.get("patient_id")),
                r_run_id,
                str(r.get("clinical_domain", "GENERAL")),
                str(r.get("abnormal_feature", "")),
                float(r.get("z_score", 0.0)) if r.get("z_score") is not None else None,
                str(r.get("severity", "Unusual")),
                str(r.get("clinical_explanation", "")),
                str(r.get("medical_suggestion", "")),
                str(r.get("investigation_suggestion", "")),
                str(r.get("primary_specialist", "General Physician / Internal Medicine Specialist")),
                json.dumps(r.get("additional_specialists", [])),
                str(r.get("medication_review", "")),
                str(r.get("procedure_review", "")),
                str(r.get("priority", "MEDIUM")),
                str(r.get("evidence_reference", ""))
            ))

        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            cursor = conn.cursor()
            if run_id is not None:
                cursor.execute("DELETE FROM clinical_recommendations WHERE dataset_id = ? AND analysis_run_id = ?", (dataset_id, run_id))
            else:
                cursor.execute("DELETE FROM clinical_recommendations WHERE dataset_id = ?", (dataset_id,))
            cursor.executemany("""
                INSERT INTO clinical_recommendations (
                    dataset_id, patient_id, analysis_run_id, clinical_domain,
                    abnormal_feature, z_score, severity, clinical_explanation,
                    medical_suggestion, investigation_suggestion, primary_specialist,
                    additional_specialists_json, medication_review, procedure_review,
                    priority, evidence_reference
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            conn.close()
        else:
            with self.engine.begin() as conn:
                if run_id is not None:
                    conn.execute(text("DELETE FROM clinical_recommendations WHERE dataset_id = :ds AND analysis_run_id = :rid"),
                                 {"ds": dataset_id, "rid": run_id})
                else:
                    conn.execute(text("DELETE FROM clinical_recommendations WHERE dataset_id = :ds"), {"ds": dataset_id})
                for row in rows:
                    conn.execute(text("""
                        INSERT INTO clinical_recommendations (
                            dataset_id, patient_id, analysis_run_id, clinical_domain,
                            abnormal_feature, z_score, severity, clinical_explanation,
                            medical_suggestion, investigation_suggestion, primary_specialist,
                            additional_specialists_json, medication_review, procedure_review,
                            priority, evidence_reference
                        ) VALUES (
                            :ds, :pid, :rid, :cdom, :feat, :z, :sev, :exp,
                            :msug, :isug, :pspec, :aspec, :med, :proc, :prio, :ref
                        )
                    """), {
                        "ds": row[0], "pid": row[1], "rid": row[2], "cdom": row[3],
                        "feat": row[4], "z": row[5], "sev": row[6], "exp": row[7],
                        "msug": row[8], "isug": row[9], "pspec": row[10], "aspec": row[11],
                        "med": row[12], "proc": row[13], "prio": row[14], "ref": row[15]
                    })

        print(f"[SUCCESS] Saved {len(rows)} clinical recommendations for dataset '{dataset_id}' to DB.")

    def get_clinical_recommendations(self, dataset_id="ds_default", patient_id=None):
        """
        Query clinical recommendations strictly partitioned by dataset_id.
        Optionally filter by patient_id.
        """
        dataset_id = str(dataset_id)
        query = "SELECT * FROM clinical_recommendations WHERE dataset_id = ?"
        params = [dataset_id]
        if patient_id:
            query += " AND patient_id = ?"
            params.append(str(patient_id))

        query += " ORDER BY recommendation_id ASC"

        results = []
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            for r in cursor.fetchall():
                d = dict(r)
                try:
                    d["additional_specialists"] = json.loads(d["additional_specialists_json"]) if d.get("additional_specialists_json") else []
                except Exception:
                    d["additional_specialists"] = []
                results.append(d)
            conn.close()
        else:
            with self.engine.connect() as conn:
                pg_query = query.replace("?", ":p")
                pdict = {f"p{i}": v for i, v in enumerate(params)}
                res = conn.execute(text(pg_query), pdict)
                for r in res.fetchall():
                    d = dict(r._mapping)
                    try:
                        d["additional_specialists"] = json.loads(d["additional_specialists_json"]) if d.get("additional_specialists_json") else []
                    except Exception:
                        d["additional_specialists"] = []
                    results.append(d)
        return results

    # =========================================================================
    # HIGH-PERFORMANCE SQL AGGREGATIONS & SERVER-SIDE PAGINATION
    # =========================================================================

    def get_dataset_summary_stats(self, dataset_id="ds_default"):
        """
        Fast SQL aggregation computing total patients, candidate cases, priority distribution,
        and anomaly score histogram bins directly inside the DB engine in a single query.
        """
        dataset_id = str(dataset_id)
        query = """
            SELECT 
                COUNT(*) as total_patients,
                SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) as candidate_cases,
                SUM(CASE WHEN review_priority = 'High Priority' THEN 1 ELSE 0 END) as high_priority,
                SUM(CASE WHEN review_priority = 'Medium Priority' THEN 1 ELSE 0 END) as medium_priority,
                SUM(CASE WHEN review_priority = 'Low Priority' THEN 1 ELSE 0 END) as low_priority,
                SUM(CASE WHEN review_priority = 'Normal Profile' OR review_priority = 'Normative Profile' THEN 1 ELSE 0 END) as normal_priority,
                SUM(CASE WHEN anomaly_score < 20 THEN 1 ELSE 0 END) as score_0_20,
                SUM(CASE WHEN anomaly_score >= 20 AND anomaly_score < 40 THEN 1 ELSE 0 END) as score_20_40,
                SUM(CASE WHEN anomaly_score >= 40 AND anomaly_score < 60 THEN 1 ELSE 0 END) as score_40_60,
                SUM(CASE WHEN anomaly_score >= 60 AND anomaly_score < 80 THEN 1 ELSE 0 END) as score_60_80,
                SUM(CASE WHEN anomaly_score >= 80 THEN 1 ELSE 0 END) as score_80_100
            FROM analysis_results
            WHERE dataset_id = ?
        """
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (dataset_id,))
            row = cursor.fetchone()
            conn.close()
            r = dict(row) if row else {}
        else:
            with self.engine.connect() as conn:
                res = conn.execute(text(query.replace("?", ":ds")), {"ds": dataset_id})
                row = res.fetchone()
                r = dict(row._mapping) if row else {}

        total = int(r.get("total_patients") or 0)
        candidates = int(r.get("candidate_cases") or 0)
        pct = round((candidates / max(1, total)) * 100.0, 1)

        priority_counts = {
            "High Priority": int(r.get("high_priority") or 0),
            "Medium Priority": int(r.get("medium_priority") or 0),
            "Low Priority": int(r.get("low_priority") or 0),
            "Normative Profile": int(r.get("normal_priority") or 0)
        }
        score_bins = {
            "0-20": int(r.get("score_0_20") or 0),
            "20-40": int(r.get("score_20_40") or 0),
            "40-60": int(r.get("score_40_60") or 0),
            "60-80": int(r.get("score_60_80") or 0),
            "80-100": int(r.get("score_80_100") or 0)
        }

        return {
            "total_patients": total,
            "candidate_cases": candidates,
            "anomaly_pct": pct,
            "priority_counts": priority_counts,
            "score_bins": score_bins
        }

    def get_cluster_distribution(self, dataset_id="ds_default"):
        """Fast SQL cluster group count aggregation."""
        dataset_id = str(dataset_id)
        query = """
            SELECT cluster_label, COUNT(*) as cnt
            FROM analysis_results
            WHERE dataset_id = ?
            GROUP BY cluster_label
            ORDER BY cluster_label ASC
        """
        rows = []
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (dataset_id,))
            rows = [dict(r) for r in cursor.fetchall()]
            conn.close()
        else:
            with self.engine.connect() as conn:
                res = conn.execute(text(query.replace("?", ":ds")), {"ds": dataset_id})
                rows = [dict(r._mapping) for r in res.fetchall()]

        cluster_data = {}
        for r in rows:
            lbl = r.get("cluster_label")
            cnt = int(r.get("cnt") or 0)
            if lbl is None:
                continue
            try:
                lbl_int = int(lbl)
                key_str = "Candidate Rare Cases (-1)" if lbl_int == -1 else f"Cluster {lbl_int}"
            except (ValueError, TypeError):
                key_str = str(lbl)
            cluster_data[key_str] = cnt
        return cluster_data

    def get_stored_pca_scatter(self, dataset_id="ds_default", limit=400):
        """Retrieve pre-computed 2D PCA projection coordinates without running sklearn on GET."""
        dataset_id = str(dataset_id)
        query = """
            SELECT patient_id, pca_x, pca_y, cluster_label, anomaly_score, is_anomaly
            FROM analysis_results
            WHERE dataset_id = ?
            ORDER BY is_anomaly DESC, anomaly_score DESC
            LIMIT ?
        """
        rows = []
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (dataset_id, limit))
            rows = [dict(r) for r in cursor.fetchall()]
            conn.close()
        else:
            with self.engine.connect() as conn:
                res = conn.execute(text("""
                    SELECT patient_id, pca_x, pca_y, cluster_label, anomaly_score, is_anomaly
                    FROM analysis_results
                    WHERE dataset_id = :ds
                    ORDER BY is_anomaly DESC, anomaly_score DESC
                    LIMIT :lim
                """), {"ds": dataset_id, "lim": limit})
                rows = [dict(r._mapping) for r in res.fetchall()]

        points = []
        for r in rows:
            points.append({
                "x": round(float(r.get("pca_x") or 0.0), 2),
                "y": round(float(r.get("pca_y") or 0.0), 2),
                "cluster": int(r.get("cluster_label") if r.get("cluster_label") is not None else 0),
                "patient_id": str(r.get("patient_id", "")),
                "score": round(float(r.get("anomaly_score") or 0.0), 1),
                "is_anomaly": int(r.get("is_anomaly") or 0)
            })
        return points

    def get_top_anomalies(self, dataset_id="ds_default", limit=5):
        """Fetch top highest scoring candidate anomalies with patient demographic data in < 2ms."""
        dataset_id = str(dataset_id)
        query = """
            SELECT p.patient_id, p.data_json, a.cluster_label, a.is_anomaly, a.anomaly_score,
                   a.review_priority, a.key_abnormal_features, a.interpretation, a.run_id
            FROM analysis_results a
            JOIN patients p ON a.dataset_id = p.dataset_id AND a.patient_id = p.patient_id
            WHERE a.dataset_id = ? AND a.is_anomaly = 1
            ORDER BY a.anomaly_score DESC
            LIMIT ?
        """
        rows = []
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (dataset_id, limit))
            rows = [dict(r) for r in cursor.fetchall()]
            conn.close()
        else:
            with self.engine.connect() as conn:
                res = conn.execute(text("""
                    SELECT p.patient_id, p.data_json, a.cluster_label, a.is_anomaly, a.anomaly_score,
                           a.review_priority, a.key_abnormal_features, a.interpretation, a.run_id
                    FROM analysis_results a
                    JOIN patients p ON a.dataset_id = p.dataset_id AND a.patient_id = p.patient_id
                    WHERE a.dataset_id = :ds AND a.is_anomaly = 1
                    ORDER BY a.anomaly_score DESC
                    LIMIT :lim
                """), {"ds": dataset_id, "lim": limit})
                rows = [dict(r._mapping) for r in res.fetchall()]

        records = []
        for r in rows:
            p_dict = json.loads(r["data_json"]) if r.get("data_json") else {}
            p_dict["patient_id"] = r["patient_id"]
            p_dict["cluster_label"] = r["cluster_label"]
            p_dict["is_anomaly"] = r["is_anomaly"]
            p_dict["anomaly_score"] = r["anomaly_score"]
            p_dict["review_priority"] = r["review_priority"]
            p_dict["key_abnormal_features"] = r["key_abnormal_features"]
            p_dict["interpretation"] = r["interpretation"]
            p_dict["run_id"] = r["run_id"]
            records.append(p_dict)
        return records

    def get_paginated_patients(self, dataset_id="ds_default", page=1, per_page=25, search=None, filter_anomaly="all", filter_cluster="all"):
        """Server-side paginated patient query using SQL LIMIT / OFFSET."""
        import math
        dataset_id = str(dataset_id)
        page = max(1, int(page))
        per_page = max(1, min(100, int(per_page)))
        offset = (page - 1) * per_page

        where_clauses = ["p.dataset_id = ?"]
        params = [dataset_id]

        if str(filter_anomaly) == "1":
            where_clauses.append("a.is_anomaly = 1")
        elif str(filter_anomaly) == "0":
            where_clauses.append("a.is_anomaly = 0")

        if filter_cluster != "all" and filter_cluster is not None:
            try:
                c_int = int(filter_cluster)
                where_clauses.append("a.cluster_label = ?")
                params.append(c_int)
            except ValueError:
                pass

        if search and search.strip():
            s_term = f"%{search.strip().lower()}%"
            where_clauses.append("(LOWER(p.patient_id) LIKE ? OR LOWER(p.data_json) LIKE ?)")
            params.extend([s_term, s_term])

        where_sql = " AND ".join(where_clauses)

        count_sql = f"""
            SELECT COUNT(*) FROM patients p
            LEFT JOIN analysis_results a ON p.dataset_id = a.dataset_id AND p.patient_id = a.patient_id
            WHERE {where_sql}
        """

        data_sql = f"""
            SELECT p.patient_id, p.data_json, a.cluster_label, a.is_anomaly, a.anomaly_score,
                   a.review_priority, a.key_abnormal_features, a.interpretation, a.run_id
            FROM patients p
            LEFT JOIN analysis_results a ON p.dataset_id = a.dataset_id AND p.patient_id = a.patient_id
            WHERE {where_sql}
            ORDER BY a.anomaly_score DESC, p.patient_id ASC
            LIMIT ? OFFSET ?
        """
        data_params = list(params) + [per_page, offset]

        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(count_sql, tuple(params))
            total_count = cursor.fetchone()[0] or 0

            cursor.execute(data_sql, tuple(data_params))
            raw_rows = [dict(r) for r in cursor.fetchall()]
            conn.close()
        else:
            with self.engine.connect() as conn:
                pdict = {f"p{i}": v for i, v in enumerate(params)}
                pg_count_sql = count_sql
                for i in range(len(params)):
                    pg_count_sql = pg_count_sql.replace("?", f":p{i}", 1)
                total_count = conn.execute(text(pg_count_sql), pdict).scalar() or 0

                pdict_data = {f"p{i}": v for i, v in enumerate(data_params)}
                pg_data_sql = data_sql
                for i in range(len(data_params)):
                    pg_data_sql = pg_data_sql.replace("?", f":p{i}", 1)
                res = conn.execute(text(pg_data_sql), pdict_data)
                raw_rows = [dict(r._mapping) for r in res.fetchall()]

        total_pages = max(1, math.ceil(total_count / per_page))
        records = []
        display_cols = []
        for r in raw_rows:
            p_dict = json.loads(r["data_json"]) if r.get("data_json") else {}
            if not display_cols:
                display_cols = [c for c in p_dict.keys() if c not in ["cluster_label", "is_anomaly", "anomaly_score", "review_priority", "key_abnormal_features", "interpretation", "run_id", "dataset_id"]][:7]
            p_dict["patient_id"] = r["patient_id"]
            p_dict["cluster_label"] = r.get("cluster_label")
            p_dict["is_anomaly"] = r.get("is_anomaly", 0)
            p_dict["anomaly_score"] = r.get("anomaly_score", 0.0)
            p_dict["review_priority"] = r.get("review_priority", "Normal Profile")
            p_dict["key_abnormal_features"] = r.get("key_abnormal_features")
            p_dict["interpretation"] = r.get("interpretation")
            p_dict["run_id"] = r.get("run_id")
            records.append(p_dict)

        # Get unique cluster labels list for filters
        cluster_query = "SELECT DISTINCT cluster_label FROM analysis_results WHERE dataset_id = ? ORDER BY cluster_label ASC"
        clusters_list = []
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            cursor = conn.cursor()
            cursor.execute(cluster_query, (dataset_id,))
            clusters_list = [r[0] for r in cursor.fetchall() if r[0] is not None]
            conn.close()
        else:
            with self.engine.connect() as conn:
                res = conn.execute(text("SELECT DISTINCT cluster_label FROM analysis_results WHERE dataset_id = :ds ORDER BY cluster_label ASC"), {"ds": dataset_id})
                clusters_list = [r[0] for r in res.fetchall() if r[0] is not None]

        return {
            "records": records,
            "total_count": total_count,
            "total_pages": total_pages,
            "current_page": page,
            "per_page": per_page,
            "display_cols": display_cols,
            "clusters_list": clusters_list
        }

    def get_paginated_anomalies(self, dataset_id="ds_default", page=1, per_page=25, search=None):
        """Server-side paginated candidate rare cases query."""
        import math
        dataset_id = str(dataset_id)
        page = max(1, int(page))
        per_page = max(1, min(100, int(per_page)))
        offset = (page - 1) * per_page

        where_clauses = ["a.dataset_id = ?", "a.is_anomaly = 1"]
        params = [dataset_id]

        if search and search.strip():
            s_term = f"%{search.strip().lower()}%"
            where_clauses.append("(LOWER(p.patient_id) LIKE ? OR LOWER(p.data_json) LIKE ?)")
            params.extend([s_term, s_term])

        where_sql = " AND ".join(where_clauses)

        stats_sql = f"""
            SELECT 
                COUNT(*) as total_anomalies,
                SUM(CASE WHEN a.review_priority = 'High Priority' THEN 1 ELSE 0 END) as high_cnt,
                SUM(CASE WHEN a.review_priority = 'Medium Priority' THEN 1 ELSE 0 END) as med_cnt,
                SUM(CASE WHEN a.review_priority = 'Low Priority' THEN 1 ELSE 0 END) as low_cnt
            FROM analysis_results a
            JOIN patients p ON a.dataset_id = p.dataset_id AND a.patient_id = p.patient_id
            WHERE {where_sql}
        """

        data_sql = f"""
            SELECT p.patient_id, p.data_json, a.cluster_label, a.is_anomaly, a.anomaly_score,
                   a.review_priority, a.key_abnormal_features, a.interpretation, a.run_id
            FROM analysis_results a
            JOIN patients p ON a.dataset_id = p.dataset_id AND a.patient_id = p.patient_id
            WHERE {where_sql}
            ORDER BY a.anomaly_score DESC, p.patient_id ASC
            LIMIT ? OFFSET ?
        """
        data_params = list(params) + [per_page, offset]

        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(stats_sql, tuple(params))
            stats_row = dict(cursor.fetchone() or {})

            cursor.execute(data_sql, tuple(data_params))
            raw_rows = [dict(r) for r in cursor.fetchall()]
            conn.close()
        else:
            with self.engine.connect() as conn:
                pdict = {f"p{i}": v for i, v in enumerate(params)}
                pg_stats_sql = stats_sql
                for i in range(len(params)):
                    pg_stats_sql = pg_stats_sql.replace("?", f":p{i}", 1)
                stats_res = conn.execute(text(pg_stats_sql), pdict).fetchone()
                stats_row = dict(stats_res._mapping) if stats_res else {}

                pdict_data = {f"p{i}": v for i, v in enumerate(data_params)}
                pg_data_sql = data_sql
                for i in range(len(data_params)):
                    pg_data_sql = pg_data_sql.replace("?", f":p{i}", 1)
                res = conn.execute(text(pg_data_sql), pdict_data)
                raw_rows = [dict(r._mapping) for r in res.fetchall()]

        total_anomalies = int(stats_row.get("total_anomalies") or 0)
        total_pages = max(1, math.ceil(total_anomalies / per_page))

        records = []
        for r in raw_rows:
            p_dict = json.loads(r["data_json"]) if r.get("data_json") else {}
            p_dict["patient_id"] = r["patient_id"]
            p_dict["cluster_label"] = r.get("cluster_label")
            p_dict["is_anomaly"] = r.get("is_anomaly", 1)
            p_dict["anomaly_score"] = r.get("anomaly_score", 0.0)
            p_dict["review_priority"] = r.get("review_priority", "High Priority")
            p_dict["key_abnormal_features"] = r.get("key_abnormal_features")
            p_dict["interpretation"] = r.get("interpretation")
            p_dict["run_id"] = r.get("run_id")
            records.append(p_dict)

        return {
            "records": records,
            "total_anomalies": total_anomalies,
            "total_pages": total_pages,
            "current_page": page,
            "per_page": per_page,
            "high_cnt": int(stats_row.get("high_cnt") or 0),
            "med_cnt": int(stats_row.get("med_cnt") or 0),
            "low_cnt": int(stats_row.get("low_cnt") or 0)
        }

    def save_feature_attributions(self, attributions, dataset_id="ds_default", run_id=None):
        """Bulk persist pre-computed feature attributions and Z-scores strictly for dataset_id."""
        dataset_id = str(dataset_id)
        if not attributions:
            return

        rows = []
        for a in attributions:
            rows.append((
                dataset_id,
                run_id,
                str(a["patient_id"]),
                str(a["feature_name"]),
                float(a["patient_value"]) if a.get("patient_value") is not None else None,
                float(a["reference_mean"]) if a.get("reference_mean") is not None else None,
                float(a["reference_std"]) if a.get("reference_std") is not None else None,
                float(a["z_score"]) if a.get("z_score") is not None else None,
                float(a["absolute_deviation"]) if a.get("absolute_deviation") is not None else None,
                int(a.get("is_abnormal", 0))
            ))

        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            cursor = conn.cursor()
            if run_id is not None:
                cursor.execute("DELETE FROM feature_attributions WHERE dataset_id = ? AND analysis_run_id = ?", (dataset_id, run_id))
            else:
                cursor.execute("DELETE FROM feature_attributions WHERE dataset_id = ?", (dataset_id,))
            cursor.executemany("""
                INSERT INTO feature_attributions (
                    dataset_id, analysis_run_id, patient_id, feature_name,
                    patient_value, reference_mean, reference_std, z_score,
                    absolute_deviation, is_abnormal
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            conn.close()
        else:
            with self.engine.begin() as conn:
                if run_id is not None:
                    conn.execute(text("DELETE FROM feature_attributions WHERE dataset_id = :ds AND analysis_run_id = :rid"), {"ds": dataset_id, "rid": run_id})
                else:
                    conn.execute(text("DELETE FROM feature_attributions WHERE dataset_id = :ds"), {"ds": dataset_id})
                for r in rows:
                    conn.execute(text("""
                        INSERT INTO feature_attributions (
                            dataset_id, analysis_run_id, patient_id, feature_name,
                            patient_value, reference_mean, reference_std, z_score,
                            absolute_deviation, is_abnormal
                        ) VALUES (:ds, :rid, :pid, :feat, :pval, :rmean, :rstd, :z, :absd, :abn)
                    """), {
                        "ds": r[0], "rid": r[1], "pid": r[2], "feat": r[3],
                        "pval": r[4], "rmean": r[5], "rstd": r[6], "z": r[7],
                        "absd": r[8], "abn": r[9]
                    })
        print(f"[SUCCESS] Saved {len(rows)} feature attribution records for dataset '{dataset_id}'.")

    def get_patient_feature_attributions(self, dataset_id, patient_id):
        """Fetch pre-computed Z-scores and feature attributions for a single patient in < 2ms."""
        dataset_id = str(dataset_id)
        patient_id = str(patient_id)
        query = """
            SELECT feature_name, patient_value, reference_mean, reference_std, z_score, absolute_deviation, is_abnormal
            FROM feature_attributions
            WHERE dataset_id = ? AND patient_id = ?
            ORDER BY ABS(z_score) DESC
        """
        rows = []
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (dataset_id, patient_id))
            rows = [dict(r) for r in cursor.fetchall()]
            conn.close()
        else:
            with self.engine.connect() as conn:
                res = conn.execute(text("""
                    SELECT feature_name, patient_value, reference_mean, reference_std, z_score, absolute_deviation, is_abnormal
                    FROM feature_attributions
                    WHERE dataset_id = :ds AND patient_id = :pid
                    ORDER BY ABS(z_score) DESC
                """), {"ds": dataset_id, "pid": patient_id})
                rows = [dict(r._mapping) for r in res.fetchall()]
        return rows

    # =========================================================================
    # ASYNCHRONOUS PIPELINE STAGE TRACKING & STATUS
    # =========================================================================

    def create_analysis_run(self, dataset_id="ds_default"):
        """Create a new tracked analysis execution job."""
        dataset_id = str(dataset_id)
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            try:
                cursor = conn.cursor()
                # Ensure parent dataset exists so foreign key is satisfied
                cursor.execute("""
                    INSERT OR IGNORE INTO datasets (dataset_id, dataset_name, source, filename, status)
                    VALUES (?, ?, 'Auto-Registered', 'patients.csv', 'Queued')
                """, (dataset_id, f"Dataset {dataset_id}"))
                cursor.execute("""
                    INSERT INTO analysis_runs (dataset_id, status, stage, progress, message)
                    VALUES (?, 'running', 'queued', 5, 'Analysis job queued...')
                """, (dataset_id,))
                run_id = cursor.lastrowid
                conn.commit()
                return run_id
            finally:
                conn.close()
        else:
            with self.engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO datasets (dataset_id, dataset_name, source, filename, status)
                    VALUES (:ds, :name, 'Auto-Registered', 'patients.csv', 'Queued')
                    ON CONFLICT (dataset_id) DO NOTHING
                """), {"ds": dataset_id, "name": f"Dataset {dataset_id}"})
                res = conn.execute(text("""
                    INSERT INTO analysis_runs (dataset_id, status, stage, progress, message)
                    VALUES (:ds, 'running', 'queued', 5, 'Analysis job queued...')
                    RETURNING run_id
                """), {"ds": dataset_id})
                run_id = res.scalar()
            return run_id

    def update_analysis_run(self, run_id, status=None, stage=None, progress=None, message=None, results_summary=None, k_distance_data=None, error_message=None):
        """Update asynchronous analysis job stage, progress, or completion metrics."""
        run_id = int(run_id)
        updates = ["updated_at = CURRENT_TIMESTAMP"]
        params = {}
        if status is not None:
            updates.append("status = :status")
            params["status"] = str(status)
        if stage is not None:
            updates.append("stage = :stage")
            params["stage"] = str(stage)
        if progress is not None:
            updates.append("progress = :progress")
            params["progress"] = int(progress)
        if message is not None:
            updates.append("message = :message")
            params["message"] = str(message)
        if results_summary is not None:
            updates.append("results_summary_json = :summary")
            params["summary"] = json.dumps(results_summary) if isinstance(results_summary, (dict, list)) else str(results_summary)
        if k_distance_data is not None:
            updates.append("k_distance_data_json = :kdist")
            params["kdist"] = json.dumps(k_distance_data) if isinstance(k_distance_data, (dict, list)) else str(k_distance_data)
        if error_message is not None:
            updates.append("error_message = :err")
            params["err"] = str(error_message)

        set_sql = ", ".join(updates)
        sql = f"UPDATE analysis_runs SET {set_sql} WHERE run_id = :run_id"
        params["run_id"] = run_id

        if self.use_sqlite_fallback:
            sqlite_sql = sql
            sqlite_params = []
            for k in params:
                sqlite_sql = sqlite_sql.replace(f":{k}", "?")
                sqlite_params.append(params[k])
            conn = self._get_raw_sqlite_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(sqlite_sql, tuple(sqlite_params))
                conn.commit()
            finally:
                conn.close()
        else:
            with self.engine.begin() as conn:
                conn.execute(text(sql), params)

    def get_analysis_run_status(self, run_id):
        """Retrieve execution status and progress for polling."""
        run_id = int(run_id)
        query = "SELECT * FROM analysis_runs WHERE run_id = ?"
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, (run_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
            finally:
                conn.close()
        else:
            with self.engine.connect() as conn:
                res = conn.execute(text("SELECT * FROM analysis_runs WHERE run_id = :rid"), {"rid": run_id})
                row = res.fetchone()
                return dict(row._mapping) if row else None

    def get_latest_analysis_run(self, dataset_id="ds_default"):
        """Retrieve most recent analysis run for dataset."""
        dataset_id = str(dataset_id)
        query = "SELECT * FROM analysis_runs WHERE dataset_id = ? ORDER BY run_id DESC LIMIT 1"
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, (dataset_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
            finally:
                conn.close()
        else:
            with self.engine.connect() as conn:
                res = conn.execute(text("SELECT * FROM analysis_runs WHERE dataset_id = :ds ORDER BY run_id DESC LIMIT 1"), {"ds": dataset_id})
                row = res.fetchone()
                return dict(row._mapping) if row else None

    def get_population_reference_stats(self, dataset_id="ds_default"):
        """Get pre-computed reference means and stds for dataset from feature_attributions."""
        dataset_id = str(dataset_id)
        query = "SELECT feature_name, reference_mean, reference_std FROM feature_attributions WHERE dataset_id = ? GROUP BY feature_name"
        means = {}
        stds = {}
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(query, (dataset_id,))
                for row in cursor.fetchall():
                    means[row[0]] = float(row[1]) if row[1] is not None else 0.0
                    stds[row[0]] = float(row[2]) if row[2] is not None else 1.0
            finally:
                conn.close()
        else:
            with self.engine.connect() as conn:
                res = conn.execute(text("SELECT feature_name, reference_mean, reference_std FROM feature_attributions WHERE dataset_id = :ds GROUP BY feature_name"), {"ds": dataset_id})
                for row in res.fetchall():
                    means[row[0]] = float(row[1]) if row[1] is not None else 0.0
                    stds[row[0]] = float(row[2]) if row[2] is not None else 1.0
        return means, stds

    def save_analysis_record(self, data):
        """
        Persist a complete ML analysis metadata record into PostgreSQL/SQLite
        and synchronously sync to Supabase analyses table.
        Does not store sensitive personal patient information.
        """
        analysis_id = str(data.get("analysis_id") or data.get("dataset_id") or f"an_{uuid.uuid4().hex[:8]}")
        rec_id = str(data.get("id") or str(uuid.uuid4()))
        dataset_name = str(data.get("dataset_name") or "Clinical Dataset")
        dataset_filename = str(data.get("dataset_filename") or data.get("filename") or "dataset.csv")
        source = str(data.get("source") or "User Upload")
        description = str(data.get("description") or "")
        total_records = int(data.get("total_records") or data.get("record_count") or 0)
        total_clusters = int(data.get("total_clusters") or data.get("number_of_clusters") or 0)
        anomalies = int(data.get("anomalies") or data.get("number_of_anomalies") or 0)
        anomaly_percentage = float(data.get("anomaly_percentage") or 0.0)
        eps = float(data.get("eps")) if data.get("eps") is not None else None
        min_samples = int(data.get("min_samples")) if data.get("min_samples") is not None else None
        silhouette_score = float(data.get("silhouette_score")) if data.get("silhouette_score") is not None else None
        status = str(data.get("status") or "completed")
        results_summary = data.get("results_summary") or {}
        results_json = json.dumps(results_summary) if isinstance(results_summary, dict) else str(results_summary)

        # 1. Save to local SQL engine (PostgreSQL or SQLite fallback)
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO analyses (
                        id, analysis_id, dataset_name, dataset_filename, source, description,
                        total_records, total_clusters, anomalies, anomaly_percentage,
                        eps, min_samples, silhouette_score, status, results_summary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(analysis_id) DO UPDATE SET
                        total_records=excluded.total_records,
                        total_clusters=excluded.total_clusters,
                        anomalies=excluded.anomalies,
                        anomaly_percentage=excluded.anomaly_percentage,
                        eps=excluded.eps,
                        min_samples=excluded.min_samples,
                        silhouette_score=excluded.silhouette_score,
                        status=excluded.status,
                        results_summary=excluded.results_summary
                """, (
                    rec_id, analysis_id, dataset_name, dataset_filename, source, description,
                    total_records, total_clusters, anomalies, anomaly_percentage,
                    eps, min_samples, silhouette_score, status, results_json
                ))
                conn.commit()
            except Exception as e:
                print(f"[WARNING] Local SQLite save_analysis_record failed: {e}")
            finally:
                conn.close()
        else:
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("""
                        INSERT INTO analyses (
                            id, analysis_id, dataset_name, dataset_filename, source, description,
                            total_records, total_clusters, anomalies, anomaly_percentage,
                            eps, min_samples, silhouette_score, status, results_summary
                        ) VALUES (
                            :id, :aid, :dname, :fname, :src, :desc,
                            :trec, :tclust, :anom, :anompct,
                            :eps, :mins, :sil, :status, CAST(:rsum AS jsonb)
                        )
                        ON CONFLICT (analysis_id) DO UPDATE SET
                            total_records=EXCLUDED.total_records,
                            total_clusters=EXCLUDED.total_clusters,
                            anomalies=EXCLUDED.anomalies,
                            anomaly_percentage=EXCLUDED.anomaly_percentage,
                            eps=EXCLUDED.eps,
                            min_samples=EXCLUDED.min_samples,
                            silhouette_score=EXCLUDED.silhouette_score,
                            status=EXCLUDED.status,
                            results_summary=EXCLUDED.results_summary
                    """), {
                        "id": rec_id, "aid": analysis_id, "dname": dataset_name, "fname": dataset_filename,
                        "src": source, "desc": description, "trec": total_records, "tclust": total_clusters,
                        "anom": anomalies, "anompct": anomaly_percentage, "eps": eps, "mins": min_samples,
                        "sil": silhouette_score, "status": status, "rsum": results_json
                    })
                    conn.commit()
            except Exception as e:
                print(f"[WARNING] PostgreSQL save_analysis_record failed: {e}")

        # 2. Sync to Supabase PostgREST table
        try:
            supabase_manager.save_analysis({
                "id": rec_id,
                "analysis_id": analysis_id,
                "dataset_name": dataset_name,
                "dataset_filename": dataset_filename,
                "source": source,
                "description": description,
                "total_records": total_records,
                "total_clusters": total_clusters,
                "anomalies": anomalies,
                "anomaly_percentage": anomaly_percentage,
                "eps": eps,
                "min_samples": min_samples,
                "silhouette_score": silhouette_score,
                "status": status,
                "results_summary": results_summary
            })
        except Exception as ex:
            print(f"[WARNING] Supabase sync deferred: {ex}")

        return analysis_id

    def get_all_analyses(self, limit=10, page=1, include_summary=False):
        """
        Fetch historical analysis runs with pagination and selective projection.
        Prioritizes live records from Supabase PostgreSQL, falls back to local database.
        """
        # Try Supabase API first if configured
        if supabase_manager.is_configured():
            remote_records = supabase_manager.get_all_analyses(limit=limit, page=page, include_summary=include_summary)
            if remote_records:
                for r in remote_records:
                    if isinstance(r.get("results_summary"), str):
                        try:
                            r["results_summary"] = json.loads(r["results_summary"])
                        except Exception:
                            r["results_summary"] = {}
                return remote_records

        # Fallback to local SQL storage
        offset = max(0, (page - 1) * limit)
        cols = "*" if include_summary else "id, analysis_id, dataset_name, dataset_filename, source, total_records, total_clusters, anomalies, anomaly_percentage, eps, min_samples, silhouette_score, status, created_at"
        query = f"SELECT {cols} FROM analyses ORDER BY created_at DESC LIMIT ? OFFSET ?"
        analyses = []
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.cursor()
                cursor.execute(query, (limit, offset))
                for row in cursor.fetchall():
                    d = dict(row)
                    if isinstance(d.get("results_summary"), str):
                        try:
                            d["results_summary"] = json.loads(d["results_summary"])
                        except Exception:
                            d["results_summary"] = {}
                    analyses.append(d)
            except Exception as e:
                print(f"[WARNING] Local analyses fetch error: {e}")
            finally:
                conn.close()
        else:
            try:
                with self.engine.connect() as conn:
                    res = conn.execute(text(f"SELECT {cols} FROM analyses ORDER BY created_at DESC LIMIT :l OFFSET :o"), {"l": limit, "o": offset})
                    for row in res.fetchall():
                        d = dict(row._mapping)
                        if isinstance(d.get("results_summary"), str):
                            try:
                                d["results_summary"] = json.loads(d["results_summary"])
                            except Exception:
                                d["results_summary"] = {}
                        analyses.append(d)
            except Exception as e:
                print(f"[WARNING] PostgreSQL analyses fetch error: {e}")
        return analyses

    def get_analyses_count(self):
        """Get total count of recorded analyses for pagination."""
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM analyses")
                res = cursor.fetchone()
                return int(res[0]) if res else 0
            except Exception:
                return 0
            finally:
                conn.close()
        else:
            try:
                with self.engine.connect() as conn:
                    res = conn.execute(text("SELECT COUNT(*) FROM analyses"))
                    val = res.scalar()
                    return int(val) if val is not None else 0
            except Exception:
                return 0

    def get_analysis_record(self, analysis_id):
        """Fetch a single analysis record by ID from Supabase or local storage."""
        analysis_id = str(analysis_id).strip()
        
        # Try Supabase first
        if supabase_manager.is_configured():
            remote = supabase_manager.get_analysis_by_id(analysis_id)
            if remote:
                if isinstance(remote.get("results_summary"), str):
                    try:
                        remote["results_summary"] = json.loads(remote["results_summary"])
                    except Exception:
                        remote["results_summary"] = {}
                return remote

        # Local fallback
        if self.use_sqlite_fallback:
            conn = self._get_raw_sqlite_connection()
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM analyses WHERE analysis_id = ? OR id = ? LIMIT 1", (analysis_id, analysis_id))
                row = cursor.fetchone()
                if row:
                    d = dict(row)
                    if isinstance(d.get("results_summary"), str):
                        try:
                            d["results_summary"] = json.loads(d["results_summary"])
                        except Exception:
                            d["results_summary"] = {}
                    return d
            finally:
                conn.close()
        else:
            try:
                with self.engine.connect() as conn:
                    res = conn.execute(text("SELECT * FROM analyses WHERE analysis_id = :id OR id = :id LIMIT 1"), {"id": analysis_id})
                    row = res.fetchone()
                    if row:
                        d = dict(row._mapping)
                        if isinstance(d.get("results_summary"), str):
                            try:
                                d["results_summary"] = json.loads(d["results_summary"])
                            except Exception:
                                d["results_summary"] = {}
                        return d
            except Exception as e:
                print(f"[WARNING] PostgreSQL fetch single analysis error: {e}")
        return None

# Global Singleton Database Manager instance
db_manager = DatabaseManager()


