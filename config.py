import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Detect serverless environment (Vercel, AWS Lambda)
IS_SERVERLESS = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

class Config:
    IS_SERVERLESS = IS_SERVERLESS
    SECRET_KEY = os.getenv("SECRET_KEY", "default-rare-disease-secret-key-2026")
    
    # Paths
    BASE_DIR = BASE_DIR
    STORAGE_ROOT = Path("/tmp") if IS_SERVERLESS else BASE_DIR
    
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = STORAGE_ROOT / "saved_models"
    OUTPUTS_DIR = STORAGE_ROOT / "outputs"
    PLOTS_DIR = OUTPUTS_DIR / "plots"
    REPORTS_DIR = OUTPUTS_DIR / "reports"
    DATASET_PATH = DATA_DIR / "patients.csv"
    
    # SQLite Database Path: /tmp/rare_disease.db on Vercel, BASE_DIR/rare_disease.db locally
    SQLITE_DB_PATH = Path("/tmp") / "rare_disease.db" if IS_SERVERLESS else BASE_DIR / "rare_disease.db"
    
    # Database URL with Supabase/PostgreSQL normalization
    raw_db_url = os.getenv("DATABASE_URL", f"sqlite:///{SQLITE_DB_PATH}")
    if raw_db_url.startswith("postgres://"):
        raw_db_url = raw_db_url.replace("postgres://", "postgresql://", 1)
    DATABASE_URL = raw_db_url
    
    # Supabase Configuration
    SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", "")

    # Model Hyperparameters defaults
    DEFAULT_EPS = 3.5
    DEFAULT_MIN_SAMPLES = 10
    
    @classmethod
    def init_app(cls, app=None):
        """Safely initialize required directories without failing on read-only filesystems."""
        for target_dir in [cls.MODELS_DIR, cls.PLOTS_DIR, cls.REPORTS_DIR]:
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                # Silently defer directory creation in read-only environments
                pass

Config.init_app()
