import os
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config

class SupabaseManager:
    """
    Manages communication with Supabase PostgreSQL via the PostgREST API
    and direct PostgreSQL connections. Does not expose credentials to the client.
    Stores and retrieves analysis metadata without storing sensitive patient PII.
    """
    SUMMARY_COLUMNS = "id,analysis_id,dataset_name,dataset_filename,source,total_records,total_clusters,anomalies,anomaly_percentage,eps,min_samples,silhouette_score,status,created_at"

    def __init__(self):
        self.supabase_url = Config.SUPABASE_URL.rstrip('/') if Config.SUPABASE_URL else ""
        self.anon_key = Config.SUPABASE_ANON_KEY or ""
        self._cache = {}
        self._cache_ttl = 15.0  # 15 seconds TTL for history queries
        self._last_cache_time = 0

    def is_configured(self):
        """Check if Supabase credentials are configured."""
        return bool(self.supabase_url and self.anon_key)

    def _get_headers(self, prefer_return=True):
        """Build request headers for Supabase PostgREST API."""
        headers = {
            "apikey": self.anon_key,
            "Authorization": f"Bearer {self.anon_key}",
            "Content-Type": "application/json",
            "User-Agent": "RARETRACE-Clinical-ML/1.0"
        }
        if prefer_return:
            headers["Prefer"] = "return=representation"
        return headers

    def save_analysis(self, data):
        """
        Insert an analysis record into Supabase analyses table.
        Does not store sensitive personal patient information.
        """
        if not self.is_configured():
            print("[INFO] Supabase credentials not configured. Skipping remote sync.")
            return False, "Supabase credentials not configured."

        # Invalidate read cache upon new write
        self._cache.clear()

        url = f"{self.supabase_url}/rest/v1/analyses"
        
        # Format payload strictly adhering to privacy guidelines
        payload = {
            "analysis_id": str(data.get("analysis_id") or data.get("dataset_id") or ""),
            "dataset_name": str(data.get("dataset_name") or "Clinical Dataset"),
            "dataset_filename": str(data.get("dataset_filename") or data.get("filename") or "dataset.csv"),
            "source": str(data.get("source") or "User Upload"),
            "total_records": int(data.get("total_records") or data.get("record_count") or 0),
            "total_clusters": int(data.get("total_clusters") or data.get("number_of_clusters") or 0),
            "anomalies": int(data.get("anomalies") or data.get("number_of_anomalies") or 0),
            "anomaly_percentage": float(data.get("anomaly_percentage") or 0.0),
            "eps": float(data.get("eps")) if data.get("eps") is not None else None,
            "min_samples": int(data.get("min_samples")) if data.get("min_samples") is not None else None,
            "silhouette_score": float(data.get("silhouette_score")) if data.get("silhouette_score") is not None else None,
            "status": str(data.get("status") or "completed"),
            "results_summary": data.get("results_summary") or {}
        }

        try:
            req_body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_body,
                headers=self._get_headers(prefer_return=True),
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                resp_text = response.read().decode("utf-8")
                records = json.loads(resp_text) if resp_text else [payload]
                print(f"[SUCCESS] Analysis record synced to Supabase (analysis_id={payload['analysis_id']})")
                return True, records[0] if records else payload

        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="replace")
            print(f"[WARNING] Supabase sync failed (HTTP {e.code}): {err_msg}")
            return False, f"HTTP {e.code}: {err_msg}"
        except Exception as e:
            print(f"[WARNING] Supabase communication error: {str(e)}")
            return False, str(e)

    def get_all_analyses(self, limit=10, page=1, include_summary=False):
        """Fetch historical analysis records from Supabase with column projection and pagination."""
        if not self.is_configured():
            return []

        import time
        now = time.time()
        cache_key = f"list_{limit}_{page}_{include_summary}"
        if cache_key in self._cache and (now - self._cache[cache_key]["time"]) < self._cache_ttl:
            return self._cache[cache_key]["data"]

        offset = max(0, (page - 1) * limit)
        cols = "*" if include_summary else self.SUMMARY_COLUMNS
        url = f"{self.supabase_url}/rest/v1/analyses?select={cols}&order=created_at.desc&limit={limit}&offset={offset}"
        
        try:
            req = urllib.request.Request(
                url,
                headers=self._get_headers(prefer_return=False),
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=6) as response:
                data = json.loads(response.read().decode("utf-8"))
                records = data if isinstance(data, list) else []
                self._cache[cache_key] = {"data": records, "time": now}
                return records
        except urllib.error.HTTPError as e:
            print(f"[INFO] Supabase get_all_analyses HTTP {e.code}")
            return []
        except Exception as e:
            print(f"[INFO] Supabase get_all_analyses error: {e}")
            return []

    def get_analysis_by_id(self, analysis_id):
        """Fetch a specific analysis record by its analysis_id from Supabase."""
        if not self.is_configured():
            return None

        clean_id = str(analysis_id).strip()
        import time
        now = time.time()
        cache_key = f"single_{clean_id}"
        if cache_key in self._cache and (now - self._cache[cache_key]["time"]) < self._cache_ttl:
            return self._cache[cache_key]["data"]

        url = f"{self.supabase_url}/rest/v1/analyses?analysis_id=eq.{clean_id}&select=*"
        try:
            req = urllib.request.Request(
                url,
                headers=self._get_headers(prefer_return=False),
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=6) as response:
                data = json.loads(response.read().decode("utf-8"))
                if data and isinstance(data, list) and len(data) > 0:
                    rec = data[0]
                    self._cache[cache_key] = {"data": rec, "time": now}
                    return rec
        except Exception as e:
            print(f"[INFO] Supabase get_analysis_by_id error: {e}")
        return None

# Singleton instance
supabase_manager = SupabaseManager()
