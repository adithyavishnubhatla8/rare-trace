import sys
import traceback
from pathlib import Path

# Add project root directory to sys.path so modules resolve cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from app import app

    class VercelPathFixerMiddleware:
        """
        WSGI Middleware to fix URL routing on Vercel Serverless environment.
        When Vercel rewrites requests to /api/index, PATH_INFO in the WSGI environment
        is overwritten with the destination file path ('/api/index' or '/api/index.py').
        This middleware extracts the client's actual requested path from Vercel's
        proxy headers (HTTP_X_MATCHED_PATH, HTTP_X_FORWARDED_URI, REQUEST_URI)
        or normalizes the path so that Flask routes (like '/', '/dashboard', '/api/process')
        match correctly without triggering Werkzeug 404s.
        """
        def __init__(self, wsgi_app):
            self.wsgi_app = wsgi_app

        def __call__(self, environ, start_response):
            matched_path = (
                environ.get("HTTP_X_MATCHED_PATH")
                or environ.get("x-matched-path")
                or environ.get("X-Matched-Path")
                or environ.get("HTTP_X_FORWARDED_URI")
                or environ.get("x-forwarded-uri")
                or environ.get("HTTP_X_VERCEL_FORWARDED_FOR_PATH")
                or environ.get("REQUEST_URI")
                or environ.get("RAW_URI")
            )
            if matched_path:
                path = str(matched_path).split("?")[0].strip()
                if path in ("", "/api/index", "/api/index.py", "/api", "/api/"):
                    environ["PATH_INFO"] = "/"
                elif path.startswith("/api/index/"):
                    environ["PATH_INFO"] = path[len("/api/index"):]
                elif path.startswith("/api/index.py/"):
                    environ["PATH_INFO"] = path[len("/api/index.py"):]
                else:
                    environ["PATH_INFO"] = path
            else:
                path = environ.get("PATH_INFO", "")
                if path in ("", "/api/index.py", "/api/index", "/api/index/", "/api", "/api/"):
                    environ["PATH_INFO"] = "/"
                elif path.startswith("/api/index.py/"):
                    environ["PATH_INFO"] = path[len("/api/index.py"):]
                elif path.startswith("/api/index/"):
                    environ["PATH_INFO"] = path[len("/api/index"):]

            return self.wsgi_app(environ, start_response)

    app.wsgi_app = VercelPathFixerMiddleware(app.wsgi_app)

except Exception as err:
    err_tb = traceback.format_exc()
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def vercel_import_error(path):
        return jsonify({
            "error": "RARETRACE Vercel Initialization Error",
            "details": str(err),
            "traceback": err_tb.splitlines()
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
