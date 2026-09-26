import sys
from pathlib import Path

# Add project root directory to sys.path so modules resolve cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app

class VercelPathFixerMiddleware:
    """
    WSGI Middleware to fix URL routing on Vercel Serverless environment.
    When Vercel rewrites requests to /api/index.py, PATH_INFO in the WSGI environment
    is overwritten with the destination file path ('/api/index.py' or '/api/index').
    This middleware extracts the client's actual requested path from Vercel's
    proxy headers (HTTP_X_MATCHED_PATH, HTTP_X_FORWARDED_URI, HTTP_X_VERCEL_FORWARDED_FOR_PATH)
    or normalizes the path so that Flask routes (like '/', '/dashboard', '/api/process')
    match correctly without triggering Werkzeug 404s.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        # 1. Check Vercel edge proxy headers containing the actual requested route
        matched_path = (
            environ.get("HTTP_X_MATCHED_PATH")
            or environ.get("HTTP_X_FORWARDED_URI")
            or environ.get("HTTP_X_VERCEL_FORWARDED_FOR_PATH")
        )
        if matched_path:
            # Strip query string from matched path
            path = matched_path.split("?")[0]
            if not path or path == "":
                path = "/"
            environ["PATH_INFO"] = path
        else:
            path = environ.get("PATH_INFO", "")
            if path in ("/api/index.py", "/api/index", "/api/index/", "/api", "/api/"):
                environ["PATH_INFO"] = "/"
            elif path.startswith("/api/index.py/"):
                environ["PATH_INFO"] = path[len("/api/index.py"):]
            elif path.startswith("/api/index/"):
                environ["PATH_INFO"] = path[len("/api/index"):]

        return self.wsgi_app(environ, start_response)

# Apply WSGI middleware
app.wsgi_app = VercelPathFixerMiddleware(app.wsgi_app)

if __name__ == "__main__":
    app.run(debug=True)
