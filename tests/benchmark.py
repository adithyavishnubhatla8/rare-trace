import urllib.request
import time

urls = [
    'http://127.0.0.1:5000/dashboard',
    'http://127.0.0.1:5000/patients',
    'http://127.0.0.1:5000/anomalies',
    'http://127.0.0.1:5000/model-analysis',
    'http://127.0.0.1:5000/datasets',
    'http://127.0.0.1:5000/api/statistics',
    'http://127.0.0.1:5000/patients/PAT-10001',
    'http://127.0.0.1:5000/patients/PAT-10001/recommendations'
]

print("=" * 65)
print(f"{'ROUTE':<45} | {'STATUS':<6} | {'LATENCY':<8}")
print("=" * 65)

for u in urls:
    t0 = time.time()
    try:
        req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as res:
            code = res.getcode()
            elapsed_ms = round((time.time() - t0) * 1000, 1)
            route = u.replace('http://127.0.0.1:5000', '')
            print(f"{route:<45} | {code:<6} | {elapsed_ms:>6.1f} ms")
    except Exception as e:
        print(f"{u:<45} | ERR    | {e}")
print("=" * 65)
