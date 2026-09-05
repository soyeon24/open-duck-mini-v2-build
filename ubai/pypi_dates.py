import json, sys, urllib.request
for pkg in sys.argv[1:]:
    d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json"))
    rows = [(f[0]["upload_time"][:10], v) for v, f in d["releases"].items() if f]
    rows.sort()
    win = [r for r in rows if "2025-05" <= r[0] <= "2025-10"]
    print(f"--- {pkg} (2025-05~2025-10) ---")
    for t, v in win[-12:]:
        print("  ", t, v)
