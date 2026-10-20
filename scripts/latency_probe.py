import json, os, subprocess, sys, time, statistics
binary = sys.argv[1]
queries = ["structure overview", "phase folders lifecycle", "migration guide", "governance layer",
           "budget", "checklist", "risk register", "onboarding", "roles and responsibilities", "timeline"]
env = dict(os.environ)
p = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, env=env)
def call(i, method, params=None):
    p.stdin.write(json.dumps({"id": i, "method": method, "params": params or {}}) + "\n"); p.stdin.flush()
    return json.loads(p.stdout.readline())
call(0, "ping")
for mode, expand in [("hybrid", False), ("hybrid", True), ("keyword", False)]:
    lat = []
    for n in range(200):
        q = queries[n % len(queries)]
        t = time.perf_counter()
        r = call(n + 1, "search", {"query": q, "limit": 5, "mode": mode, "expand_links": expand})
        lat.append((time.perf_counter() - t) * 1000)
        assert r.get("ok"), r
    lat.sort()
    print(f"mode={mode} expand={expand} n={len(lat)} p50={statistics.median(lat):.2f}ms p95={lat[int(0.95*len(lat))-1]:.2f}ms max={lat[-1]:.2f}ms")
print(call(999, "stats")["result"]["documents_total"], "documents")
p.stdin.close(); p.wait(timeout=10)
