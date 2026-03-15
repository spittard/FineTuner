"""
Direct full workflow: build FAISS index → match plugging records → generate report.
Runs the build in-process (no subprocess) to maximize encoding speed.
Starts RPC server AFTER the build to avoid CPU competition during encoding.

Usage:
    python _full_workflow.py
"""
import os, sys, time, subprocess, json

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

ROOT    = os.path.dirname(os.path.abspath(__file__))
PYTHON  = sys.executable
REF_JSON     = os.path.join(ROOT, "companies_with_location.json")
PLUG_JSON    = os.path.join(ROOT, "plugging_records.json")
MATCHES_JSON = os.path.join(ROOT, "plugging_matches.json")
REPORT_MD    = os.path.join(ROOT, "plugging_report.md")
REPORT_CSV   = os.path.join(ROOT, "plugging_report.csv")
RPC_PORT = 9876

def banner(msg):
    print(); print("="*65); print(f"  {msg}"); print("="*65)

def kill_rpc():
    try:
        from finetuner.core.cache_rpc import is_server_running, connect
        if is_server_running():
            try: connect().shutdown()
            except Exception: pass
            time.sleep(3)
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["powershell","-Command",
             f"Get-NetTCPConnection -LocalPort {RPC_PORT} -State Listen -ErrorAction SilentlyContinue "
             f"| Select-Object -ExpandProperty OwningProcess"],
            capture_output=True, text=True)
        for pid in result.stdout.strip().splitlines():
            if pid.strip().isdigit():
                subprocess.run(["taskkill","/F","/PID",pid.strip()], capture_output=True)
    except Exception:
        pass

def wait_for_rpc(timeout=600):
    from finetuner.core.cache_rpc import is_server_running
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_server_running(): return True
        time.sleep(5); print("  ...", end="", flush=True)
    return False


# ── Step 1: Kill any stale RPC server ────────────────────────────────────────
banner("Step 1 — Kill stale RPC server")
kill_rpc()
print("  Done.")


# ── Step 2: Build FAISS index IN-PROCESS (maximum CPU speed) ─────────────────
banner("Step 2 — Build FAISS index (in-process, single-threaded fallback)")
print(f"  Source: {REF_JSON} ({os.path.getsize(REF_JSON)/1e6:.0f} MB)")

from finetuner.core.matcher import CompanyMatcher

t_build_start = time.time()
matcher = CompanyMatcher()   # reads model from model_config.json
ok = matcher.build_index_with_location(filepath=REF_JSON)

if not ok:
    print("ERROR: build_index_with_location() returned False. Aborting.")
    sys.exit(1)

build_elapsed = time.time() - t_build_start
print(f"\n  [OK] Build complete in {build_elapsed/60:.1f} min")

# Find the cache key that was just written
import glob
cache_dir = os.path.join(ROOT, "company_matcher_cache")
meta_files = glob.glob(os.path.join(cache_dir, "*_metadata.pkl"))
if not meta_files:
    print("ERROR: No cache metadata found after build.")
    sys.exit(1)
# Pick the newest metadata file
newest_meta = max(meta_files, key=os.path.getmtime)
cache_key = os.path.basename(newest_meta).replace("_metadata.pkl", "")
print(f"  Cache key: {cache_key}")


# ── Step 3: Start RPC server and load the new cache ──────────────────────────
banner("Step 3 — Start RPC cache server and load new index")
env = os.environ.copy()
env["PYTHONPATH"] = os.path.join(ROOT, "src")
rpc_proc = subprocess.Popen([PYTHON, "-m", "finetuner.core.cache_rpc", "--serve"],
                             cwd=ROOT, env=env)
print(f"  RPC server PID: {rpc_proc.pid}")
print("  Waiting for RPC server...")
if not wait_for_rpc():
    print("ERROR: RPC server did not start.")
    sys.exit(1)
print("  RPC server ready.")

from finetuner.core.cache_rpc import connect
srv = connect()
srv._pyroTimeout = 600
print(f"  Loading cache: {cache_key} ...")
ok = srv.load_cache(cache_key)
if not ok:
    print(f"ERROR: RPC server failed to load cache {cache_key}")
    sys.exit(1)
info = srv.get_cache_info(cache_key)
print(f"  Loaded: {info.get('num_companies',0):,} companies, location={info.get('has_location_data','?')}")


# ── Step 4: Match plugging records ────────────────────────────────────────────
banner("Step 4 — Batch-match plugging records")
print(f"  Input: {PLUG_JSON}")
with open(PLUG_JSON) as f:
    n_plug = len(json.load(f))
print(f"  Records: {n_plug:,}")

t_match = time.time()
cmd = [PYTHON, "-u", "match_plugging_records.py",
       "--top-k", "5",
       "--no-resume",
       "--checkpoint-every", "100"]
env2 = os.environ.copy()
env2["PYTHONPATH"] = os.path.join(ROOT, "src")
result = subprocess.run(cmd, cwd=ROOT, env=env2)
if result.returncode != 0:
    print("ERROR: Matching failed.")
    sys.exit(1)
print(f"  [OK] Matching done in {(time.time()-t_match)/60:.1f} min")


# ── Step 5: Generate report ───────────────────────────────────────────────────
banner("Step 5 — Generate SME plugging report")
result = subprocess.run(
    [PYTHON, os.path.join("tests","generate_plugging_report.py"), "--top-k", "5"],
    cwd=ROOT, env=env2)
if result.returncode != 0:
    print("WARNING: Report generation had errors.")


# ── Step 6: Shutdown ──────────────────────────────────────────────────────────
banner("Step 6 — Shutdown RPC server")
try:
    srv.shutdown()
except Exception:
    pass
if rpc_proc.poll() is None:
    rpc_proc.terminate()
    try: rpc_proc.wait(timeout=10)
    except Exception: rpc_proc.kill()
print("  Done.")


# ── Summary ───────────────────────────────────────────────────────────────────
total = time.time() - t_build_start
m, s = divmod(int(total), 60)
banner("Workflow Complete")
print(f"  Total time:      {m}m {s}s")
print(f"  Build time:      {build_elapsed/60:.1f} min")
print(f"  Cache key:       {cache_key}")
print(f"  Companies:       {info.get('num_companies',0):,}")
print(f"  Report (MD):     {REPORT_MD}")
print(f"  Report (CSV):    {REPORT_CSV}")
print()
