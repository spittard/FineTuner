"""
Plugging Records Full Pipeline
==============================

Runs all steps to produce the SME plugging records match report:

  Step 1 — Kill any existing RPC cache server (port 9876)
  Step 2 — Re-extract reference companies from DB (excludes PluggingStatus='P')
  Step 3 — Start RPC cache server and build/load FAISS index from new data
  Step 4 — Extract plugging records (PluggingStatus='P') from DB
  Step 5 — Batch-match plugging records against reference index (resumable)
  Step 6 — Generate SME report (plugging_report.md + plugging_report.csv)
  Step 7 — Shut down RPC cache server

Usage:
    # Full pipeline (all steps)
    python run_plugging_pipeline.py

    # Skip re-extraction (reuse existing companies_with_location.json)
    python run_plugging_pipeline.py --skip-extract

    # Skip extraction + plugging pull (reuse existing JSON files)
    python run_plugging_pipeline.py --skip-extract --skip-plugging-extract

    # Resume matching from checkpoint, then regenerate report
    python run_plugging_pipeline.py --skip-extract --skip-plugging-extract --skip-match

    # Just regenerate the report from existing plugging_matches.json
    python run_plugging_pipeline.py --report-only

    # Sample mode: only extract/match/report N plugging records (for testing)
    python run_plugging_pipeline.py --sample 500
"""

import os
import sys
import time
import subprocess
import argparse
import signal

# Force offline mode before any imports that might trigger HuggingFace
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

REFERENCE_JSON = os.path.join(ROOT, "companies_with_location.json")
PLUGGING_JSON  = os.path.join(ROOT, "plugging_records.json")
MATCHES_JSON   = os.path.join(ROOT, "plugging_matches.json")
REPORT_MD      = os.path.join(ROOT, "plugging_report.md")
REPORT_CSV     = os.path.join(ROOT, "plugging_report.csv")

RPC_PORT = 9876
RPC_STARTUP_TIMEOUT = 600   # seconds to wait for cache server to load (large index = slow)
RPC_POLL_INTERVAL   = 5     # seconds between readiness polls


# ── Helpers ──────────────────────────────────────────────────────────────────

def banner(title: str):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def run(cmd: list, desc: str = "", check: bool = True) -> int:
    """Run a subprocess, streaming output. Returns exit code."""
    if desc:
        print(f"\n>>> {desc}")
    # Insert -u after the python executable for unbuffered output
    if cmd and cmd[0] == PYTHON:
        cmd = [cmd[0], "-u"] + cmd[1:]
    print(f"    {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if check and result.returncode != 0:
        print(f"\nERROR: Command failed (exit {result.returncode}): {' '.join(cmd)}")
        sys.exit(result.returncode)
    return result.returncode


def kill_rpc_server():
    """Kill any process listening on the RPC port."""
    banner("Step 1 — Stop existing RPC cache server (if running)")
    try:
        from finetuner.core.cache_rpc import is_server_running, connect
        if is_server_running():
            print("  RPC server is running — sending shutdown...")
            try:
                srv = connect()
                srv.shutdown()
                time.sleep(3)
                print("  Shutdown sent.")
            except Exception:
                pass
        else:
            print("  No RPC server running.")
    except Exception as e:
        print(f"  Could not check/stop via Pyro5 ({e}), trying port kill...")

    # Fallback: kill via netstat on Windows
    try:
        import subprocess as sp
        result = sp.run(
            ["powershell", "-Command",
             f"Get-NetTCPConnection -LocalPort {RPC_PORT} -State Listen -ErrorAction SilentlyContinue "
             f"| Select-Object -ExpandProperty OwningProcess"],
            capture_output=True, text=True
        )
        pids = [p.strip() for p in result.stdout.strip().splitlines() if p.strip().isdigit()]
        for pid in pids:
            print(f"  Killing PID {pid} on port {RPC_PORT}...")
            sp.run(["taskkill", "/F", "/PID", pid], capture_output=True)
    except Exception:
        pass

    print("  Done.")


def start_rpc_server() -> subprocess.Popen:
    """Start the Pyro5 RPC server in a background subprocess."""
    banner("Step 3a — Start RPC cache server")
    cmd = [PYTHON, "-m", "finetuner.core.cache_rpc", "--serve"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(ROOT, "src")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
    print(f"  Started RPC server (PID {proc.pid}) on port {RPC_PORT}")
    return proc


def wait_for_rpc_ready(timeout: int = RPC_STARTUP_TIMEOUT) -> bool:
    """Poll until the RPC server responds or timeout."""
    from finetuner.core.cache_rpc import is_server_running
    print(f"  Waiting for RPC server to become ready (timeout {timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_server_running():
            print("  RPC server is ready.")
            return True
        time.sleep(RPC_POLL_INTERVAL)
        print("  ...", end="", flush=True)
    print("\nERROR: RPC server did not become ready in time.")
    return False


def load_cache_into_rpc() -> str:
    """
    Load the best available location-aware cache into the RPC server.
    Prefers the largest valid cache already on disk to avoid rebuilding.
    Only triggers a full build when no usable cache exists.
    Returns the cache key that was loaded.
    """
    banner("Step 3b — Build / load FAISS cache")
    from finetuner.core.cache_rpc import connect

    srv = connect()
    srv._pyroTimeout = RPC_STARTUP_TIMEOUT

    # ── 1. Already loaded? ─────────────────────────────────────────────────
    loaded = srv.list_loaded_caches()
    if loaded:
        cache_key = loaded[0]
        info = srv.get_cache_info(cache_key)
        n = info.get('num_companies', 0) if info else 0
        has_loc = info.get('has_location_data', False) if info else False
        print(f"  Cache already loaded: {cache_key} ({n:,} companies, loc={has_loc})")
        print(f"  >>> RPC index: {cache_key}")
        return cache_key

    # ── 2. Find the best cache on disk ─────────────────────────────────────
    # Pick the location-aware cache with the most companies.
    available = srv.list_available_caches()
    loc_caches = [c for c in available
                  if c.get('has_location_data', False)
                  and isinstance(c.get('num_companies'), int)
                  and c['num_companies'] > 0]
    loc_caches.sort(key=lambda c: c.get('num_companies', 0), reverse=True)

    if loc_caches:
        best = loc_caches[0]
        cache_key = best['cache_key']
        n = best['num_companies']
        print(f"  Best cache on disk: {cache_key} ({n:,} companies)")
        print(f"  Loading into RPC server...")
        ok = srv.load_cache(cache_key)
        if ok:
            info = srv.get_cache_info(cache_key)
            if info:
                print(f"  Loaded: {info.get('num_companies', '?'):,} companies, "
                      f"location={info.get('has_location_data', False)}")
            print(f"  >>> Index loaded into RPC: {cache_key}")
            return cache_key
        print(f"  WARNING: Load failed for {cache_key}, will attempt rebuild.")

    # ── 3. No usable cache — build from reference JSON ────────────────────
    print(f"  No usable cache found on disk.")
    print(f"  Building from {REFERENCE_JSON} — this may take 30-60+ minutes...")
    build_cmd = [
        PYTHON, "-c",
        (
            "import os,sys;"
            "os.environ.setdefault('TRANSFORMERS_OFFLINE','1');"
            "os.environ.setdefault('HF_DATASETS_OFFLINE','1');"
            f"sys.path.insert(0,r'{os.path.join(ROOT, 'src')}');"
            "from finetuner.core.matcher import CompanyMatcher;"
            "m=CompanyMatcher();"
            f"ok=m.build_index_with_location(filepath=r'{REFERENCE_JSON}');"
            "print('Build OK:',ok)"
        )
    ]
    rc = run(build_cmd, desc="Building FAISS index", check=False)
    if rc != 0:
        print("ERROR: Index build failed.")
        sys.exit(1)

    # After build, pick the new best cache
    available = srv.list_available_caches()
    loc_caches = [c for c in available
                  if c.get('has_location_data', False)
                  and isinstance(c.get('num_companies'), int)
                  and c['num_companies'] > 0]
    loc_caches.sort(key=lambda c: c.get('num_companies', 0), reverse=True)
    if not loc_caches:
        print("ERROR: No cache found after build.")
        sys.exit(1)

    cache_key = loc_caches[0]['cache_key']
    print(f"  Loading newly-built cache: {cache_key}")
    ok = srv.load_cache(cache_key)
    if not ok:
        print(f"ERROR: RPC server failed to load cache: {cache_key}")
        sys.exit(1)
    return cache_key


def stop_rpc_server(proc: subprocess.Popen):
    """Gracefully stop the RPC server subprocess."""
    banner("Step 7 — Shut down RPC cache server")
    try:
        from finetuner.core.cache_rpc import connect, is_server_running
        if is_server_running():
            srv = connect()
            srv.shutdown()
            time.sleep(2)
    except Exception:
        pass
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("  RPC server stopped.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plugging Records Full Pipeline')
    parser.add_argument('--skip-extract',          action='store_true',
                        help='Skip Step 2: reuse existing companies_with_location.json')
    parser.add_argument('--skip-plugging-extract', action='store_true',
                        help='Skip Step 4: reuse existing plugging_records.json')
    parser.add_argument('--skip-match',            action='store_true',
                        help='Skip Step 5: reuse existing plugging_matches.json')
    parser.add_argument('--report-only',           action='store_true',
                        help='Only run Step 6 (generate report from existing matches)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Limit plugging extract + match to N records (e.g. --sample 10)')
    parser.add_argument('--top-k',                 type=int, default=5,
                        help='Matches per plugging record (default: 5)')
    args = parser.parse_args()

    if args.report_only:
        args.skip_extract = True
        args.skip_plugging_extract = True
        args.skip_match = True

    rpc_proc = None
    loaded_cache_key = None
    start_time = time.time()

    try:
        # ── Step 1: Kill any existing RPC server ─────────────────────────────
        kill_rpc_server()

        # ── Step 2: Re-extract reference companies ───────────────────────────
        if not args.skip_extract:
            banner("Step 2 — Extract reference companies (PluggingStatus != 'P')")
            print(f"  Output: {REFERENCE_JSON}")
            print("  NOTE: Full DB extraction may take 30+ minutes.")
            run([PYTHON, "extract_full_location_data.py"], desc="Extracting reference companies")
        else:
            banner("Step 2 — SKIPPED (using existing companies_with_location.json)")
            if not os.path.exists(REFERENCE_JSON):
                print(f"ERROR: {REFERENCE_JSON} not found. Remove --skip-extract to extract.")
                sys.exit(1)
            mb = os.path.getsize(REFERENCE_JSON) / (1024 * 1024)
            print(f"  Using existing file: {REFERENCE_JSON} ({mb:.0f} MB)")

        # ── Step 3: Start RPC server + build/load FAISS cache ────────────────
        if not args.skip_match and not args.report_only:
            rpc_proc = start_rpc_server()
            if not wait_for_rpc_ready():
                sys.exit(1)
            loaded_cache_key = load_cache_into_rpc()

        # ── Step 4: Extract plugging records ─────────────────────────────────
        if not args.skip_plugging_extract:
            banner("Step 4 — Extract plugging records (PluggingStatus = 'P')"
                   + (f"  [sample: {args.sample}]" if args.sample else ""))
            cmd = [PYTHON, "extract_plugging_records.py"]
            if args.sample:
                cmd += ["--max-rows", str(args.sample)]
            run(cmd, desc="Extracting plugging records")
        else:
            banner("Step 4 — SKIPPED (using existing plugging_records.json)")
            if not os.path.exists(PLUGGING_JSON):
                print(f"ERROR: {PLUGGING_JSON} not found. Remove --skip-plugging-extract.")
                sys.exit(1)
            import json
            with open(PLUGGING_JSON) as f:
                n = len(json.load(f))
            print(f"  Using existing file: {PLUGGING_JSON} ({n:,} records)")

        # ── Step 5: Batch match ───────────────────────────────────────────────
        if not args.skip_match:
            sample_note = f"  [limiting to {args.sample} records]" if args.sample else ""
            banner(f"Step 5 — Batch-match plugging records against reference index{sample_note}")
            checkpoint = min(args.sample, 10) if args.sample else 100
            cmd = [PYTHON, "match_plugging_records.py",
                   "--top-k", str(args.top_k),
                   "--no-resume",
                   "--checkpoint-every", str(checkpoint)]
            if args.sample:
                cmd += ["--limit", str(args.sample)]
            run(cmd, desc="Batch matching")
        else:
            banner("Step 5 — SKIPPED (using existing plugging_matches.json)")
            if not os.path.exists(MATCHES_JSON):
                print(f"ERROR: {MATCHES_JSON} not found. Remove --skip-match.")
                sys.exit(1)
            import json
            with open(MATCHES_JSON) as f:
                n = len(json.load(f))
            print(f"  Using existing file: {MATCHES_JSON} ({n:,} matched records)")

        # ── Step 6: Generate report ───────────────────────────────────────────
        banner("Step 6 — Generate SME plugging report")
        run([PYTHON,
             os.path.join("tests", "generate_plugging_report.py"),
             "--top-k", str(args.top_k)],
            desc="Generating report")

    finally:
        # ── Step 7: Shut down RPC server ──────────────────────────────────────
        if rpc_proc is not None:
            stop_rpc_server(rpc_proc)

    total = time.time() - start_time
    m, s = divmod(int(total), 60)
    banner("Pipeline Complete")
    print(f"  Total time:        {m}m {s}s")
    if loaded_cache_key:
        print(f"  RPC index loaded:   {loaded_cache_key}")
    print(f"  Reference index:   {REFERENCE_JSON}")
    print(f"  Plugging records:  {PLUGGING_JSON}")
    print(f"  Match results:     {MATCHES_JSON}")
    print(f"  Report (Markdown): {REPORT_MD}")
    print(f"  Report (CSV):      {REPORT_CSV}")
    print()


if __name__ == "__main__":
    main()
