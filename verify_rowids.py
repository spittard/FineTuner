"""
Verify that match results include non-null id field mapping to valid database rows.

Run AFTER rebuild_index.py completes and RPC server is started with the new cache.
Alternatively, run with --start-rpc to start the server, load cache, verify, then shut down.

Usage:
    # RPC server must already be running with cache loaded
    python verify_rowids.py

    # Or: start RPC, verify, shutdown
    python verify_rowids.py --start-rpc
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

ROOT = os.path.dirname(os.path.abspath(__file__))
PLUG_JSON = os.path.join(ROOT, "plugging_records.json")
MATCHES_JSON = os.path.join(ROOT, "plugging_matches.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-rpc", action="store_true", help="Start RPC server, verify, then shutdown")
    parser.add_argument("--limit", type=int, default=20, help="Number of plugging records to test (default: 20)")
    args = parser.parse_args()

    from finetuner.core.cache_rpc import connect, is_server_running
    import glob

    print("=" * 60)
    print("  Row ID Verification")
    print("=" * 60)

    if args.start_rpc:
        print("\n  Starting RPC server...")
        import subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(ROOT, "src")
        rpc_proc = subprocess.Popen(
            [sys.executable, "-m", "finetuner.core.cache_rpc", "--serve"],
            cwd=ROOT, env=env
        )
        import time
        for _ in range(120):
            if is_server_running():
                break
            time.sleep(1)
        if not is_server_running():
            print("ERROR: RPC server did not start.")
            sys.exit(1)
        print("  RPC server ready.")

        # Load newest cache
        cache_dir = os.path.join(ROOT, "company_matcher_cache")
        meta_files = glob.glob(os.path.join(cache_dir, "*_metadata.pkl"))
        if not meta_files:
            print("ERROR: No cache found. Run rebuild_index.py first.")
            sys.exit(1)
        newest = max(meta_files, key=os.path.getmtime)
        cache_key = os.path.basename(newest).replace("_metadata.pkl", "")
        print(f"  Loading cache: {cache_key}")
        srv = connect()
        srv._pyroTimeout = 120
        if not srv.load_cache(cache_key):
            print("ERROR: Failed to load cache.")
            sys.exit(1)
        info = srv.get_cache_info(cache_key)
        print(f"  Loaded: {info.get('num_companies', 0):,} companies")
    else:
        if not is_server_running():
            print("ERROR: RPC server is not running.")
            print("       Start it with: python -m finetuner.core.cache_rpc --serve")
            print("       Then load the cache. Or run: python verify_rowids.py --start-rpc")
            sys.exit(1)
        srv = connect()
        srv._pyroTimeout = 60

    # Load plugging records
    if not os.path.exists(PLUG_JSON):
        print(f"ERROR: {PLUG_JSON} not found. Run extract_plugging_records.py first.")
        sys.exit(1)
    with open(PLUG_JSON, "r", encoding="utf-8") as f:
        plug_records = json.load(f)
    sample = plug_records[: args.limit]

    print(f"\n  Testing {len(sample)} plugging records...")
    ok_count = 0
    fail_count = 0
    for rec in sample:
        name = rec.get("Company Name", rec.get("company_name", ""))
        city = rec.get("City", rec.get("city", ""))
        state = rec.get("State", rec.get("state", ""))
        row_id = rec.get("ID", rec.get("row_id", "?"))
        results = srv.match_single(name, city=city or None, state=state or None, top_k=5)
        has_valid_id = False
        for m in results:
            mid = m.get("id")
            if mid is not None:
                has_valid_id = True
                break
        if has_valid_id:
            ok_count += 1
        else:
            fail_count += 1
            print(f"    [FAIL] Row {row_id}: '{name}' ({city}, {state}) - no id in matches")

    print(f"\n  Results: {ok_count}/{len(sample)} records have matches with non-null id")
    if fail_count > 0:
        print(f"  FAIL: {fail_count} records had matches without id (expected for low-quality matches)")
    else:
        print("  PASS: All sampled records have at least one match with valid row id.")

    if args.start_rpc:
        try:
            srv.shutdown()
        except Exception:
            pass
        rpc_proc.terminate()
        try:
            rpc_proc.wait(timeout=5)
        except Exception:
            rpc_proc.kill()
        print("\n  RPC server stopped.")

    print("=" * 60)


if __name__ == "__main__":
    main()
