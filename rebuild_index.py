"""
5M Company Index Rebuild with Row IDs

Connects to SQL Server, extracts all companies with location (including plugging),
builds the FAISS index using all-MiniLM-L6-v2, and prints the cache key.

Uses Priority 0 (boost-only location) and Priority 1 (regional concept anchors)
from MATCHING_ARCHITECTURE.md.

Usage:
    python rebuild_index.py
    python rebuild_index.py --kill-previous    # Kill other rebuild processes first
    python rebuild_index.py --kill-only        # Only kill, then exit

Requires:
    - SQL Server access (Windows Auth)
    - ODBC Driver 17 for SQL Server
    - model_config.json with active_model = all-MiniLM-L6-v2
"""
import os
import sys
import glob
import argparse
import subprocess
import time

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

ROOT = os.path.dirname(os.path.abspath(__file__))

# DB config (matches extract_full_location_data.py)
SERVER = "TLG-DATA3\\TLG_DEV"
DATABASE = "SQLWebRefTable"
TABLE = "AcctRef.Master"


def _get_rebuild_pids() -> list:
    """Return list of PIDs running rebuild_index.py (excluding current process)."""
    my_pid = os.getpid()
    pids = []
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                if not cmdline:
                    continue
                cmdstr = " ".join(str(c) for c in cmdline)
                if "rebuild_index" in cmdstr and proc.info["pid"] != my_pid:
                    pids.append(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        if sys.platform == "win32":
            try:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" -ErrorAction SilentlyContinue | "
                     "Where-Object { $_.CommandLine -like '*rebuild_index*' } | "
                     "Select-Object -ExpandProperty ProcessId"],
                    capture_output=True, text=True, timeout=15
                )
                for line in (result.stdout or "").strip().splitlines():
                    if line.strip().isdigit():
                        pid = int(line.strip())
                        if pid != my_pid:
                            pids.append(pid)
            except Exception:
                pass
    return pids


def kill_previous_rebuild_processes() -> int:
    """
    Kill any other Python processes running rebuild_index.py.
    Returns number of processes killed.
    """
    pids = _get_rebuild_pids()
    killed = 0
    for pid in pids:
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               capture_output=True, timeout=5)
            else:
                os.kill(pid, 9)
            print(f"  Killed rebuild process PID {pid}")
            killed += 1
        except Exception as e:
            print(f"  WARNING: Could not kill PID {pid}: {e}")
    if killed:
        time.sleep(2)
    return killed


def main():
    parser = argparse.ArgumentParser(description="5M Company Index Rebuild with Row IDs")
    parser.add_argument("--kill-previous", action="store_true",
                        help="Kill any other running rebuild_index.py processes before starting")
    parser.add_argument("--kill-only", action="store_true",
                        help="Only kill previous rebuild processes, then exit")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit companies to N (for smoke tests, e.g. 100000)")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Force work dir for chunks (e.g. rebuild_work/24bf..._loc) - use to resume from existing chunks when cache key differs")
    args = parser.parse_args()

    if args.kill_previous or args.kill_only:
        print("Checking for previous rebuild processes...")
        n = kill_previous_rebuild_processes()
        if n == 0:
            print("  No previous rebuild processes found.")
        if args.kill_only:
            print("Done (--kill-only).")
            return

    from finetuner.data.dataset import CreateDataSet
    from finetuner.core.matcher import CompanyMatcher

    print("=" * 65)
    print("  5M Company Index Rebuild (with Row IDs)")
    print("=" * 65)

    # Step 1: Connect to SQL Server
    print("\nStep 1 — Connect to SQL Server")
    print(f"  Server:   {SERVER}")
    print(f"  Database: {DATABASE}")
    print(f"  Table:    {TABLE}")

    with CreateDataSet("sqlserver") as ds:
        if not ds.connect(DATABASE, server=SERVER, trusted_connection=True):
            print("ERROR: Failed to connect to SQL Server.")
            sys.exit(1)

        # Step 2: Extract companies with location (exclude_plugging=False per plan)
        print("\nStep 2 — Extract companies with location (exclude_plugging=False)")
        companies = ds.extract_data_with_location(
            TABLE,
            company_column="Original",
            city_column="City",
            state_column="State",
            row_column="Row",
            max_rows=args.limit,
            exclude_plugging=False,
            plugging_column="PluggingStatus",
        )

    if not companies:
        print("ERROR: No company data extracted.")
        sys.exit(1)

    print(f"  Extracted {len(companies):,} company records with Row IDs")

    # Step 3: Build index with CompanyMatcher (checkpointed rebuild for resume)
    print("\nStep 3 — Build FAISS index (all-MiniLM-L6-v2)")
    matcher = CompanyMatcher(model_name="all-MiniLM-L6-v2")
    cache_key = matcher.get_cache_key([c.get("Company Name", "") for c in companies]) + "_loc"
    work_dir = args.work_dir if hasattr(args, 'work_dir') and args.work_dir else os.path.join(ROOT, "rebuild_work", cache_key)
    print(f"  Work dir:  {work_dir} (checkpoint/resume enabled)")
    # Pass work_dir for checkpoint/resume. If you get TypeError, your matcher may not have work_dir yet;
    # fallback: build_index_with_location(data=companies)
    try:
        ok = matcher.build_index_with_location(data=companies, work_dir=work_dir)
    except TypeError:
        # Fallback if matcher lacks work_dir (e.g. stale cache)
        print("  (work_dir not supported, using one-shot encoding)")
        ok = matcher.build_index_with_location(data=companies)

    if not ok:
        print("ERROR: build_index_with_location() returned False.")
        sys.exit(1)

    # Step 4: Determine cache key and print
    cache_dir = os.path.join(ROOT, "company_matcher_cache")
    meta_files = glob.glob(os.path.join(cache_dir, "*_metadata.pkl"))
    if meta_files:
        newest = max(meta_files, key=os.path.getmtime)
        cache_key = os.path.basename(newest).replace("_metadata.pkl", "")
    else:
        cache_key = "(see company_matcher_cache for cache key)"

    print("\n" + "=" * 65)
    print("  Rebuild Complete")
    print("=" * 65)
    print(f"  Cache key:  {cache_key}")
    print(f"  Companies:  {len(companies):,}")
    print("  Use this cache key with the RPC server and webapp.")
    print()


if __name__ == "__main__":
    main()
