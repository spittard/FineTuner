"""
Batch-match plugging records against the reference FAISS index via RPC.

Reads:   plugging_records.json   (from extract_plugging_records.py)
Writes:  plugging_matches.json   (one entry per plugging record)

Supports checkpointing: re-running resumes from where it left off.

Usage:
    python match_plugging_records.py
    python match_plugging_records.py --top-k 5 --checkpoint-every 200
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from finetuner.core.cache_rpc import connect, is_server_running

INPUT_FILE = "plugging_records.json"
OUTPUT_FILE = "plugging_matches.json"
CHECKPOINT_EVERY = 100
TOP_K = 5


def load_checkpoint(output_file: str) -> dict:
    """Load existing results for resume support. Returns dict keyed by row ID."""
    if not os.path.exists(output_file):
        return {}
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        # Index by row_id for fast lookup
        return {str(entry['row_id']): entry for entry in existing if 'row_id' in entry}
    except Exception as e:
        print(f"WARNING: Could not load checkpoint ({e}), starting fresh.")
        return {}


def save_results(results: list, output_file: str):
    """Save results list to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def build_entry(record: dict, matches: list) -> dict:
    """Build a plugging result entry from a source record and its top matches."""
    return {
        'row_id': record.get('ID'),
        'query_company': record.get('Company Name', ''),
        'query_city': record.get('City', ''),
        'query_state': record.get('State', ''),
        'matches': [
            {
                'rank': rank + 1,
                'name': m.get('name', ''),
                'city': m.get('city', ''),
                'state': m.get('state', ''),
                'id': m.get('id'),
                'score': m.get('score', 0.0),
                'name_score': m.get('name_score', 0.0),
                'string_score': m.get('string_score', 0.0),
                'semantic_score': m.get('semantic_score', 0.0),
                'normalized_semantic_score': m.get('normalized_semantic_score', m.get('semantic_score', 0.0)),
                'location_score': m.get('location_score', 0.0),
                'location_boost': m.get('location_boost', 0.0),
                'count': m.get('count', 0),
                'match_type': m.get('match_type', ''),
                'acronym_fidelity': m.get('acronym_fidelity', 0.0),
                'concept_alignment': m.get('concept_alignment', 0.0),
                'lexical_boost': m.get('lexical_boost', 0.0),
                'popularity_boost': m.get('popularity_boost', 0.0),
            }
            for rank, m in enumerate(matches)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='Batch-match plugging records against reference index')
    parser.add_argument('--input', default=INPUT_FILE, help=f'Input file (default: {INPUT_FILE})')
    parser.add_argument('--output', default=OUTPUT_FILE, help=f'Output file (default: {OUTPUT_FILE})')
    parser.add_argument('--top-k', type=int, default=TOP_K, help=f'Matches per record (default: {TOP_K})')
    parser.add_argument('--checkpoint-every', type=int, default=CHECKPOINT_EVERY,
                        help=f'Save checkpoint every N records (default: {CHECKPOINT_EVERY})')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only the first N records (for quick testing)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Queries per batched RPC search (default: 128). Set 1 to disable batching.')
    parser.add_argument('--no-resume', action='store_true',
                        help='Ignore existing output and start fresh')
    args = parser.parse_args()

    print("=" * 60)
    print("Plugging Records Batch Matcher")
    print(f"  Input:            {args.input}")
    print(f"  Output:           {args.output}")
    print(f"  Top-K matches:    {args.top_k}")
    print(f"  Checkpoint every: {args.checkpoint_every}")
    print("=" * 60)

    # Load input
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        print("       Run extract_plugging_records.py first.")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        plugging_records = json.load(f)

    print(f"\nLoaded {len(plugging_records):,} plugging records from {args.input}")

    if args.limit:
        plugging_records = plugging_records[:args.limit]
        print(f"Limiting to first {args.limit} records (--limit flag)")

    # Connect to RPC server
    if not is_server_running():
        print("ERROR: RPC cache server is not running (port 9876).")
        print("       Start it with: python -m finetuner.core.cache_rpc --serve")
        sys.exit(1)

    print("Connecting to RPC cache server...")
    server = connect()
    # Large index / long queries can exceed default Pyro timeout
    server._pyroTimeout = float(os.environ.get("PLUGGING_RPC_TIMEOUT", "900.0"))

    loaded = server.list_loaded_caches()
    if not loaded:
        print("ERROR: No caches are loaded in the RPC server.")
        print("       Load a cache first via the server, then retry.")
        sys.exit(1)

    cache_key = loaded[0]
    cache_info = server.get_cache_info(cache_key)
    print(f"Using cache: {cache_key}")
    if cache_info:
        print(f"  Companies in index: {cache_info.get('num_companies', 'unknown'):,}")
        print(f"  Location data:      {cache_info.get('has_location_data', False)}")

    # Load checkpoint (already processed records)
    if args.no_resume:
        done = {}
    else:
        done = load_checkpoint(args.output)
        if done:
            print(f"\nResuming: {len(done):,} records already processed, skipping.")

    # Collect results (preserve existing + add new)
    results = list(done.values())

    # Filter to unprocessed
    pending = [r for r in plugging_records if str(r.get('ID', '')) not in done]
    total = len(plugging_records)
    skipped = total - len(pending)

    print(f"\nProcessing {len(pending):,} remaining records ({skipped:,} skipped from checkpoint)...")
    print()

    start_time = time.time()
    processed = 0
    errors = 0

    batch_size = max(1, int(args.batch_size))
    for bstart in range(0, len(pending), batch_size):
        chunk = pending[bstart:bstart + batch_size]
        items = [
            [r.get('Company Name', ''), r.get('City', '') or '', r.get('State', '') or '']
            for r in chunk
        ]

        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = len(pending) - bstart
        eta = remaining / rate if rate > 0 else 0

        print(f"[{skipped + bstart + 1}-{skipped + bstart + len(chunk)}/{total}] "
              f"({elapsed:.0f}s, ETA: {eta:.0f}s) batch of {len(chunk)}…", end="", flush=True)
        try:
            t_rpc = time.time()
            batch_results = server.batch_search_loc(items, top_k=args.top_k)
            dt = time.time() - t_rpc
            print(f" {dt:.1f}s ({dt / max(1, len(chunk)):.2f}s/rec)", flush=True)

            for r, res in zip(chunk, batch_results):
                matches = res.get('results', []) if isinstance(res, dict) else []
                results.append(build_entry(r, matches))
                processed += 1

        except Exception as e:
            # One bad batch should not lose the whole chunk — fall back to per-record.
            print(f" BATCH ERROR: {e} — retrying per-record", flush=True)
            for r in chunk:
                try:
                    res = server.search(
                        r.get('Company Name', ''),
                        top_k=args.top_k,
                        city=(r.get('City') or None),
                        state=(r.get('State') or None),
                    )
                    results.append(build_entry(r, res.get('results', [])))
                    processed += 1
                except Exception as e2:
                    errors += 1
                    entry = build_entry(r, [])
                    entry['error'] = str(e2)
                    results.append(entry)

        # Checkpoint after every batch.
        save_results(results, args.output)
        print(f"   [checkpoint] {len(results):,} records saved to {args.output}", flush=True)

    # Final save
    save_results(results, args.output)

    total_time = time.time() - start_time
    file_mb = os.path.getsize(args.output) / (1024 * 1024)

    print()
    print("=" * 60)
    print("DONE")
    print(f"  Total records:    {total:,}")
    print(f"  Processed now:    {processed:,}")
    print(f"  Errors:           {errors:,}")
    print(f"  Output:           {args.output} ({file_mb:.1f} MB)")
    print(f"  Total time:       {total_time:.1f}s")
    if processed > 0:
        print(f"  Rate:             {processed / total_time:.1f} records/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
