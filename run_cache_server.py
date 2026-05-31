#!/usr/bin/env python3
"""
Cache Index Server Runner.

Starts the Cache Index Server with optional auto-loading and file watching.

Usage:
    # Start with default settings
    python run_cache_server.py
    
    # Start and auto-load the main index
    python run_cache_server.py --load companies_with_location.json
    
    # Start with file watching enabled
    python run_cache_server.py --watch
    
    # Full example
    python run_cache_server.py --load companies_with_location.json --watch --port 8765
"""

import argparse
import sys
import os

# Prevent sentence-transformers / HuggingFace from making network calls.
# The model is already cached locally; this avoids proxy/firewall errors on startup.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from finetuner.core.cache_server import CacheIndexServer
from finetuner.core.cache_api import run_api, get_server


def main():
    parser = argparse.ArgumentParser(
        description='Cache Index Server - Persistent in-memory cache for company matching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Start server:     python run_cache_server.py
  Auto-load cache:  python run_cache_server.py --load companies_with_location.json
  With watching:    python run_cache_server.py --load companies_with_location.json --watch
  Custom port:      python run_cache_server.py --port 9000
        """
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8765,
        help='Port to run the API server on (default: 8765)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--load', '-l',
        type=str,
        help='Source file to auto-load on startup (e.g., companies_with_location.json)'
    )
    
    parser.add_argument(
        '--watch', '-w',
        action='store_true',
        help='Enable file watching for source changes'
    )
    
    parser.add_argument(
        '--cache-key', '-k',
        type=str,
        help='Specific cache key to load (overrides auto-detection from file)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable Flask debug mode'
    )
    
    parser.add_argument(
        '--list-caches',
        action='store_true',
        help='List available caches and exit'
    )
    
    args = parser.parse_args()
    
    # Initialize server
    server = get_server()
    
    # List caches mode
    if args.list_caches:
        print("\nAvailable caches:")
        print("-" * 60)
        caches = server.list_available_caches()
        if not caches:
            print("  No caches found in company_matcher_cache/")
        else:
            for c in caches:
                status = "LOADED" if c.get('loaded') else ""
                companies = c.get('num_companies', 'unknown')
                loc = " (with location)" if c.get('has_location_data') else ""
                print(f"  {c['cache_key']}: {companies:>10} companies{loc} {status}")
        print()
        return
    
    # Auto-load if specified
    if args.load:
        load_file = args.load
        if not os.path.exists(load_file):
            print(f"Error: File not found: {load_file}")
            sys.exit(1)

        print(f"\nAuto-loading from: {load_file}")

        from finetuner.core.matcher import CompanyMatcher, _load_active_model

        active_model = _load_active_model()
        cache_key = None
        keys_to_try = []

        if args.cache_key:
            keys_to_try = [args.cache_key]
        else:
            temp_matcher = CompanyMatcher(model_name=active_model)
            file_cache_key = temp_matcher.get_cache_key_from_file(load_file)
            if file_cache_key:
                keys_to_try = [file_cache_key + "_loc", file_cache_key]
            else:
                print(f"Warning: Could not determine cache key for {load_file}")

        for ck in keys_to_try:
            if server.load_cache(ck, model_name=active_model):
                cache_key = ck
                print(f"[OK] Loaded cache: {cache_key} (model={active_model})")
                break

        if keys_to_try and cache_key is None:
            print(f"Warning: No cache on disk matched {load_file} for model {active_model}.")
            print("         Use --cache-key <key> to load an existing index (see --list-caches).")
            print("         Otherwise the index will be built on first use (full embedding run).")

        # Set up file watching if requested
        if args.watch and cache_key:
            server.watch_source(load_file, cache_key)
            print(f"File watching enabled for: {load_file}")
    
    # Start API server
    print(f"\nStarting Cache Index Server...")
    run_api(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
