#!/usr/bin/env python3
"""
Build an isolated FAISS cache under experiments/industry_test/ (never company_matcher_cache).

Uses the same model as model_config.json. Pass the plain or embed sample JSON; use a
unique --cache-name per variant so on-disk keys do not collide.

Example:
  python build_industry_shadow_index.py --input companies_sample_plain.json --cache-name shadow_plain
  python build_industry_shadow_index.py --input companies_sample_embed.json --cache-name shadow_embed
"""
from __future__ import annotations

import argparse
import os
import sys

DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from finetuner.core.matcher import CompanyMatcher, _load_active_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        required=True,
        help="Path to companies JSON (plain or embed), under industry_test/ or absolute",
    )
    p.add_argument(
        "--cache-name",
        default="shadow_embed",
        help="Subfolder under industry_test/ for FAISS cache (default: shadow_embed)",
    )
    args = p.parse_args()

    inpath = args.input
    if not os.path.isabs(inpath):
        inpath = os.path.join(DIR, inpath)
    if not os.path.exists(inpath):
        print(f"ERROR: not found: {inpath}")
        sys.exit(1)

    cache_root = os.path.join(DIR, "cache", args.cache_name)
    os.makedirs(cache_root, exist_ok=True)

    model = _load_active_model()
    m = CompanyMatcher(model)
    m.cache_dir = cache_root
    m.ensure_cache_dir()
    print(f"Cache dir: {cache_root}")
    print(f"Input:     {inpath}")

    ok = m.build_index_with_location(filepath=inpath)
    if not ok:
        print("ERROR: build_index_with_location returned False")
        sys.exit(1)
    print("OK: shadow index build finished.")


if __name__ == "__main__":
    main()
