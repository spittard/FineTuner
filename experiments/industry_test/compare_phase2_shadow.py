#!/usr/bin/env python3
"""
Phase 2: Compare top-1 match for the same (company, city, state) on two shadow indexes:
  - Plain: name-only vectors (companies_sample_plain.json)
  - Embed: industry-augmented vectors (companies_sample_embed.json)

Query is the **company name** for both runs (asymmetric: industry only in index-side EmbedText).
Uses isolated cache dirs; does not use production company_matcher_cache.

Requires:
  - row_industry_map.json (SIC concordance proxy)
  - companies_sample_*.json from export_companies_sample.py
  - cache/shadow_plain and cache/shadow_embed from build_industry_shadow_index.py

Output: phase2_compare.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys

DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from finetuner.core.matcher import CompanyMatcher, _load_active_model  # noqa: E402


def _norm_sic(s: str) -> str | None:
    if not s or not s.strip() or s.strip().upper() == "TBD":
        return None
    return s.strip()


def top_id(res: list) -> int | None:
    if not res:
        return None
    i = res[0].get("id")
    if i is None:
        return None
    try:
        return int(i)
    except (TypeError, ValueError):
        return None


def sic_match(lookup: dict, qid: int | str, cid: int | None) -> str:
    if cid is None:
        return "no_candidate"
    ks = str(int(qid)) if not isinstance(qid, str) else str(qid)
    cs = str(int(cid)) if not isinstance(cid, str) else str(cid)
    rq = (lookup.get(ks) or {}).get("SIC", "")
    cc = (lookup.get(cs) or {}).get("SIC", "")
    a, b = _norm_sic(rq), _norm_sic(cc)
    if a and b and a == b:
        return "1"
    if not a or not b:
        return "na"
    return "0"


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=200, help="max plugging rows to compare")
    p.add_argument(
        "--plain-json",
        default="companies_sample_plain.json",
        help="path under industry_test/ or abs",
    )
    p.add_argument(
        "--embed-json",
        default="companies_sample_embed.json",
    )
    args = p.parse_args()

    frozen = os.path.join(DIR, "plugging_matches_frozen.json")
    mpath = os.path.join(DIR, "row_industry_map.json")
    for fpath in (frozen, mpath):
        if not os.path.exists(fpath):
            print(f"ERROR: missing {fpath}")
            sys.exit(1)

    with open(frozen, "r", encoding="utf-8") as f:
        entries = json.load(f)
    with open(mpath, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(DIR, p)

    p_json = resolve(args.plain_json)
    e_json = resolve(args.embed_json)
    c_plain = os.path.join(DIR, "cache", "shadow_plain")
    c_embed = os.path.join(DIR, "cache", "shadow_embed")
    for path, name in ((p_json, "plain"), (e_json, "embed"), (c_plain, "cache plain"), (c_embed, "cache embed")):
        if not os.path.exists(path):
            print(
                f"ERROR: {name} not found: {path}\n"
                "  Run: export_companies_sample.py && build_industry_shadow_index for both variants"
            )
            sys.exit(1)

    batch = []
    for e in entries:
        if len(batch) >= args.limit:
            break
        company = (e.get("query_company") or "").strip()
        if not company:
            continue
        city = (e.get("query_city") or "").strip() or None
        state = (e.get("query_state") or "").strip() or None
        batch.append(
            (e.get("row_id"), company, city, state)
        )

    def run_pass(cache_dir, json_path) -> list[tuple]:
        model = _load_active_model()
        m = CompanyMatcher(model)
        m.cache_dir = cache_dir
        k = m.get_cache_key_from_file(json_path) + "_loc"
        if not m.load_from_cache(k):
            print(f"ERROR: could not load {cache_dir} / {k}")
            sys.exit(1)
        out = []
        for qid, company, city, state in batch:
            try:
                r = m.match_with_location(
                    company, city=city, state=state, top_k=5
                )
                out.append((qid, company, top_id(r)))
            except Exception as ex:
                out.append((qid, company, f"err:{ex}"))
        return out

    print("Loading plain index and scoring...")
    plain_hits = run_pass(c_plain, p_json)
    print("Loading embed index and scoring...")
    embed_hits = run_pass(c_embed, e_json)

    out = os.path.join(DIR, "phase2_compare.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "row_id",
                "query_company",
                "top1_id_plain",
                "top1_id_embed",
                "agree",
                "sic_concord_plain",
                "sic_concord_embed",
            ]
        )
        for (q1, c1, t1), (q2, c2, t2) in zip(plain_hits, embed_hits, strict=True):
            assert q1 == q2
            if isinstance(t1, str) and str(t1).startswith("err"):
                w.writerow([q1, c1, t1, t2, "", "", ""])
                continue
            t1a = t1
            t2a = t2 if not (isinstance(t2, str) and str(t2).startswith("err")) else None
            w.writerow(
                [
                    q1,
                    c1[:120],
                    t1a,
                    t2a,
                    1 if t1a == t2a else 0,
                    sic_match(lookup, q1, t1a if isinstance(t1a, int) else None),
                    sic_match(lookup, q1, t2a if isinstance(t2a, int) else None),
                ]
            )
    n = len(plain_hits)
    print(f"Wrote {n} rows to {out}")


if __name__ == "__main__":
    main()
