#!/usr/bin/env python3
"""
Phase 1: Counterfactual industry bonus on a frozen plugging_matches file.

Requires:
  - plugging_matches_frozen.json
  - row_industry_map.json  (from fetch_row_industry_map.py)

Outputs:
  - phase1_sensitivity.csv   (grid over lambda and optional mkt weight)
  - phase1_rank_swaps.json   (entries where top-1 candidate id would change, default lambda)
  - phase1_per_entry_default.csv  (one row per query at default lambda for audit)

No production files are modified.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(DIR, "..", ".."))


@dataclass
class Ind:
    sic: str
    mkt: str


def _norm_sic(s: str | None) -> str | None:
    if not s:
        return None
    t = s.strip()
    if not t or t.upper() == "TBD":
        return None
    return t


def get_ind(lookup: dict, row_id: int | str | None) -> Ind | None:
    if row_id is None:
        return None
    try:
        key = str(int(row_id))
    except (TypeError, ValueError):
        key = str(row_id)
    rec = lookup.get(key)
    if not rec:
        return None
    nsic = _norm_sic(rec.get("SIC", ""))
    return Ind(
        sic=nsic or "",
        mkt=(rec.get("MarketSegment") or "").strip() or "",
    )


def mkt_effective(m: str) -> str | None:
    if not m or m.upper() == "TBD":
        return None
    return m


def industry_bonus(
    q: Ind | None, c: Ind | None, lambda_sic: float, lambda_mkt: float
) -> float:
    if not q or not c:
        return 0.0
    b = 0.0
    if q.sic and c.sic and q.sic == c.sic:
        b += lambda_sic
    if (
        mkt_effective(q.mkt)
        and mkt_effective(c.mkt)
        and mkt_effective(q.mkt) == mkt_effective(c.mkt)
    ):
        b += lambda_mkt
    return b


def rerank_entry(
    entry: dict,
    lookup: dict,
    lambda_sic: float,
    lambda_mkt: float,
) -> tuple[list[dict], list[dict]]:
    """Return (old_matches, new_matches_sorted) with synthetic scores; copies only."""
    qid = entry.get("row_id")
    q = get_ind(lookup, qid)
    matches = deepcopy(entry.get("matches") or [])

    scored = []
    for m in matches:
        c = get_ind(lookup, m.get("id"))
        base = float(m.get("score", 0.0))
        if q and c:
            if (not q.sic) and (not mkt_effective(q.mkt)):
                extra = 0.0
            else:
                extra = industry_bonus(
                    Ind(q.sic, q.mkt), Ind(c.sic, c.mkt), lambda_sic, lambda_mkt
                )
        else:
            extra = 0.0
        adj = base + extra
        scored.append((adj, m))
    # Sort by adjusted score desc, then original rank
    scored.sort(
        key=lambda t: (
            -t[0],
            t[1].get("rank", 99),
        )
    )
    new_matches = [t[1] for t in scored]
    for i, m in enumerate(new_matches, 1):
        m["rank"] = i
    return matches, new_matches


def top_id(matches: list) -> int | None:
    if not matches:
        return None
    i = matches[0].get("id")
    try:
        return int(i) if i is not None else None
    except (TypeError, ValueError):
        return None


def sic_concordance(
    qid: int | str,
    top_id: int | None,
    lookup: dict,
) -> bool | None:
    """True if query and top-1 SIC (non empty) are equal. None if not evaluable."""
    q = get_ind(lookup, qid)
    c = get_ind(lookup, top_id) if top_id is not None else None
    if not q or not c:
        return None
    if not q.sic or not c.sic:
        return None
    return q.sic == c.sic


def main():
    default_lambdas = [0.0, 0.01, 0.03, 0.05, 0.1]
    default_lambda_write = 0.05
    mkt_w = 0.02  # small secondary boost if SIC+Market match

    frozen_path = os.path.join(DIR, "plugging_matches_frozen.json")
    map_path = os.path.join(DIR, "row_industry_map.json")
    for p in (frozen_path, map_path):
        if not os.path.exists(p):
            print(f"ERROR: missing {p}")
            print("  Run: python fetch_row_industry_map.py  (and ensure frozen file exists).")
            sys.exit(1)

    with open(frozen_path, "r", encoding="utf-8") as f:
        entries: list[dict] = json.load(f)
    with open(map_path, "r", encoding="utf-8") as f:
        lookup: dict = json.load(f)

    # --- Sensitivity grid ---
    sens_path = os.path.join(DIR, "phase1_sensitivity.csv")
    with open(sens_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "lambda_sic",
                "lambda_mkt",
                "entries",
                "entries_with_k_matches",
                "swaps",
                "swap_rate",
                "evaluable_sic_concordance_baseline",
                "sic_match_rate_baseline",
                "sic_match_rate_after_rerank_same_metric",
            ]
        )
        for lam in default_lambdas:
            swap_count = 0
            n_with = 0
            eval_sic = 0
            base_match = 0
            eval_after = 0
            after_match = 0
            for e in entries:
                ms = e.get("matches") or []
                if not ms:
                    continue
                n_with += 1
                old = deepcopy(ms)
                _, newm = rerank_entry(e, lookup, lam, mkt_w if lam > 0 else 0.0)
                if top_id(old) != top_id(newm):
                    swap_count += 1
                sc = sic_concordance(e.get("row_id"), top_id(old), lookup)
                if sc is not None:
                    eval_sic += 1
                    if sc:
                        base_match += 1
                sc2 = sic_concordance(e.get("row_id"), top_id(newm), lookup)
                if sc2 is not None:
                    eval_after += 1
                    if sc2:
                        after_match += 1
            w.writerow(
                [
                    lam,
                    mkt_w if lam > 0 else 0.0,
                    len(entries),
                    n_with,
                    swap_count,
                    f"{swap_count / n_with:.4f}" if n_with else "",
                    eval_sic,
                    f"{base_match / eval_sic:.4f}" if eval_sic else "",
                    f"{after_match / eval_after:.4f}" if eval_after else "",
                ]
            )

    print(f"Wrote {sens_path}")

    # --- Default lambda: swaps + per-entry audit ---
    swaps = []
    per_rows = []
    for e in entries:
        old_m, new_m = rerank_entry(
            e, lookup, default_lambda_write, mkt_w
        )
        o1 = top_id(old_m)
        n1 = top_id(new_m)
        swapped = o1 is not None and n1 is not None and o1 != n1
        if swapped:
            swaps.append(
                {
                    "row_id": e.get("row_id"),
                    "query_company": e.get("query_company", ""),
                    "old_top_id": o1,
                    "new_top_id": n1,
                }
            )
        qn = e.get("row_id")
        qi = get_ind(lookup, qn)
        oi = get_ind(lookup, o1) if o1 is not None else None
        ni = get_ind(lookup, n1) if n1 is not None else None
        per_rows.append(
            {
                "row_id": e.get("row_id"),
                "swapped": swapped,
                "query_sic": qi.sic if qi else "",
                "old_top1_sic": oi.sic if oi else "",
                "new_top1_sic": ni.sic if ni else "",
            }
        )

    out_swaps = os.path.join(DIR, "phase1_rank_swaps.json")
    with open(out_swaps, "w", encoding="utf-8") as f:
        json.dump(
            {
                "lambda_sic": default_lambda_write,
                "lambda_mkt": mkt_w,
                "swap_count": len(swaps),
                "swaps": swaps,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_swaps} ({len(swaps)} swaps)")

    per_path = os.path.join(DIR, "phase1_per_entry_default.csv")
    with open(per_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "row_id",
                "swapped",
                "query_sic",
                "old_top1_sic",
                "new_top1_sic",
            ],
        )
        w.writeheader()
        w.writerows(per_rows)
    print(f"Wrote {per_path} ({len(per_rows)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
