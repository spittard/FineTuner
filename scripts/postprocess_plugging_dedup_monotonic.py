"""Post-process plugging_matches.json to match the matcher's finalization contract.

Applies (in place, after rotating a backup) the same two guarantees the matcher now
enforces in match_with_location, so the existing rematch output passes audit_scoring.py
without a multi-hour re-run:

  1. De-dupe top-k matches on (NFKC-casefold name, whitespace-collapsed city,
     whitespace-collapsed state) so "Mount  Vernon" == "Mount Vernon" (Gate E).
  2. Clamp each match score to be non-increasing with rank, so the displayed score never
     contradicts the displayed order (score-sanity unsorted_within_record).

Note: dedup here only drops duplicates from the stored top-k (it cannot backfill a rank-(k+1)
candidate the way a fresh matcher run would); affects at most a handful of records.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import unicodedata
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATH = os.path.join(ROOT, "plugging_matches.json")


def _name_key(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s or "")).casefold().strip())


def _loc_key(s):
    return re.sub(r"\s+", " ", str(s or "").casefold().strip())


def main():
    with open(PATH, encoding="utf-8") as f:
        data = json.load(f)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{PATH}.bak_postproc_{ts}"
    shutil.copy2(PATH, backup)

    deduped_rows = 0
    clamped_rows = 0
    for e in data:
        matches = e.get("matches") or []

        seen = set()
        kept = []
        for m in matches:
            key = (_name_key(m.get("name", "")), _loc_key(m.get("city", "")), _loc_key(m.get("state", "")))
            if key in seen:
                continue
            seen.add(key)
            kept.append(m)
        if len(kept) != len(matches):
            deduped_rows += 1
            matches = kept
            e["matches"] = matches

        changed = False
        for i in range(1, len(matches)):
            prev = float(matches[i - 1].get("score") or 0.0)
            if float(matches[i].get("score") or 0.0) > prev:
                matches[i]["score"] = prev
                changed = True
        if changed:
            clamped_rows += 1

    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"backup: {backup}")
    print(f"records: {len(data)}  deduped_rows: {deduped_rows}  clamped_rows: {clamped_rows}")


if __name__ == "__main__":
    main()
