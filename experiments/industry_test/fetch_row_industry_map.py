#!/usr/bin/env python3
"""
Read-only: load Row, SIC, MarketSegment for all row IDs in plugging_matches_frozen.json.
Writes row_industry_map.json as { "row_id_str": { "SIC": "...", "MarketSegment": "..." } }.

Does not touch production data. Requires SQL Server access (same as extract_plugging_records.py).
"""
from __future__ import annotations

import json
import os
import sys

# Project root: experiments/industry_test -> ../..
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FROZEN = os.path.join(os.path.dirname(__file__), "plugging_matches_frozen.json")
OUT = os.path.join(os.path.dirname(__file__), "row_industry_map.json")

# Match extract_plugging_records.py
SERVER = r"TLG-DATA3\TLG_DEV"
DATABASE = "SQLWebRefTable"
BATCH = 1000  # keep IN list under SQL Server limits

sys.path.insert(0, os.path.join(ROOT, "src"))


def collect_ids(path: str) -> set:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    ids: set = set()
    for e in entries:
        rid = e.get("row_id")
        if rid is not None:
            ids.add(int(rid))
        for m in e.get("matches") or []:
            mid = m.get("id")
            if mid is not None:
                try:
                    ids.add(int(mid))
                except (TypeError, ValueError):
                    pass
    return ids


def main():
    if not os.path.exists(FROZEN):
        print(f"ERROR: {FROZEN} not found. Copy plugging_matches.json there first.")
        sys.exit(1)

    all_ids = sorted(collect_ids(FROZEN))
    print(f"Collected {len(all_ids):,} unique Row IDs (query + candidates).")

    import pyodbc

    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};"
        f"DATABASE={DATABASE};Trusted_Connection=yes;TrustServerCertificate=yes;Encrypt=no"
    )
    cur = conn.cursor()
    out_map: dict = {}
    for i in range(0, len(all_ids), BATCH):
        batch = all_ids[i : i + BATCH]
        placeholders = ",".join("?" for _ in batch)
        sql = f"""
        SELECT [Row], LTRIM(RTRIM(CAST([SIC] AS NVARCHAR(100)))),
               LTRIM(RTRIM(CAST([MarketSegment] AS NVARCHAR(100))))
        FROM [AcctRef].[Master]
        WHERE [Row] IN ({placeholders})
        """
        cur.execute(sql, [int(x) for x in batch])
        for row, sic, mkt in cur.fetchall():
            key = str(int(row))
            out_map[key] = {
                "SIC": (sic or "").strip(),
                "MarketSegment": (mkt or "").strip(),
            }
    conn.close()

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out_map, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out_map):,} rows to {OUT}")


if __name__ == "__main__":
    main()
