#!/usr/bin/env python3
"""
Read-only: export a deterministic sample of AcctRef.Master rows for shadow-index experiments.
Writes:
  - companies_sample_plain.json  — no EmbedText; vectors would match name-only
  - companies_sample_embed.json  — optional per-row EmbedText for industry-augmented vectors

Format matches build_index_with_location: Company Name, City, State, ID (Row), Count, plus SIC/MarketSegment.
Does not write to production JSON paths.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

SERVER = r"TLG-DATA3\TLG_DEV"
DATABASE = "SQLWebRefTable"


def embed_line(name: str, sic: str, mkt: str) -> str:
    s = (sic or "").strip()
    t = s.upper() == "TBD"
    s2 = "" if t or not s else s
    m = (mkt or "").strip()
    t2 = m.upper() == "TBD"
    m2 = "" if t2 or not m else m
    return f"{name} | SIC: {s2} | Seg: {m2}".strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=50000, help="TOP N rows (default 50000)")
    p.add_argument(
        "--out-plain",
        default=os.path.join(
            os.path.dirname(__file__), "companies_sample_plain.json"
        ),
    )
    p.add_argument(
        "--out-embed",
        default=os.path.join(
            os.path.dirname(__file__), "companies_sample_embed.json"
        ),
    )
    args = p.parse_args()

    import pyodbc

    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};"
        f"DATABASE={DATABASE};Trusted_Connection=yes;TrustServerCertificate=yes;Encrypt=no"
    )
    cur = conn.cursor()
    sql = f"""
    SELECT TOP ({args.rows})
        [Row],
        LTRIM(RTRIM(CAST([ORIGINAL] AS NVARCHAR(MAX)))),
        LTRIM(RTRIM(CAST([City] AS NVARCHAR(200)))),
        LTRIM(RTRIM(CAST([State] AS NVARCHAR(200)))),
        LTRIM(RTRIM(CAST([SIC] AS NVARCHAR(200)))),
        LTRIM(RTRIM(CAST([MarketSegment] AS NVARCHAR(200))))
    FROM [AcctRef].[Master]
    WHERE [ORIGINAL] IS NOT NULL AND LTRIM(RTRIM([ORIGINAL])) <> ''
    ORDER BY [Row]
    """
    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()

    plain, embed = [], []
    for r in rows:
        row, name, city, st, sic, mkt = r
        rec = {
            "Company Name": name or "",
            "City": city or "",
            "State": st or "",
            "ID": int(row),
            "Count": 1,
            "SIC": (sic or "").strip(),
            "MarketSegment": (mkt or "").strip(),
        }
        plain.append(dict(rec))
        e = dict(rec)
        e["EmbedText"] = embed_line(rec["Company Name"], rec["SIC"], rec["MarketSegment"])
        embed.append(e)

    for path, data in ((args.out_plain, plain), (args.out_embed, embed)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(data):,} rows to {path}")


if __name__ == "__main__":
    main()
