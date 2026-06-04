#!/usr/bin/env python3
"""
Shadow Tier-0 industry labeler: fill the TBD SIC / MarketSegment on plugging rows by
similarity-weighted majority vote of their name-embedding neighbors' (curated) labels.

NON-DESTRUCTIVE. Reads only; writes new CSVs. Does NOT touch the live index, the
matcher, plugging_matches.json, or the DB (no writes).

Why this approach: every NON-plugging row already carries a human-curated SIC +
MarketSegment, and the FAISS index already embeds all of them. So we propagate labels
from a plugging row's nearest name-neighbors instead of training a new model.

Modes
-----
label    : label the 3,217 plugging rows.
           neighbors come from plugging_matches.json (fast, 5 voters) OR a fresh
           RPC search with a larger top_k (--neighbors rpc --top-k 25).
           -> writes plugging_industry_labels.csv

backtest : measure accuracy on a held-out sample of ALREADY-LABELED non-plugging rows
           (search each via RPC, drop self, vote, compare vote vs ground-truth).
           -> writes plugging_industry_backtest.csv + prints accuracy

Requires SQL Server (same conn as extract_plugging_records.py) for the label lookup,
and for `backtest`/`--neighbors rpc` a running cache RPC server (run_cache_server / cache_rpc).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

SERVER = r"TLG-DATA3\TLG_DEV"
DATABASE = "SQLWebRefTable"
DB_BATCH = 1000
EMPTY_LABELS = {"", "TBD", "tbd", "N/A", "NONE", "UNKNOWN"}


# --------------------------------------------------------------------------- DB
def _connect_db():
    import pyodbc

    return pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};"
        f"DATABASE={DATABASE};Trusted_Connection=yes;TrustServerCertificate=yes;Encrypt=no",
        timeout=15,
    )


def fetch_label_map(ids) -> dict:
    """Row id -> {'SIC': str, 'MarketSegment': str} for the given ids (batched)."""
    ids = sorted({int(i) for i in ids if i is not None})
    if not ids:
        return {}
    conn = _connect_db()
    cur = conn.cursor()
    out: dict = {}
    for i in range(0, len(ids), DB_BATCH):
        batch = ids[i : i + DB_BATCH]
        ph = ",".join("?" for _ in batch)
        cur.execute(
            f"""SELECT [Row],
                       LTRIM(RTRIM(CAST([SIC] AS NVARCHAR(100)))),
                       LTRIM(RTRIM(CAST([MarketSegment] AS NVARCHAR(100))))
                FROM [AcctRef].[Master] WHERE [Row] IN ({ph})""",
            [int(x) for x in batch],
        )
        for row, sic, mkt in cur.fetchall():
            out[int(row)] = {"SIC": (sic or "").strip(), "MarketSegment": (mkt or "").strip()}
    conn.close()
    return out


# ----------------------------------------------------------------------- voting
def _clean(label: str) -> str:
    return "" if (label or "").strip() in EMPTY_LABELS else (label or "").strip()


def vote(neighbors, label_map, field):
    """neighbors: list of (id, weight). Returns (pred, confidence, n_voters)."""
    weights = defaultdict(float)
    total = 0.0
    n = 0
    for nid, w in neighbors:
        rec = label_map.get(int(nid)) if nid is not None else None
        if not rec:
            continue
        lab = _clean(rec.get(field, ""))
        if not lab:
            continue
        w = max(float(w), 1e-6)
        weights[lab] += w
        total += w
        n += 1
    if not weights:
        return "", 0.0, 0
    pred = max(weights, key=weights.get)
    return pred, weights[pred] / total if total else 0.0, n


# ------------------------------------------------------------------------ label
def run_label(args):
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    use_rpc = args.neighbors == "rpc"
    server = None
    if use_rpc:
        from finetuner.core.cache_rpc import connect, is_server_running

        if not is_server_running():
            print("ERROR: --neighbors rpc requires a running cache RPC server.", file=sys.stderr)
            sys.exit(2)
        server = connect()

    # Build per-record neighbor lists (id, weight) and collect ids for label lookup.
    per_rec = []
    all_ids = set()
    for rec in data:
        if use_rpc:
            res = server.search(
                rec.get("query_company") or "",
                top_k=args.top_k,
                city=(rec.get("query_city") or None),
                state=(rec.get("query_state") or None),
            )
            ms = res.get("results", []) if isinstance(res, dict) else []
        else:
            ms = rec.get("matches") or []
        neigh = [(m.get("id"), m.get("score") or 0.0) for m in ms]
        top1_name = (ms[0].get("name") or "") if ms else ""
        per_rec.append((rec, neigh, top1_name))
        all_ids.update(nid for nid, _ in neigh if nid is not None)

    print(f"Collected {len(all_ids):,} unique neighbor ids; fetching labels from DB...")
    label_map = fetch_label_map(all_ids)
    print(f"Fetched labels for {len(label_map):,} ids.")

    out_path = os.path.join(ROOT, "plugging_industry_labels.csv")
    fields = [
        "row_id", "query_company", "query_city", "query_state",
        "n_neighbors", "pred_sic", "sic_confidence", "sic_voters",
        "pred_segment", "segment_confidence", "segment_voters",
        "top1_name", "top1_sic", "top1_segment",
    ]
    n_sic = n_seg = 0
    conf_hi = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec, neigh, top1_name in per_rec:
            sic, sic_c, sic_n = vote(neigh, label_map, "SIC")
            seg, seg_c, seg_n = vote(neigh, label_map, "MarketSegment")
            t1 = neigh[0][0] if neigh else None
            t1l = label_map.get(int(t1)) if t1 is not None else None
            if sic:
                n_sic += 1
            if seg:
                n_seg += 1
            if sic and sic_c >= args.accept:
                conf_hi += 1
            w.writerow({
                "row_id": rec.get("row_id", ""),
                "query_company": (rec.get("query_company") or "")[:200],
                "query_city": rec.get("query_city") or "",
                "query_state": rec.get("query_state") or "",
                "n_neighbors": len(neigh),
                "pred_sic": sic,
                "sic_confidence": f"{sic_c:.3f}",
                "sic_voters": sic_n,
                "pred_segment": seg,
                "segment_confidence": f"{seg_c:.3f}",
                "segment_voters": seg_n,
                "top1_name": top1_name[:120],
                "top1_sic": (t1l or {}).get("SIC", "") if t1l else "",
                "top1_segment": (t1l or {}).get("MarketSegment", "") if t1l else "",
            })
    tot = len(per_rec)
    print(f"\nWrote {out_path} ({tot} rows)")
    print(f"  SIC predicted:        {n_sic}/{tot} ({100*n_sic/tot:.1f}%)")
    print(f"  MarketSegment pred:   {n_seg}/{tot} ({100*n_seg/tot:.1f}%)")
    print(f"  SIC conf >= {args.accept}: {conf_hi}/{tot} ({100*conf_hi/tot:.1f}%) auto-acceptable")


# --------------------------------------------------------------------- backtest
def run_backtest(args):
    from finetuner.core.cache_rpc import connect, is_server_running

    if not is_server_running():
        print("ERROR: backtest requires a running cache RPC server.", file=sys.stderr)
        sys.exit(2)
    server = connect()

    conn = _connect_db()
    cur = conn.cursor()
    cur.execute(
        f"""SELECT TOP {args.sample} [Row], LTRIM(RTRIM([ORIGINAL])),
                   LTRIM(RTRIM(CAST([SIC] AS NVARCHAR(100)))),
                   LTRIM(RTRIM(CAST([MarketSegment] AS NVARCHAR(100)))),
                   LTRIM(RTRIM([City])), LTRIM(RTRIM([State]))
            FROM [AcctRef].[Master]
            WHERE ([PluggingStatus] IS NULL OR [PluggingStatus] <> 'P')
              AND [ORIGINAL] IS NOT NULL AND LTRIM(RTRIM([ORIGINAL])) <> ''
              AND [SIC] IS NOT NULL AND LTRIM(RTRIM([SIC])) NOT IN ('','TBD')
            ORDER BY NEWID()""")
    sample = [
        {"row": int(r[0]), "name": (r[1] or "").strip(), "sic": (r[2] or "").strip(),
         "seg": (r[3] or "").strip(), "city": (r[4] or "").strip(), "state": (r[5] or "").strip()}
        for r in cur.fetchall()
    ]
    conn.close()
    print(f"Sampled {len(sample):,} labeled non-plugging rows for backtest.")

    # Search each via RPC, drop self, collect neighbor ids.
    per = []
    all_ids = set()
    for i, s in enumerate(sample):
        res = server.search(s["name"], top_k=args.top_k + 3)
        ms = res.get("results", []) if isinstance(res, dict) else []
        neigh = [(m.get("id"), m.get("score") or 0.0) for m in ms if m.get("id") != s["row"]][: args.top_k]
        per.append((s, neigh))
        all_ids.update(nid for nid, _ in neigh if nid is not None)
        if (i + 1) % 50 == 0:
            print(f"  searched {i+1}/{len(sample)}")

    label_map = fetch_label_map(all_ids)

    out_path = os.path.join(ROOT, "plugging_industry_backtest.csv")
    fields = ["row", "name", "true_sic", "pred_sic", "sic_conf", "sic_ok",
              "true_segment", "pred_segment", "seg_conf", "seg_ok", "voters"]
    sic_ok = sic_tot = seg_ok = seg_tot = 0
    sic_ok_hi = sic_tot_hi = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s, neigh in per:
            psic, pc, pn = vote(neigh, label_map, "SIC")
            pseg, gc, gn = vote(neigh, label_map, "MarketSegment")
            ok_s = bool(psic) and psic.casefold() == s["sic"].casefold()
            ok_g = bool(pseg) and pseg.casefold() == s["seg"].casefold()
            if psic:
                sic_tot += 1
                sic_ok += int(ok_s)
                if pc >= args.accept:
                    sic_tot_hi += 1
                    sic_ok_hi += int(ok_s)
            if pseg:
                seg_tot += 1
                seg_ok += int(ok_g)
            w.writerow({
                "row": s["row"], "name": s["name"][:120], "true_sic": s["sic"],
                "pred_sic": psic, "sic_conf": f"{pc:.3f}", "sic_ok": int(ok_s),
                "true_segment": s["seg"], "pred_segment": pseg, "seg_conf": f"{gc:.3f}",
                "seg_ok": int(ok_g), "voters": pn,
            })

    def pct(a, b):
        return f"{100*a/b:.1f}%" if b else "n/a"

    print(f"\nWrote {out_path}")
    print("=" * 60)
    print(f"SIC accuracy (where predicted):     {sic_ok}/{sic_tot} = {pct(sic_ok, sic_tot)}")
    print(f"SIC accuracy @conf>={args.accept}:        {sic_ok_hi}/{sic_tot_hi} = {pct(sic_ok_hi, sic_tot_hi)}")
    print(f"MarketSegment accuracy (predicted): {seg_ok}/{seg_tot} = {pct(seg_ok, seg_tot)}")
    print(f"Coverage (got a SIC vote):          {sic_tot}/{len(sample)} = {pct(sic_tot, len(sample))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    pl = sub.add_parser("label", help="label the plugging rows")
    pl.add_argument("--input", default=os.path.join(ROOT, "plugging_matches.json"))
    pl.add_argument("--neighbors", choices=["matches", "rpc"], default="matches",
                    help="use neighbors already in plugging_matches.json, or fresh RPC search")
    pl.add_argument("--top-k", type=int, default=25, help="neighbors per record when --neighbors rpc")
    pl.add_argument("--accept", type=float, default=0.6, help="confidence threshold for auto-accept count")
    pl.set_defaults(func=run_label)

    bt = sub.add_parser("backtest", help="measure accuracy on labeled hold-outs")
    bt.add_argument("--sample", type=int, default=200)
    bt.add_argument("--top-k", type=int, default=25)
    bt.add_argument("--accept", type=float, default=0.6)
    bt.set_defaults(func=run_backtest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
