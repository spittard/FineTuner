#!/usr/bin/env python3
"""
Plugging eval harness — turns the failure patterns we found into a re-runnable
PROPERTY-based eval set, so the next matcher batch can be scored before/after.

There is no ground-truth "correct company" for the plugging corpus, so instead of
guessing labels we assert the SME *rules* that must hold (these are checkable):

Universal checks (every case, on the live top-5):
  - scores_monotonic        : score non-increasing by rank
  - no_dup_locrows          : no two of top-5 share (name, city, state)
  - no_us_state_echo        : no displayed candidate has a US-state-echo city (UT,UT)
  - valid_100               : score==100% only if exact name AND query+cand full geo equal

Seed-specific checks:
  - us_state    : expected_top1_contains holds AND no us-state echo shown
  - gate_a      : no_tie_mush (top-5 not all identical scores)
  - intl_geo    : intl_city_present (a real city==region echo, e.g. Beijing, still shown)
  - geoless     : geoless_exact_first (if an exact-name cand is in top-5, it ranks #1)
  - sample      : universal checks only

NON-DESTRUCTIVE. `build` reads plugging_matches.json + the assessment CSV and writes
tests/fixtures/plugging_eval_set.json. `run` executes each case via the live RPC and
writes plugging_eval_results.csv (+ prints pass/fail). Diff two run CSVs for before/after.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from finetuner.utils.text_preprocessor import TextPreprocessor as TP

EVAL_PATH = os.path.join(ROOT, "tests", "fixtures", "plugging_eval_set.json")


def nkey(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def is_us_state_token(s: str) -> bool:
    n = TP.normalize_state(s) if s else ""
    return len(n) == 2 and n in TP.STATE_ABBREV


def is_us_echo(m: dict) -> bool:
    c = str(m.get("city") or "").strip()
    st = str(m.get("state") or "").strip()
    return bool(c) and bool(st) and c.casefold() == st.casefold() and is_us_state_token(c)


def is_intl_echo(m: dict) -> bool:
    c = str(m.get("city") or "").strip()
    st = str(m.get("state") or "").strip()
    return bool(c) and bool(st) and c.casefold() == st.casefold() and not is_us_state_token(c)


# --------------------------------------------------------------------------- build
def build(args):
    data = json.load(open(os.path.join(ROOT, "plugging_matches.json"), encoding="utf-8"))
    by_id = {str(r.get("row_id")): r for r in data}

    assess = {}
    csv_path = os.path.join(ROOT, "plugging_report_assessment.csv")
    if os.path.exists(csv_path):
        for x in csv.DictReader(open(csv_path, encoding="utf-8")):
            assess[str(x["row_id"])] = x

    cases = []
    seen = set()

    def add(rec, source, **extra):
        rid = str(rec.get("row_id"))
        if rid in seen:
            return
        seen.add(rid)
        case = {
            "id": rid,
            "query": rec.get("query_company") or "",
            "city": rec.get("query_city") or "",
            "state": rec.get("query_state") or "",
            "source": source,
        }
        case.update(extra)
        cases.append(case)

    # 1) genuine US-state-echo highs + 2) Gate-A tie clusters (from assessment)
    for rid, x in assess.items():
        codes = x.get("issue_codes", "")
        rec = by_id.get(rid)
        if not rec:
            continue
        if "US_STATE_IN_CITY_FIELD" in codes:
            # expected top-1 = the strongest same-name candidate (defensible, unambiguous)
            ms = rec.get("matches") or []
            exp = ms[0].get("name") if ms else ""
            add(rec, "us_state", expected_top1_contains=nkey(exp).split(" ")[0] if exp else "")
        elif "GATE_A_TOP5_TIE_CLUSTER" in codes:
            add(rec, "gate_a")

    # 3) international city==region echoes (guard must NOT blank these)
    intl = 0
    for rec in data:
        if intl >= 12:
            break
        for m in (rec.get("matches") or []):
            if is_intl_echo(m):
                add(rec, "intl_geo", intl_city=str(m.get("city")).strip())
                intl += 1
                break

    # 4) stratified random sample: geo vs no-geo
    rng = random.Random(20260531)
    have_geo = [r for r in data if (str(r.get("query_city") or "").strip() or str(r.get("query_state") or "").strip())]
    no_geo = [r for r in data if not (str(r.get("query_city") or "").strip() or str(r.get("query_state") or "").strip())]
    rng.shuffle(have_geo)
    rng.shuffle(no_geo)
    for rec in have_geo[:20]:
        add(rec, "sample")
    for rec in no_geo[:20]:
        add(rec, "geoless")

    os.makedirs(os.path.dirname(EVAL_PATH), exist_ok=True)
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    from collections import Counter
    c = Counter(x["source"] for x in cases)
    print(f"Wrote {EVAL_PATH} ({len(cases)} cases)")
    for k, n in c.most_common():
        print(f"  {k:10} {n}")


# ----------------------------------------------------------------------------- run
def _checks(case, top):
    """Return dict check_name -> bool/None (None = not applicable)."""
    out = {}
    eps = 1e-6
    # scores_monotonic
    out["scores_monotonic"] = all(
        float(top[i].get("score") or 0) <= float(top[i - 1].get("score") or 0) + eps
        for i in range(1, len(top))
    ) if len(top) > 1 else True
    # no_dup_locrows
    keys = [(nkey(m.get("name")), str(m.get("city") or "").casefold().strip(),
             str(m.get("state") or "").casefold().strip()) for m in top]
    out["no_dup_locrows"] = len(set(keys)) == len(keys)
    # no_us_state_echo
    out["no_us_state_echo"] = not any(is_us_echo(m) for m in top)
    # valid_100
    if top and float(top[0].get("score") or 0) >= 0.9995:
        q, qc, qs = case["query"], case["city"], case["state"]
        t = top[0]
        ok = bool(qc.strip() and qs.strip()) and nkey(t.get("name")) == nkey(q)
        ok = ok and TP.normalize_state(t.get("state") or "") == TP.normalize_state(qs)
        ok = ok and TP.normalize_city(t.get("city") or "") == TP.normalize_city(qc)
        out["valid_100"] = ok
    else:
        out["valid_100"] = True

    src = case["source"]
    if src == "us_state":
        exp = case.get("expected_top1_contains") or ""
        out["expected_top1"] = (exp in nkey(top[0].get("name"))) if (top and exp) else None
    if src == "gate_a":
        scores = [round(float(m.get("score") or 0), 4) for m in top[:5]]
        out["no_tie_mush"] = len(set(scores)) > 1 if len(scores) > 1 else True
    if src == "intl_geo":
        city = nkey(case.get("intl_city") or "")
        out["intl_city_present"] = any(nkey(m.get("city")) == city for m in top) if city else None
    if src == "geoless":
        exacts = [m for m in top if nkey(m.get("name")) == nkey(case["query"])]
        if exacts:
            out["geoless_exact_first"] = nkey(top[0].get("name")) == nkey(case["query"])
        else:
            out["geoless_exact_first"] = None
    return out


def run(args):
    from finetuner.core.cache_rpc import connect, is_server_running

    if not is_server_running():
        print("ERROR: run needs the cache RPC server up with current matcher.py", file=sys.stderr)
        sys.exit(2)
    server = connect()
    cases = json.load(open(EVAL_PATH, encoding="utf-8"))

    out_path = os.path.join(ROOT, args.out)
    rows = []
    totals = {}
    fails = 0
    for case in cases:
        res = server.search(case["query"], top_k=10,
                            city=(case["city"].strip() or None),
                            state=(case["state"].strip() or None))
        top = (res.get("results", []) if isinstance(res, dict) else [])[:5]
        checks = _checks(case, top)
        applicable = {k: v for k, v in checks.items() if v is not None}
        passed = all(applicable.values())
        if not passed:
            fails += 1
        for k, v in applicable.items():
            t = totals.setdefault(k, [0, 0])
            t[1] += 1
            t[0] += int(bool(v))
        rows.append({
            "id": case["id"], "source": case["source"], "query": case["query"][:60],
            "geo": f"{case['city']},{case['state']}".strip(","),
            "top1": (top[0].get("name") if top else "")[:50],
            "top1_score": f"{float(top[0].get('score') or 0):.4f}" if top else "",
            "passed": int(passed),
            "failed_checks": ";".join(k for k, v in applicable.items() if not v),
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "source", "query", "geo", "top1",
                                          "top1_score", "passed", "failed_checks"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out_path} ({len(rows)} cases)")
    print("=" * 60)
    print(f"Cases passing ALL applicable checks: {len(rows) - fails}/{len(rows)}")
    print("\nPer-check pass rate:")
    for k in sorted(totals):
        ok, n = totals[k]
        print(f"  {k:22} {ok}/{n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build", help="select cases + assertions from flagged rows")
    b.set_defaults(func=build)
    r = sub.add_parser("run", help="run eval via RPC, write results CSV")
    r.add_argument("--out", default="plugging_eval_results.csv")
    r.set_defaults(func=run)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
