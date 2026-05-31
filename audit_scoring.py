#!/usr/bin/env python3
"""Audit plugging_matches.json against the agreed scoring quality gates.

Read-only. Deterministic. Prints PASS/FAIL per gate and exits with status
0 only if every gate passes. Designed so a cheaper model can iterate on
matcher.py without ambiguity:

    python match_plugging_records.py --no-resume --output plugging_matches.json
    python tests/generate_plugging_report.py
    python audit_scoring.py        # exit 0 = goal reached

Each gate has a hard numeric threshold (or a list of row IDs) so no
judgment is needed. The script never modifies any file.

Pattern -> Gate mapping (see plan doc):

  A  Gate-cap collapse:     <= 5 records with all top-5 within 0.001
  B  Token-reorder identity: row 7925804 'Henry Linda' top-1 score < 0.95
                             row 7880903 'Corporation of Hamilton' top-1 score < 0.95
  C  Acronym junk top-1:    <= 5 short-query acronym top-1 with score >= 0.85
                             AND name_score < 0.05
  D  Wrong-state top-1:     all listed Groups360 rows top-1 in queried state
                             row 7924623 SANCC top-1 in VA
  E  Same-record dupes:     0 records with same (norm-name, norm-city,
                             norm-state) in top-5
  F  0.88 cliff cluster:    <= 100 records top-1 == 0.880 (down from 252)
  H  Freq asymmetry:        N/A (structural; verified via Pattern A relief)
  J  Low-confidence float:  rows below 0.55 are present and are detectable;
                             this is informational, not a hard gate
  Score sanity:              every score in [0, 1]
                             every score has score == score (no NaN)
                             top-K within each record is sorted DESC

Exit codes:
  0  All hard gates pass.
  1  At least one gate failed.
  2  Input file missing or unreadable.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata

INPUT_PATH = os.environ.get(
    "AUDIT_INPUT_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugging_matches.json"),
)


def _norm_legal(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s or "")).casefold().strip())


def _norm_loc(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").casefold().strip())


def _load(path: str):
    if not os.path.exists(path):
        print(f"FAIL: input file not found: {path}", file=sys.stderr)
        sys.exit(2)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _by_id(d, rid):
    return next((e for e in d if int(e.get("row_id") or 0) == int(rid)), None)


def gate_score_sanity(d) -> tuple[bool, str]:
    bad_range = 0
    bad_nan = 0
    bad_sort = 0
    for e in d:
        prev = None
        for m in e.get("matches", []):
            s = m.get("score")
            try:
                sf = float(s)
            except Exception:
                bad_nan += 1
                continue
            if sf != sf:
                bad_nan += 1
                continue
            if not (0.0 <= sf <= 1.0):
                bad_range += 1
            if prev is not None and sf > prev + 1e-9:
                bad_sort += 1
            prev = sf
    ok = (bad_range == 0 and bad_nan == 0 and bad_sort == 0)
    msg = (
        f"score sanity: out_of_range={bad_range} nan={bad_nan} unsorted_within_record={bad_sort}"
    )
    return ok, msg


def gate_a_tied_clusters(d, max_allowed: int = 5) -> tuple[bool, str]:
    n = 0
    for e in d:
        ms = e.get("matches", [])
        if len(ms) < 5:
            continue
        scores = [float(m.get("score") or 0) for m in ms[:5]]
        if max(scores) - min(scores) < 0.001:
            n += 1
    ok = n <= max_allowed
    return ok, f"Gate A (top-5 tie clusters within 0.001): {n} (max allowed {max_allowed})"


def gate_b_token_reorder(d) -> tuple[bool, str]:
    failures = []
    for rid, q, max_score in [
        (7925804, "Henry Linda", 0.95),
        (7880903, "Corporation of Hamilton", 0.95),
    ]:
        e = _by_id(d, rid)
        if not e or not e.get("matches"):
            failures.append(f"row {rid}: no matches")
            continue
        s = float(e["matches"][0].get("score") or 0)
        if s >= max_score:
            failures.append(f"row {rid} '{q}' top-1 score {s:.3f} >= {max_score}")
    ok = not failures
    return ok, f"Gate B (token reorder demoted): {'OK' if ok else '; '.join(failures)}"


def gate_c_acronym_junk(d, max_allowed: int = 5) -> tuple[bool, str]:
    """Short-query acronym path top-1 should not score >= 0.85 with name_score < 0.05."""
    n = 0
    examples = []
    for e in d:
        q = (e.get("query_company") or "").strip()
        if len(q) > 4:
            continue
        ms = e.get("matches", [])
        if not ms:
            continue
        top = ms[0]
        mt = (top.get("match_type") or "").lower()
        if mt not in ("acronym_expansion", "acronym_reverse"):
            continue
        score = float(top.get("score") or 0)
        ns = float(top.get("name_score") or 0)
        if score >= 0.85 and ns < 0.05:
            n += 1
            if len(examples) < 3:
                examples.append(f"{e.get('row_id')} '{q}'->'{top.get('name')}'@{score:.3f}")
    ok = n <= max_allowed
    suffix = "" if ok else "  ex=" + " | ".join(examples)
    return ok, f"Gate C (acronym junk top-1 >= 0.85 with name_score<0.05): {n} (max {max_allowed}){suffix}"


def gate_d_wrong_state(d) -> tuple[bool, str]:
    """Specific rows where the top-1 must be in the queried state."""
    failures = []
    expectations = [
        (7925893, "md"),
        (7927251, "mn"),
        (7927362, "nh"),
        (7927747, "ut"),
        (7924623, "va"),
    ]
    for rid, want_state in expectations:
        e = _by_id(d, rid)
        if not e:
            failures.append(f"row {rid}: missing")
            continue
        ms = e.get("matches", [])
        if not ms:
            failures.append(f"row {rid}: no matches")
            continue
        st = _norm_loc(ms[0].get("state") or "")
        if st != want_state:
            failures.append(f"row {rid} top-1 state={st!r} want={want_state!r}")
    ok = not failures
    return ok, f"Gate D (wrong-state top-1): {'OK' if ok else '; '.join(failures)}"


def gate_e_dupes(d) -> tuple[bool, str]:
    n = 0
    for e in d:
        seen = set()
        for m in e.get("matches", []):
            key = (_norm_legal(m.get("name") or ""), _norm_loc(m.get("city") or ""), _norm_loc(m.get("state") or ""))
            if key in seen:
                n += 1
                break
            seen.add(key)
    ok = n == 0
    return ok, f"Gate E (same (name, city, state) duplicates in top-5): {n} (max 0)"


def gate_f_cliff_88(d, max_allowed: int = 100) -> tuple[bool, str]:
    n = sum(
        1 for e in d
        if e.get("matches") and abs(float(e["matches"][0].get("score") or 0) - 0.880) < 1e-3
    )
    ok = n <= max_allowed
    return ok, f"Gate F (top-1 clustered at 0.880): {n} (max {max_allowed})"


def info_low_conf(d) -> str:
    n = sum(1 for e in d if e.get("matches") and float(e["matches"][0].get("score") or 0) < 0.55)
    return f"INFO  low-confidence top-1 (< 0.55): {n}"


def summary_distribution(d) -> str:
    scores = [float(e["matches"][0].get("score") or 0) for e in d if e.get("matches")]
    scores.sort(reverse=True)
    n = len(scores) or 1

    def bucket(lo, hi):
        return sum(1 for s in scores if lo <= s < hi)

    parts = [
        f"records={n}",
        f">=0.95={bucket(0.95, 1.001)}",
        f"0.85-0.95={bucket(0.85, 0.95)}",
        f"0.70-0.85={bucket(0.70, 0.85)}",
        f"<0.70={bucket(0.0, 0.70)}",
    ]
    return "INFO  distribution: " + "  ".join(parts)


def main() -> int:
    print(f"audit input: {INPUT_PATH}")
    d = _load(INPUT_PATH)
    print(f"records loaded: {len(d)}")

    gates = [
        gate_score_sanity(d),
        gate_a_tied_clusters(d),
        gate_b_token_reorder(d),
        gate_c_acronym_junk(d),
        gate_d_wrong_state(d),
        gate_e_dupes(d),
        gate_f_cliff_88(d),
    ]

    failed = 0
    print("\n--- gates ---")
    for ok, msg in gates:
        prefix = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"{prefix}  {msg}")

    print("\n--- info ---")
    print(info_low_conf(d))
    print(summary_distribution(d))

    if failed:
        print(f"\nRESULT: FAIL ({failed} gate(s) failed)")
        return 1
    print("\nRESULT: PASS (all hard gates green)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
