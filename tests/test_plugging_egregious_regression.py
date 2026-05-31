"""
RPC regression for manufactured egregious plugging cases.

See docs/CLAUDE_PLUGGING_CLOSED_LOOP.md Phase B.

Run (requires RPC + loaded cache):
  set RUN_PLUGGING_REGRESSION=1
  python -m pytest tests/test_plugging_egregious_regression.py -q
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "plugging_egregious_cases.json")

sys.path.insert(0, os.path.join(ROOT, "src"))


def _load_cases():
    if not os.path.isfile(FIXTURE):
        return []
    with open(FIXTURE, encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("cases") or []


def _rpc_available():
    if os.environ.get("RUN_PLUGGING_REGRESSION", "").strip() != "1":
        return False
    try:
        from finetuner.core.cache_rpc import is_server_running

        return bool(is_server_running())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _rpc_available(),
    reason="Set RUN_PLUGGING_REGRESSION=1 and start RPC cache server with cache loaded",
)


@pytest.fixture(scope="module")
def rpc():
    from finetuner.core.cache_rpc import connect

    s = connect()
    s._pyroTimeout = float(os.environ.get("PLUGGING_RPC_TIMEOUT", "120.0"))
    return s


def _bare(m):
    return not (str(m.get("city") or "").strip() or str(m.get("state") or "").strip())


def _norm_name(n):
    """Normalize a company name for same-entity comparison (case/punctuation-insensitive)."""
    import re as _re

    return _re.sub(r"[^a-z0-9]+", " ", str(n or "").lower()).strip()


def test_egregious_bundle_smoke(rpc):
    """Smoke over exported cases; tighten per issue_code as matcher guarantees land."""
    cases = _load_cases()[:80]
    if not cases:
        pytest.skip(f"No cases in {FIXTURE}; run export_plugging_egregious_cases.py")

    for case in cases:
        q = case.get("query_company") or ""
        assert q, case
        city = case.get("query_city") or None
        state = case.get("query_state") or None
        if isinstance(city, str) and not city.strip():
            city = None
        if isinstance(state, str) and not state.strip():
            state = None
        out = rpc.search(q, top_k=5, city=city, state=state)
        matches = out.get("results") or []
        assert len(matches) >= 1, case.get("row_id")
        codes = case.get("issue_codes") or ""
        if "NO_QUERY_GEO_BARE_ROW_NOT_FIRST" in codes and not (city or state):
            # SME rule (per the issue narrative): prefer a bare/national row over a
            # geo-tagged row ONLY when scores essentially TIE. A near-exact named match
            # that merely happens to carry an address must still outrank an unrelated
            # bare row (e.g. "Wahupa Educational Service" @0.94 vs bare
            # "Educational Support Services" @0.76). So enforce bare-first only when a
            # bare candidate is within a tie band of rank-1.
            top_score = float(matches[0].get("score") or 0.0)
            top_norm = _norm_name(matches[0].get("name"))
            # Same entity (same normalized name) appearing as both a geo-tagged and a bare
            # row, tied on score: the bare/national row should lead when the query has no geo.
            # Different-named near-ties (e.g. three distinct wine marathons) are NOT forced.
            same_name_tied_bare = [
                m for m in matches
                if _bare(m)
                and _norm_name(m.get("name")) == top_norm
                and abs(float(m.get("score") or 0.0) - top_score) <= 0.005
            ]
            if same_name_tied_bare and not _bare(matches[0]):
                raise AssertionError(
                    f"row {case.get('row_id')}: same-name bare row ties rank-1 ({top_score:.4f}) "
                    f"but rank-1 has geo city={matches[0].get('city')!r} "
                    f"state={matches[0].get('state')!r}; bare row should win the tie"
                )
