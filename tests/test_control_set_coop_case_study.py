"""
Regression: control-set case study — same legal name, two geographies (see companies_control_set.json).
Runs without RPC; uses a minimal in-memory index.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finetuner.core.matcher import CompanyMatcher, _legal_name_match_key
from finetuner.utils.text_preprocessor import TextPreprocessor


COOP = "CU Cooperative Systems, Inc. dba CO-OP Solutions"

# Intentional cosmetic difference on second row (trailing space) — must still merge for multi-site ranking.
SYNTH = [
    {"Company Name": COOP, "City": "Rancho Cucamonga", "State": "CA", "Count": 1},
    {"Company Name": COOP + " ", "City": "Durham", "State": "CT", "Count": 99999},
]


def _coop_from_control_set():
    path = os.path.join(os.path.dirname(__file__), "..", "companies_control_set.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [x for x in data if x.get("Company Name") == COOP and x.get("control_match_state")]


def test_control_set_includes_coop_case_study():
    rows = _coop_from_control_set()
    assert len(rows) == 2, rows
    st = {TextPreprocessor.normalize_state(r["State"]) for r in rows}
    assert st == {TextPreprocessor.normalize_state("CA"), TextPreprocessor.normalize_state("CT")}


def test_coop_each_query_picks_matching_office():
    m = CompanyMatcher()
    m.build_index_with_location(data=SYNTH)

    r_ca = m.match_with_location(
        COOP, city="Rancho Cucamonga", state="CA", top_k=3
    )
    assert _legal_name_match_key(r_ca[0]["name"]) == _legal_name_match_key(COOP)
    assert TextPreprocessor.normalize_state(r_ca[0].get("state", "")) == TextPreprocessor.normalize_state("CA")
    assert "rancho" in (r_ca[0].get("city") or "").lower()

    r_ct = m.match_with_location(COOP, city="Durham", state="CT", top_k=3)
    assert _legal_name_match_key(r_ct[0]["name"]) == _legal_name_match_key(COOP)
    assert TextPreprocessor.normalize_state(r_ct[0].get("state", "")) == TextPreprocessor.normalize_state("CT")
    assert "durham" in (r_ct[0].get("city") or "").lower()


def run():
    test_control_set_includes_coop_case_study()
    test_coop_each_query_picks_matching_office()
    print("test_control_set_coop_case_study: OK")


if __name__ == "__main__":
    run()
