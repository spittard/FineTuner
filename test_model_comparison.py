"""
Model Comparison Test — SME Scenario Coverage
==============================================
Tests two models head-to-head using cosine similarity on pairs drawn from:
  - The live NWACUHO failure case (directional disambiguation)
  - SME report control set scenarios (109 queries across multiple categories)
  - Serge Nation showcase edge cases

No FAISS index or RPC server required — pure embedding similarity.

Usage:
    python test_model_comparison.py
    python test_model_comparison.py --model-a paraphrase-MiniLM-L3-v2 --model-b all-MiniLM-L6-v2
"""

import os, sys, argparse, json
# Keep offline mode — both models are locally cached; avoid proxy errors
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import sys
import numpy as np
from sentence_transformers import SentenceTransformer

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Test suite ─────────────────────────────────────────────────────────────────
# Format: (query, good_match, bad_match, category, description)
# Pass criterion: cosine(query, good) > cosine(query, bad)
CASES = [

    # ── DIRECTIONAL DISAMBIGUATION (the live failure) ────────────────────────
    (
        "NWACUHO - Northwest Association of College & University Housing Officers",
        "Northwest Association of College and University Housing Officers",
        "Southwest Association College University Housing Officers",
        "Directional", "NW vs SW — core failure case"
    ),
    (
        "Northeast Medical Group",
        "Northeast Medical Associates",
        "Southwest Medical Group",
        "Directional", "NE vs SW medical"
    ),
    (
        "North Texas Food Bank",
        "North Texas Community Foundation",
        "South Texas Food Bank",
        "Directional", "North vs South Texas"
    ),
    (
        "Pacific Northwest Diabetes Research Inst",
        "Pacific Northwest Diabetes Research Institute",
        "Pacific Southwest Diabetes Research Institute",
        "Directional", "Pacific NW vs SW diabetes institute (SME #52)"
    ),
    (
        "Southern Vermont Deerfield Valley Chamber of commerce",
        "Deerfield Valley Chamber of Commerce",
        "Northern Vermont Chamber of Commerce",
        "Directional", "Southern Vermont geo-specificity (SME #74)"
    ),
    (
        "North Shore Senior Center",
        "North Shore Senior Services",
        "South Shore Senior Center",
        "Directional", "North vs South Shore (SME #56)"
    ),
    (
        "Chicago South Swim Club",
        "Chicago South Shore Swim Club",
        "Chicago North Swim Club",
        "Directional", "South vs North Chicago swim (SME #59)"
    ),

    # ── ACRONYM EXPANSION ───────────────────────────────────────────────────
    (
        "NIH",
        "National Institutes of Health",
        "Nordic Institute of Health",
        "Acronym", "NIH → National Institutes of Health (SME #1)"
    ),
    (
        "IBM",
        "International Business Machines",
        "International Business Management",
        "Acronym", "IBM expansion (SME #108)"
    ),
    (
        "PDMA",
        "Product Development and Management Association",
        "Public Data Management Authority",
        "Acronym", "PDMA expansion (SME #6 / #107)"
    ),
    (
        "ABA",
        "American Bar Association",
        "Australian Banking Association",
        "Acronym", "ABA expansion (SME #106)"
    ),
    (
        "GE",
        "General Electric",
        "General Entertainment",
        "Acronym", "GE expansion (SME #109)"
    ),
    (
        "NFC Forum",
        "NFC Forum",
        "NFL Forum",
        "Acronym", "NFC Forum — acronym exact match (SME #27 / Serge #8)"
    ),

    # ── EXACT / NEAR-EXACT MATCH ────────────────────────────────────────────
    (
        "Ohio University",
        "Ohio University",
        "Ohio State University",
        "Exact", "Exact match beats similar (SME #2)"
    ),
    (
        "Next Level Events",
        "Next Level Events",
        "Next Level Site Selection & Events",
        "Exact", "Exact beats longer variant (SME #9 / Serge #1)"
    ),
    (
        "DermaQuest Inc",
        "DermaQuest Inc",
        "DermaCare Inc",
        "Exact", "Exact with suffix (SME #12 / Serge #2)"
    ),
    (
        "Ellwood Group Inc",
        "Ellwood Group Inc",
        "Elmwood Group Inc",
        "Exact", "1-char diff Ellwood vs Elmwood (SME #13)"
    ),

    # ── CORPORATE SUFFIX HANDLING ───────────────────────────────────────────
    (
        "DermaQuest",
        "DermaQuest Inc",
        "DermaCare",
        "Suffix", "Suffix omission still matches (Serge #2)"
    ),
    (
        "Amedysis",
        "Amedisys Incorporated",
        "Amedicare Inc",
        "Suffix", "Typo + suffix drop (SME #64)"
    ),
    (
        "Acacia Pharma Group",
        "Acacia Pharma Group Inc.",
        "Acacia Healthcare Group",
        "Suffix", "Suffix stripped (SME #81)"
    ),

    # ── PARTIAL / ABBREVIATION ──────────────────────────────────────────────
    (
        "Western University",
        "Western University",
        "Eastern University",
        "Partial", "Western vs Eastern university (SME #3)"
    ),
    (
        "JP Morgan Chase",
        "JPMorgan Chase & Co",
        "Morgan Stanley",
        "Partial", "Token reorder (SME-style)"
    ),
    (
        "Kruger Products",
        "Kruger Products Inc",
        "Kroger Products",
        "Partial", "Kruger vs Kroger 1-char (SME #4)"
    ),
    (
        "Mitsubishi Motor Sales of America, Incorporated",
        "Mitsubishi Motors North America Inc",
        "Mitsubishi Electric America Inc",
        "Partial", "Mitsubishi Motors vs Electric (SME #98 / Serge #4)"
    ),

    # ── SEMANTIC / CONTEXTUAL ───────────────────────────────────────────────
    (
        "Hartford Hospital School of Nursing",
        "Hartford Hospital School of Allied Health",
        "Hartford Community College School of Nursing",
        "Semantic", "Hospital school of nursing vs allied health (SME #18 / Serge #3)"
    ),
    (
        "Seafood Nutrition Partnership",
        "National Fisheries Institute",
        "Organic Food Partnership",
        "Semantic", "Seafood/nutrition domain (SME #16)"
    ),
    (
        "Institute of Health Technology Transformation",
        "Health Technology Assessment International",
        "Institute of Financial Technology",
        "Semantic", "Health tech institute (SME #90)"
    ),
    (
        "Spina Bifida Coalition of Cincinnati",
        "Spina Bifida Association of America",
        "Cincinnati Children's Hospital Coalition",
        "Semantic", "Medical condition org (SME #20)"
    ),
    (
        "Global Interagency Security Forum",
        "Global Security Forum",
        "International Business Forum",
        "Semantic", "Security forum (SME #31)"
    ),

    # ── INTERNATIONAL / FOREIGN LANGUAGE ───────────────────────────────────
    (
        "AVIAKOMPANIYA SIBIR, PAO",
        "S7 Airlines",
        "Aeroflot Russian Airlines",
        "International", "Sibir Airlines Russian name (SME #17)"
    ),
    (
        "Volkswagen Group China",
        "Volkswagen Group of America",
        "BMW Group China",
        "International", "VW China vs America (SME #72 / Serge #4-adjacent)"
    ),
    (
        "Telefonica Global Solutions",
        "Telefonica S.A.",
        "T-Mobile Global Solutions",
        "International", "Telefonica brand (SME #69)"
    ),

    # ── NOISE / JUNK QUERIES ────────────────────────────────────────────────
    (
        "Nicolas/Sanchez Wedding",
        "Wedding Planning Services",
        "Nicolas & Sanchez Law Firm",
        "Noise", "Personal event query (SME #7 / Serge #10)"
    ),
    (
        "X DO NOT USE - FRANCIS PARKER SCHOOL",
        "Francis Parker School",
        "Parker School",
        "Noise", "Prefixed junk — core name still extractable (SME #97 / Serge #11)"
    ),
    (
        "NaLA 2024 fall conference M01709226216947 02-29-24 12:03:46",
        "National Association of Landscape Architects",
        "National Library Association",
        "Noise", "Conference ID garbage in query (SME #54)"
    ),
    (
        "Internal J&J Meeting and Breakfast",
        "Johnson & Johnson",
        "Internal Revenue Service",
        "Noise", "Internal meeting label (SME #19)"
    ),
    (
        "Donnelley Work Session",
        "RR Donnelley",
        "Donnelley Financial Solutions",
        "Noise", "Internal session label (SME #55)"
    ),

    # ── PERSON / INDIVIDUAL NAMES ───────────────────────────────────────────
    (
        "Denise Roberge",
        "Denise Roberge Ltd",
        "Denise Richards Enterprises",
        "Person", "Person name with company variant (SME #25)"
    ),
    (
        "Stephen Rourke",
        "Stephen Rourke Consulting",
        "Rourke & Associates",
        "Person", "Individual consultant name (SME #43)"
    ),

    # ── GEOGRAPHIC / LOCATION-SPECIFIC ─────────────────────────────────────
    (
        "City of Dallas-Parks & Recreation",
        "City of Dallas Parks and Recreation Department",
        "Dallas County Parks & Recreation",
        "Geographic", "Municipal department (SME #47)"
    ),
    (
        "Louisiana State University Swim",
        "Louisiana State University",
        "University of Louisiana Swimming",
        "Geographic", "LSU swim program (SME #96)"
    ),
    (
        "Boys and Girls Club of Dawson Community Centre",
        "Boys & Girls Club of America",
        "Dawson Community College",
        "Geographic", "B&G Club with community centre suffix (SME #61)"
    ),

    # ── UNUSUAL / EDGE CASES ────────────────────────────────────────────────
    (
        "1960",
        "Class of 1960 Reunion",
        "1960 Media Group",
        "Edge", "Numeric-only query (SME #51)"
    ),
    (
        "Edna, Dabra@SAP.IO",
        "SAP SE",
        "SAP Labs India",
        "Edge", "Email address as query (SME #60)"
    ),
    (
        "Interim WG Meeting - BIER",
        "IETF BIER Working Group",
        "Beer Industry Working Group",
        "Edge", "Technical WG acronym (SME #11)"
    ),
    (
        "Linklaters CIS",
        "Linklaters LLP",
        "CIS Consulting Group",
        "Edge", "Law firm with CIS suffix (SME #83)"
    ),
]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def evaluate(model_name: str, cases: list) -> dict:
    print(f"\n{'-'*68}")
    print(f"  Model: {model_name}")
    print(f"{'-'*68}")
    model = SentenceTransformer(model_name)

    # Collect all unique strings for batch encoding
    strings = []
    for query, good, bad, cat, desc in cases:
        strings += [query, good, bad]
    unique = list(dict.fromkeys(strings))
    print(f"  Encoding {len(unique)} unique strings...", end="", flush=True)
    vecs = model.encode(unique, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    vec_map = {s: vecs[i] for i, s in enumerate(unique)}
    print(" done.")

    by_category = {}
    results = []
    for query, good, bad, cat, desc in cases:
        sim_good = cosine(vec_map[query], vec_map[good])
        sim_bad  = cosine(vec_map[query], vec_map[bad])
        correct  = sim_good > sim_bad
        gap      = sim_good - sim_bad

        by_category.setdefault(cat, []).append(correct)
        results.append({
            'desc': desc, 'category': cat, 'correct': correct,
            'sim_good': sim_good, 'sim_bad': sim_bad, 'gap': gap,
            'query': query, 'good': good, 'bad': bad
        })
        status = "PASS" if correct else "FAIL"
        print(f"  [{status}] [{cat:14s}] {desc}")
        if not correct:
            print(f"           good='{good[:50]}' sim={sim_good:.4f}")
            print(f"           bad= '{bad[:50]}' sim={sim_bad:.4f}  gap={gap:+.4f}")

    total = len(cases)
    wins  = sum(r['correct'] for r in results)
    print(f"\n  Overall: {wins}/{total} ({100*wins/total:.0f}%)")
    print(f"\n  By category:")
    for cat, vals in sorted(by_category.items()):
        w = sum(vals)
        print(f"    {cat:18s}  {w}/{len(vals)}")

    return {'model': model_name, 'wins': wins, 'total': total,
            'by_category': {c: {'wins': sum(v), 'total': len(v)} for c, v in by_category.items()},
            'results': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-a', default='paraphrase-MiniLM-L3-v2')
    parser.add_argument('--model-b', default='all-MiniLM-L6-v2')
    args = parser.parse_args()

    print("\n" + "=" * 68)
    print("  Model Comparison — SME Scenario Coverage")
    print(f"  {len(CASES)} test cases across {len(set(c[3] for c in CASES))} categories")
    print("=" * 68)

    result_a = evaluate(args.model_a, CASES)
    result_b = evaluate(args.model_b, CASES)

    print(f"\n{'='*68}")
    print(f"  FINAL SUMMARY")
    print("="*68)
    print(f"  {args.model_a:50s}  {result_a['wins']}/{result_a['total']}  ({100*result_a['wins']/result_a['total']:.0f}%)")
    print(f"  {args.model_b:50s}  {result_b['wins']}/{result_b['total']}  ({100*result_b['wins']/result_b['total']:.0f}%)")

    print(f"\n  Category breakdown:")
    all_cats = sorted(set(list(result_a['by_category']) + list(result_b['by_category'])))
    print(f"  {'Category':18s}  {'A':>6}  {'B':>6}  Delta")
    for cat in all_cats:
        a = result_a['by_category'].get(cat, {'wins': 0, 'total': 0})
        b = result_b['by_category'].get(cat, {'wins': 0, 'total': 0})
        pct_a = 100 * a['wins'] / a['total'] if a['total'] else 0
        pct_b = 100 * b['wins'] / b['total'] if b['total'] else 0
        delta = pct_b - pct_a
        flag = " ^" if delta > 0 else (" v" if delta < 0 else "")
        print(f"  {cat:18s}  {a['wins']}/{a['total']:>2}({pct_a:3.0f}%)  {b['wins']}/{b['total']:>2}({pct_b:3.0f}%)  {delta:+.0f}%{flag}")

    improvements = [r for ra, rb in zip(result_a['results'], result_b['results'])
                    for r in [rb] if not ra['correct'] and rb['correct']]
    regressions  = [r for ra, rb in zip(result_a['results'], result_b['results'])
                    for r in [rb] if ra['correct'] and not rb['correct']]

    if improvements:
        print(f"\n  New passes with {args.model_b}:")
        for r in improvements:
            print(f"    + [{r['category']}] {r['desc']}")
    if regressions:
        print(f"\n  Regressions with {args.model_b}:")
        for r in regressions:
            print(f"    - [{r['category']}] {r['desc']}")

    with open('model_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump({'model_a': result_a, 'model_b': result_b}, f, indent=2)
    print(f"\n  Full results → model_comparison_results.json")

    winner = args.model_b if result_b['wins'] >= result_a['wins'] else args.model_a
    print(f"  Recommended:   {winner}\n")


if __name__ == '__main__':
    main()
