"""Unit tests for CompanyMatcher._apply_lexical_floor (P0 #1: de-flatten tie mush).

The old Phase-2 logic clamped every below-floor candidate to a single flat value
(0.95 / 0.90), collapsing distinct candidates into identical scores and destroying
their true ordering ("tie mush"). The monotonic floor must:
  * never drop a qualifying candidate below the visibility floor,
  * preserve strict ordering among below-floor candidates (no more ties),
  * leave at/above-floor candidates completely unchanged,
  * bound inflation to < `band` (default 0.0099).

Pure-function test: no FAISS / RPC / cache needed (but importing matcher pulls in
sentence_transformers, so this is slower to start than a typical unit test).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finetuner.core.matcher import CompanyMatcher  # noqa: E402

f = CompanyMatcher._apply_lexical_floor
BAND = 0.0099


def test_floor_guarantee_and_bounded_inflation():
    below = [i / 100 for i in range(0, 95)]  # 0.00 .. 0.94
    for x in below:
        y = f(x, 0.95)
        assert y >= 0.95, f"floor violated: {x} -> {y}"
        assert y < 0.95 + BAND + 1e-9, f"inflation exceeds band: {x} -> {y}"


def test_strictly_monotonic_below_floor():
    below = [i / 100 for i in range(0, 95)]
    ys = [f(x, 0.95) for x in below]
    for i in range(len(ys) - 1):
        assert ys[i] < ys[i + 1], f"not strictly increasing at {below[i]}"


def test_at_or_above_floor_unchanged():
    assert f(0.95, 0.95) == 0.95
    assert f(0.97, 0.95) == 0.97
    assert f(0.99, 0.95) == 0.99
    assert f(1.0, 0.95) == 1.0


def test_previously_clamped_cluster_is_deflattened():
    # Prestige-style raw blends that all used to collapse to a flat 0.95
    cluster = [0.811, 0.886, 0.897]
    mapped = [f(c, 0.95) for c in cluster]
    assert len(set(mapped)) == 3, f"cluster still tied: {mapped}"
    assert mapped == sorted(mapped), f"cluster order not preserved: {mapped}"


def test_second_tier_floor_behaves_the_same():
    below = [i / 100 for i in range(0, 90)]
    ys = [f(x, 0.90) for x in below]
    assert all(y >= 0.90 for y in ys)
    assert all(y < 0.90 + BAND + 1e-9 for y in ys)
    assert all(ys[i] < ys[i + 1] for i in range(len(ys) - 1))
    assert f(0.92, 0.90) == 0.92  # above this floor, untouched


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
