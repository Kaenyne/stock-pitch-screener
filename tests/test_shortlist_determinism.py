"""Cohort ranking must be reproducible.

Several strength columns saturate — `capacity_score_raw` pinned 26 of 83
names at exactly 1.0 in the 2025-10-15 replay — so a single-key sort left the
order inside the tied block decided by input row order. Re-scoring identical
data reshuffled the cohort (VITL moved rank 25 -> 8), which makes the
exemplar before/after diff unreadable: a rank move inside a tied block is
noise that looks exactly like signal.
"""

import pandas as pd

from screener import shortlist


def _tied(n=12):
    """Half the rows tied at the cohort max, distinct composites."""
    return pd.DataFrame({
        "ticker": [f"T{i:02d}" for i in range(n)],
        "capacity_score_raw": [1.0] * (n // 2) + [0.5] * (n - n // 2),
        "short_composite": [50.0 + i for i in range(n)],
        "n_thesis_tags": [2] * n,
    })


class TestStableSort:
    def test_ties_broken_by_composite_then_ticker(self):
        df = _tied()
        out = shortlist._stable_sort(df, ["capacity_score_raw"])
        top = out.head(6)
        assert (top["capacity_score_raw"] == 1.0).all()
        # inside the tied block, highest composite first
        assert list(top["short_composite"]) == sorted(
            top["short_composite"], reverse=True)

    def test_row_order_does_not_change_the_ranking(self):
        """The exact failure observed: same data, different input order."""
        df = _tied()
        a = shortlist._stable_sort(df, ["capacity_score_raw"])["ticker"].tolist()
        shuffled = df.iloc[::-1].reset_index(drop=True)
        b = shortlist._stable_sort(shuffled, ["capacity_score_raw"])["ticker"].tolist()
        assert a == b

    def test_fully_tied_falls_through_to_ticker(self):
        df = pd.DataFrame({
            "ticker": ["ZZZ", "AAA", "MMM"],
            "capacity_score_raw": [1.0, 1.0, 1.0],
            "short_composite": [60.0, 60.0, 60.0],
        })
        out = shortlist._stable_sort(df, ["capacity_score_raw"])
        assert out["ticker"].tolist() == ["AAA", "MMM", "ZZZ"]

    def test_does_not_duplicate_an_explicit_key(self):
        df = _tied()
        out = shortlist._stable_sort(df, ["short_composite"])
        assert out["short_composite"].tolist() == sorted(
            df["short_composite"], reverse=True)

    def test_missing_tiebreak_columns_tolerated(self):
        df = pd.DataFrame({"capacity_score_raw": [0.2, 0.9, 0.5]})
        out = shortlist._stable_sort(df, ["capacity_score_raw"])
        assert out["capacity_score_raw"].tolist() == [0.9, 0.5, 0.2]


class TestCohortsAreReproducible:
    def _universe(self, n=40):
        return pd.DataFrame({
            "ticker": [f"T{i:02d}" for i in range(n)],
            "sic": [2000 + (i % 3) for i in range(n)],
            "sector2": [20 + (i % 3) for i in range(n)],
            "long_composite": [40.0 + (i % 7) for i in range(n)],
            "short_composite": [40.0 + (i % 5) for i in range(n)],
            "is_financial": False, "zero_low_revenue": False,
            "is_commodity": False,
            "capacity_decay": [i % 2 == 0 for i in range(n)],
            "capacity_score_raw": [1.0 if i % 4 == 0 else 0.4
                                   for i in range(n)],
            "story_short_score": [10.0] * n,
        })

    def test_same_data_two_orders_same_cohorts(self):
        df = self._universe()
        a = shortlist.rank_outputs(df)
        b = shortlist.rank_outputs(df.iloc[::-1].reset_index(drop=True))
        assert set(a) == set(b)
        for k in a:
            assert a[k]["ticker"].tolist() == b[k]["ticker"].tolist(), \
                f"cohort {k} is order-dependent"
