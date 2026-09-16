"""Step 1: the Q&A-fatal false positives.

Each of these fired on a tear sheet and would lose the question it invites:
- `margin_vs_peak` scored NRP full credit off a 104.95% "operating margin",
  and with `inflection_score` unmeasurable that 0.15-weight row carried the
  whole theme -> t_inflection 100, LONG RANK #7, inflection_breadth 0;
- `restated` fired on 1,495 of 2,138 names while 1,459 had no 8-K 4.02, and
  the checklist told the reader to go pull that non-existent filing;
- `pmv_score_raw` put a price-vs-volume checklist on lender tear sheets.
"""

import numpy as np
import pandas as pd
import pytest

from screener import config, forensic, inflection, scoring
from screener.edgar import SubmissionsInfo


def _margins(vals, end="2026-03-31"):
    idx = pd.date_range(end=end, periods=len(vals), freq="QE")
    return pd.Series(vals, index=idx)


ASOF = pd.Timestamp("2026-06-30")


class TestMarginVsPeakGuards:
    def test_normal_series_still_scores(self):
        s = _margins([0.20] * 8 + [0.18, 0.15, 0.10, 0.08])
        score, det = inflection.margin_vs_peak(s, ASOF)
        assert np.isfinite(score) and score > 0
        assert det["peak"] == pytest.approx(0.20)

    def test_implausible_peak_nulls_the_metric(self):
        """The NRP shape: a peak above 100% means the numerator and the
        denominator are not the same business."""
        s = _margins([1.0495] * 4 + [0.95] * 4 + [0.8366] * 4)
        score, det = inflection.margin_vs_peak(s, ASOF)
        assert np.isnan(score)
        assert det["skipped"] == "implausible_peak"

    def test_peak_exactly_at_bound_is_allowed(self):
        s = _margins([1.0] * 4 + [0.9] * 4 + [0.80] * 4)
        score, _ = inflection.margin_vs_peak(s, ASOF)
        assert np.isfinite(score)

    def test_stale_series_nulls_the_metric(self):
        """`cur` is just the last observation, so without this a series that
        stopped resolving years ago is scored as if it were current."""
        s = _margins([0.20] * 8 + [0.05] * 4, end="2022-12-31")
        score, det = inflection.margin_vs_peak(s, ASOF)
        assert np.isnan(score)
        assert det["skipped"] == "stale_series"
        assert det["age_days"] > config.MARGIN_VS_PEAK_STALE_DAYS

    def test_bank_negative_efficiency_ratio_unaffected(self):
        """Banks pass a NEGATED efficiency ratio, so only the upper bound
        binds — the guard must not null every bank."""
        s = _margins([-0.55] * 8 + [-0.62, -0.68, -0.70, -0.72])
        score, det = inflection.margin_vs_peak(s, ASOF)
        assert np.isfinite(score)
        assert "skipped" not in det

    def test_short_series_still_returns_nan(self):
        score, _ = inflection.margin_vs_peak(_margins([0.2, 0.1]), ASOF)
        assert np.isnan(score)


class TestThinBasisShrink:
    def _frame(self, n=40):
        """One NRP-shaped row (inflection unmeasurable, margin_vs_peak maxed)
        against a normal cohort."""
        rows = {
            "ticker": [f"T{i:02d}" for i in range(n)],
            "inflection_score": [0.3 + (i % 5) * 0.05 for i in range(n)],
            "margin_vs_peak": [0.2 + (i % 4) * 0.1 for i in range(n)],
        }
        df = pd.DataFrame(rows)
        df.loc[0, "ticker"] = "NRP"
        df.loc[0, "inflection_score"] = np.nan
        df.loc[0, "margin_vs_peak"] = 1.0
        return df

    def _t_inflection(self, df):
        df = df.copy()
        df["s_inflection"] = df["inflection_score"] * 100.0
        df["s_margin_vs_peak"] = df["margin_vs_peak"] * 100.0
        row, cov = scoring.weighted_theme(
            df, {"s_inflection": 0.85, "s_margin_vs_peak": 0.15})
        med = float(np.nanmedian(row.values))
        thin = cov < 1.0
        return row.where(~thin, cov * row + (1.0 - cov) * med), cov

    def test_solo_margin_row_no_longer_maxes_the_theme(self):
        df = self._frame()
        t, cov = self._t_inflection(df)
        assert cov.iloc[0] == pytest.approx(0.15)
        assert t.iloc[0] < 60, "the 0.15-weight row must not carry the theme"

    def test_full_coverage_rows_are_untouched(self):
        df = self._frame()
        t, cov = self._t_inflection(df)
        full = cov == 1.0
        assert full.sum() > 1
        raw = (0.85 * df["inflection_score"] * 100.0
               + 0.15 * df["margin_vs_peak"] * 100.0)
        pd.testing.assert_series_equal(t[full], raw[full], check_names=False)

    def test_shrink_is_toward_the_median_not_zero(self):
        df = self._frame()
        t, _ = self._t_inflection(df)
        assert t.iloc[0] > 15.0, "0.15 * 100 alone would be a hard haircut"


class _CD:
    """Minimal CompanyData stand-in — event_flags only reads `.flags`."""
    def __init__(self, flags):
        self.flags = set(flags)


def _sub(items=(), amended=(), nt=0):
    return SubmissionsInfo(
        cik=1,
        recent_8k_items=[("2026-01-05", i) for i in items],
        amended_periodic=list(amended),
        nt_filings_recent=nt)


class TestRestatedRedefinition:
    def test_value_move_alone_is_not_a_restatement(self):
        """The 1,459-name case: a >3% retained-value move with no amendment
        and no 4.02 behind it."""
        ev = forensic.event_flags(_CD({"restated"}), _sub())
        assert ev["restated"] is False
        assert ev["comparatives_represented"] is True
        assert ev["restatement_filing"] == ""

    def test_non_reliance_alone_is_a_restatement(self):
        ev = forensic.event_flags(_CD(set()), _sub(items=["4.02"]))
        assert ev["restated"] is True
        assert ev["restatement_filing"] == "8-K 4.02"

    def test_amendment_plus_value_move_is_a_restatement(self):
        ev = forensic.event_flags(
            _CD({"restated"}), _sub(amended=[("10-K/A", "2025-11-01", "a-1")]))
        assert ev["restated"] is True
        assert ev["restatement_filing"] == "10-K/A"

    def test_amendment_without_value_move_is_not(self):
        """A 10-K/A adding Part III or an exhibit changed no number."""
        ev = forensic.event_flags(
            _CD(set()), _sub(amended=[("10-K/A", "2025-11-01", "a-1")]))
        assert ev["restated"] is False

    def test_every_firing_name_names_a_filing_to_pull(self):
        """The whole point: the checklist must never send the reader after a
        document that does not exist."""
        for cd, sub in [(_CD({"restated"}), _sub(items=["4.02"])),
                        (_CD({"restated"}),
                         _sub(amended=[("10-Q/A", "2025-06-01", "b-1")])),
                        (_CD(set()), _sub(items=["4.02"]))]:
            ev = forensic.event_flags(cd, sub)
            assert ev["restated"] is True
            assert ev["restatement_filing"], "fired with no filing named"

    def test_no_submissions_info_is_never_a_restatement(self):
        """With no filing history there is no filing to pull, so the claim
        cannot be made — the value move is still reported on its own."""
        ev = forensic.event_flags(_CD({"restated"}), None)
        assert ev["restated"] is False
        assert ev["comparatives_represented"] is True
        assert ev["restatement_filing"] == ""


class TestFinancialsPmvNull:
    def test_pmv_nulled_for_financials(self):
        df = pd.DataFrame({
            "ticker": ["BANK", "RETAIL"],
            "is_financial": [True, False],
            "zero_low_revenue": [False, False],
            "pmv_score_raw": [0.9, 0.9],
            "pricing_masking_volume": [True, True],
            "ran_on_pricing_success": [True, True],
            "ma_flag": [False, False],
            "capacity_decay": [True, True],
        })
        is_fin = df["is_financial"]
        for c in ("pmv_score_raw", "pricing_masking_volume",
                  "ran_on_pricing_success"):
            df.loc[is_fin, c] = (False if df[c].dtype == bool else np.nan)
        assert np.isnan(df.loc[0, "pmv_score_raw"])
        assert df.loc[0, "pricing_masking_volume"] is False or \
            df.loc[0, "pricing_masking_volume"] == False  # noqa: E712
        assert df.loc[1, "pmv_score_raw"] == 0.9
        assert bool(df.loc[1, "pricing_masking_volume"]) is True
