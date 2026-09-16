"""Forensic-signal unit tests (synthetic CompanyData shapes)."""

import numpy as np
import pandas as pd

from screener.forensic import (capacity_turnover_decay, external_price_signal,
                               nonop_earnings_share, tax_tailwind)
from screener.fundamentals import CompanyData


def _monthly(vals, end="2026-05-01"):
    idx = pd.date_range(end=end, periods=len(vals), freq="MS")
    return pd.Series(vals, index=idx, dtype=float)

from tests.conftest import qdates


def _cd(pretax, operating, interest):
    idx = qdates(len(pretax))
    return CompanyData(cik=1, ticker="T", ttm={
        "pretax_income": pd.Series(pretax, index=idx, dtype=float),
        "operating_income": pd.Series(operating, index=idx, dtype=float),
        "interest_expense": pd.Series(interest, index=idx, dtype=float),
    })


def _cd_cap(rev, ppe, assets_val):
    idx = qdates(len(rev))
    return CompanyData(cik=1, ticker="T",
        ttm={"revenue": pd.Series(rev, index=idx, dtype=float)},
        inst={"ppe_net": pd.Series(ppe, index=idx, dtype=float),
              "assets": pd.Series([assets_val] * len(rev), index=idx,
                                  dtype=float)})


def _cd_tax(pretax, tax, ni):
    idx = qdates(len(pretax))
    return CompanyData(cik=1, ticker="T", ttm={
        "pretax_income": pd.Series(pretax, index=idx, dtype=float),
        "tax_expense": pd.Series(tax, index=idx, dtype=float),
        "net_income": pd.Series(ni, index=idx, dtype=float)})


class TestExternalPriceSignal:
    ASOF = pd.Timestamp("2026-06-01")

    def test_windfall_elevated_and_rolling_over_fires(self):
        v = [100.0] * 48 + list(np.linspace(100, 150, 7)) + \
            list(np.linspace(150, 130, 5))            # spike to 150, roll to 130
        cd = CompanyData(cik=1, ticker="T", ttm={})
        out = external_price_signal(cd, _monthly(v), self.ASOF)
        assert out["ppi_zscore"] >= 1.0
        assert out["ppi_windfall"] is True

    def test_at_peak_not_rolling_over_no_fire(self):
        v = [100.0] * 48 + list(np.linspace(100, 150, 12))   # ends at the peak
        out = external_price_signal(CompanyData(cik=1, ticker="T", ttm={}),
                                    _monthly(v), self.ASOF)
        assert out["ppi_windfall"] is False

    def test_flat_series_no_windfall(self):
        out = external_price_signal(CompanyData(cik=1, ticker="T", ttm={}),
                                    _monthly([100.0] * 60), self.ASOF)
        assert out["ppi_windfall"] is False

    def test_real_rev_negative_when_price_outpaces_revenue(self):
        ppi = _monthly([100.0] * 47 + list(np.linspace(100, 120, 13)))  # +20% YoY
        rev = pd.Series([98, 99, 100, 100, 101, 102, 104, 105],
                        index=qdates(8), dtype=float)                    # +5% YoY
        cd = CompanyData(cik=1, ticker="T", ttm={"revenue": rev})
        out = external_price_signal(cd, ppi, self.ASOF)
        assert out["real_rev_growth"] < 0
        assert out["real_rev_deflated"] is True


class TestTaxTailwind:
    def test_low_current_rate_fires(self):
        pretax = [100] * 9
        tax = [25] * 8 + [5]                     # median 25%, latest 5%
        ni = [p - t for p, t in zip(pretax, tax)]
        out = tax_tailwind(_cd_tax(pretax, tax, ni))
        assert out["eff_rate"] < 0.10 and out["median_rate"] > 0.20
        assert out["boost_share"] > 0.05 and out["flag"] is True

    def test_stable_rate_no_fire(self):
        out = tax_tailwind(_cd_tax([100] * 9, [25] * 9, [75] * 9))
        assert out["flag"] is False
        assert out["boost_share"] == 0.0 or out["boost_share"] < 0.01


class TestCapacityTurnoverDecay:
    def test_capacity_driven_growth_fires(self):
        rev = [80, 85, 90, 100, 105, 108, 112, 115]   # +15% YoY (-5 vs -1)
        ppe = [80, 85, 90, 100, 110, 115, 120, 125]   # +25% YoY (faster)
        out = capacity_turnover_decay(_cd_cap(rev, ppe, 300.0))
        assert out["organic_proxy"] < 0                # rev grew slower than capacity
        assert out["flag"] is True
        assert out["score_raw"] > 0

    def test_organic_demand_growth_does_not_fire(self):
        rev = [80, 85, 90, 100, 110, 118, 124, 125]   # +25%
        ppe = [95, 97, 99, 100, 104, 106, 107, 108]   # +8% (slower)
        out = capacity_turnover_decay(_cd_cap(rev, ppe, 300.0))
        assert out["organic_proxy"] > 0
        assert out["flag"] is False

    def test_asset_light_gated_out(self):
        rev = [80, 85, 90, 100, 105, 108, 112, 115]
        ppe = [80, 85, 90, 100, 110, 115, 120, 125]
        out = capacity_turnover_decay(_cd_cap(rev, ppe, 10000.0))  # PP&E ~1% of assets
        assert np.isnan(out["score_raw"])


class TestNonopEarningsShare:
    def test_high_and_rising_share_fires(self):
        # operating margin of pretax shrinks -> below-line income share rises
        operating = [95, 95, 95, 90, 85, 80, 75, 72, 70]
        cd = _cd([100] * 9, operating, [10] * 9)
        out = nonop_earnings_share(cd)
        # latest: (100-70+10)/100 = 0.40
        assert abs(out["share"] - 0.40) < 1e-9
        assert out["share_yoy_delta"] > 0.04
        assert out["score_raw"] > 0.9      # level kernel saturates
        assert out["flag"] is True

    def test_degeneracy_operating_equals_pretax(self):
        """The operating_income ladder falls back to the pretax tag: bridge
        collapses to interest expense -> must return NaN, not a fire."""
        cd = _cd([100] * 9, [100] * 9, [10] * 9)
        out = nonop_earnings_share(cd)
        assert out["degenerate"] is True
        assert np.isnan(out["score_raw"])
        assert "nonop_oi_is_pretax" in cd.flags

    def test_falling_share_scores_zero(self):
        # below-line share declining (improving quality) -> not a short signal
        operating = [70, 72, 75, 80, 85, 90, 95, 95, 95]
        cd = _cd([100] * 9, operating, [10] * 9)
        out = nonop_earnings_share(cd)
        assert out["share_yoy_delta"] < 0
        assert out["score_raw"] == 0.0
        assert out["flag"] is False

    def test_negative_pretax_no_score(self):
        cd = _cd([-50] * 9, [-60] * 9, [10] * 9)
        out = nonop_earnings_share(cd)
        assert np.isnan(out["score_raw"]) or out["score_raw"] == 0.0
