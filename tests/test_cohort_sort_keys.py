"""Cohort sort keys must be able to ORDER their cohort.

A discovery cohort is only as useful as its ordering — the reader works down
the list. A systematic check of all six archetype sort keys found four fine
and two that could not rank at all:

- `capacity_score_raw` = clip(-organic / organic_full, 0, 1): a HARD CLIP, so
  22 of 80 members (28%) sat at exactly 1.0 and the cohort's worst offenders
  were unorderable. The clip also HID an accounting artifact at the top (RJET,
  net PP&E +6,558pp vs revenue, ma_flag False) by lumping it with the ties.
- `ppi_zscore`: comes from the 2-digit-SIC crosswalk, so it is a SECTOR
  constant. All 61 members shared TWO distinct values (52 in SIC 13, 9 in SIC
  29) — the list was ordered only by the tie-break.

Both fixes are sort keys only; the tags carry no composite weight.
"""

import numpy as np
import pandas as pd

from screener import config, scoring


def _frame(n=12):
    return pd.DataFrame({
        "ticker": [f"T{i:02d}" for i in range(n)],
        "capacity_decay": True,
        "capacity_organic_proxy": np.linspace(-0.05, -0.90, n),
        "ppi_windfall": True,
        "ppi_zscore": 1.25,                     # identical: a SECTOR constant
        "rev_yoy_latest": np.linspace(0.02, 0.60, n),
        "real_rev_growth": np.linspace(0.01, 0.10, n),
    })


def _apply(df):
    """Run just the sort-key block's logic."""
    organic = pd.to_numeric(df["capacity_organic_proxy"], errors="coerce")
    implausible = organic < -config.CAPACITY_ORGANIC_IMPLAUSIBLE
    df = df.copy()
    df.loc[implausible, "capacity_decay"] = False
    df["capacity_decay_strength"] = np.where(
        df["capacity_decay"].fillna(False).astype(bool), -organic, np.nan)
    price_driven = (pd.to_numeric(df["rev_yoy_latest"], errors="coerce")
                    - pd.to_numeric(df["real_rev_growth"], errors="coerce"))
    df["ppi_windfall_strength"] = np.where(
        df["ppi_windfall"].fillna(False).astype(bool),
        pd.to_numeric(df["ppi_zscore"], errors="coerce")
        * price_driven.clip(lower=0.0), np.nan)
    return df


class TestCapacityStrength:
    def test_orders_past_the_old_clip_point(self):
        """Everything beyond organic_full used to collapse onto 1.0."""
        full = config.CAPACITY_DECAY["organic_full"]
        df = _apply(_frame())
        beyond = df[df["capacity_decay_strength"] > full]
        assert len(beyond) > 1
        assert beyond["capacity_decay_strength"].nunique() == len(beyond)

    def test_implausible_proxy_nulls_the_tag(self):
        df = _frame()
        df.loc[0, "capacity_organic_proxy"] = -65.58      # the RJET reading
        out = _apply(df)
        assert bool(out.loc[0, "capacity_decay"]) is False
        assert np.isnan(out.loc[0, "capacity_decay_strength"])

    def test_plausible_extreme_is_kept(self):
        df = _frame()
        df.loc[0, "capacity_organic_proxy"] = -(config.CAPACITY_ORGANIC_IMPLAUSIBLE - 0.1)
        out = _apply(df)
        assert bool(out.loc[0, "capacity_decay"]) is True
        assert out.loc[0, "capacity_decay_strength"] > 0

    def test_non_members_get_nan(self):
        df = _frame()
        df.loc[1, "capacity_decay"] = False
        assert np.isnan(_apply(df).loc[1, "capacity_decay_strength"])


class TestPpiWindfallStrength:
    def test_discriminates_within_one_sector(self):
        """The whole defect: every member of a sector shares one z-score, so
        the key must come from company-level data."""
        out = _apply(_frame())
        assert out["ppi_zscore"].nunique() == 1          # sector constant
        s = out["ppi_windfall_strength"]
        assert s.nunique() == len(out), "still cannot order within a sector"

    def test_ranks_by_price_driven_growth(self):
        """More of the growth explained by the price move = stronger windfall."""
        out = _apply(_frame())
        pd_growth = out["rev_yoy_latest"] - out["real_rev_growth"]
        assert out["ppi_windfall_strength"].corr(pd_growth) > 0.99

    def test_volume_growth_is_not_a_windfall(self):
        """Nominal == real means no price contribution at all."""
        df = _frame()
        df["real_rev_growth"] = df["rev_yoy_latest"]
        assert (_apply(df)["ppi_windfall_strength"] == 0).all()

    def test_negative_price_contribution_floors_at_zero(self):
        df = _frame()
        df["real_rev_growth"] = df["rev_yoy_latest"] + 0.2   # deflator helped
        assert (_apply(df)["ppi_windfall_strength"] == 0).all()

    def test_non_members_get_nan(self):
        df = _frame()
        df.loc[2, "ppi_windfall"] = False
        assert np.isnan(_apply(df).loc[2, "ppi_windfall_strength"])


class TestEndToEnd:
    def test_score_universe_emits_both_keys(self):
        import importlib
        m = importlib.import_module("tests.test_scoring")
        df = m.synthetic_universe()
        out = scoring.score_universe(df)
        assert "capacity_decay_strength" in out.columns
        assert "ppi_windfall_strength" in out.columns
