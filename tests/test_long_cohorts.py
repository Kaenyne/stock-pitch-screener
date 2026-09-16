"""Long-side discovery cohorts (Step 3).

The defect these fix is structural, not a weighting problem: the short side
has 10+ cohort CSVs that surface an archetype regardless of composite rank,
while the long side had exactly ONE route (long_composite >= threshold). An
archetype the composite is not built to reward was therefore invisible rather
than low-ranked — 14 of the top 20 names by inflection_score reached no long
output file at all.

The property that must hold forever: these are ADDITIVE. They may never
change a score, a shortlist, or an existing cohort.
"""

import numpy as np
import pandas as pd
import pytest

from screener import config, shortlist


def _universe(n=40):
    """Core names with the full set of long-cohort inputs."""
    i = np.arange(n)
    return pd.DataFrame({
        "ticker": [f"T{k:02d}" for k in i],
        "sic": 2000 + (i % 5),
        "sector2": 20 + (i % 5),
        "is_financial": False,
        "zero_low_revenue": False,
        "is_commodity": False,
        "long_composite": 40.0 + (i % 11),
        "short_composite": 30.0 + (i % 7),
        "m_roic_spread": np.linspace(0, 100, n),
        "moat_demonstrated": np.linspace(0.01, 0.30, n),
        "moat_trailing_5y": np.linspace(0.01, 0.30, n),
        "spread_now": np.linspace(-0.10, 0.20, n),
        "m_valuation": np.linspace(0, 100, n),
        "drawdown": np.linspace(0.0, 0.9, n),
        "f_score": (i % 10).astype(float),
        "surv_traffic_light": ["green"] * n,
        "inflection_score": np.linspace(0.05, 0.58, n),
    })


class TestAdditiveOnly:
    def test_no_existing_output_changes(self):
        """The whole safety argument: adding these cannot move anything."""
        df = _universe()
        base = {k: v.copy() for k, v in shortlist.rank_outputs(df).items()
                if k not in config.LONG_COHORTS}
        saved = dict(config.LONG_COHORTS)
        try:
            config.LONG_COHORTS = {}
            without = shortlist.rank_outputs(df)
        finally:
            config.LONG_COHORTS = saved
        for k, v in base.items():
            assert k in without, f"{k} vanished"
            assert v["ticker"].tolist() == without[k]["ticker"].tolist(), \
                f"long cohorts perturbed existing output {k}"

    def test_cohorts_are_emitted(self):
        out = shortlist.rank_outputs(_universe())
        assert "derated_compounder" in out
        assert "inflecting_thin_moat" in out
        assert "surviving_distressed_value" in out

    def test_no_score_column_written(self):
        df = _universe()
        before = set(df.columns)
        shortlist.long_cohorts(df)
        assert set(df.columns) == before, "long_cohorts mutated the input frame"


class TestGating:
    def test_financials_and_zero_rev_excluded(self):
        df = _universe()
        df.loc[0, "is_financial"] = True
        df.loc[1, "zero_low_revenue"] = True
        out = shortlist.long_cohorts(df)
        for name, frame in out.items():
            assert "T00" not in set(frame["ticker"]), f"{name} kept a financial"
            assert "T01" not in set(frame["ticker"]), f"{name} kept a zero-rev"

    def test_derated_compounder_needs_moat_and_derating(self):
        df = _universe()
        out = shortlist.long_cohorts(df)["derated_compounder"]
        c = config.LONG_COHORTS["derated_compounder"]
        assert (out["m_roic_spread"] >= c["moat_rank_min"]).all()
        derated = (out["drawdown"] >= c["drawdown_min"]) | \
                  (out["m_valuation"] >= c["valuation_rank_min"])
        assert derated.all()

    def test_distressed_requires_every_survivability_leg(self):
        """Without these legs this cohort is a falling-knife screen — the
        adverse literature endorses the shape ONLY with solvency attached."""
        df = _universe()
        out = shortlist.long_cohorts(df).get("surviving_distressed_value")
        c = config.LONG_COHORTS["surviving_distressed_value"]
        assert out is not None and len(out)
        assert (out["surv_traffic_light"] == "green").all()
        assert (out["f_score"] >= c["f_score_min"]).all()
        assert out["drawdown"].between(c["drawdown_lo"], c["drawdown_hi"]).all()

    def test_red_light_never_reaches_inflecting_cohort(self):
        df = _universe()
        df["surv_traffic_light"] = "red"
        out = shortlist.long_cohorts(df)
        assert "inflecting_thin_moat" not in out or \
            not len(out["inflecting_thin_moat"])

    def test_inflecting_cohort_has_no_moat_gate(self):
        """By design: this is the archetype the moat-first composite cannot
        reward, so gating it on moat would reintroduce the defect."""
        df = _universe()
        df["m_roic_spread"] = 0.0          # no moat anywhere
        out = shortlist.long_cohorts(df)
        assert len(out["inflecting_thin_moat"]) > 0


class TestRobustness:
    def test_missing_columns_do_not_raise(self):
        df = _universe().drop(columns=["moat_trailing_5y", "f_score",
                                       "surv_traffic_light"])
        out = shortlist.long_cohorts(df)          # must not raise
        assert "surviving_distressed_value" not in out  # its legs are gone

    def test_nan_never_counts_as_passing(self):
        df = _universe()
        df["m_roic_spread"] = np.nan
        assert "derated_compounder" not in shortlist.long_cohorts(df)

    def test_negative_demonstrated_spread_is_not_retention(self):
        """trailing/demonstrated with both negative would read as a retained
        moat; it must not qualify on that leg."""
        df = _universe()
        df["moat_demonstrated"] = -0.20
        df["moat_trailing_5y"] = -0.15     # ratio 0.75, but meaningless
        df["spread_now"] = -0.05           # and current economics negative
        out = shortlist.long_cohorts(df)
        assert "derated_compounder" not in out or \
            not len(out["derated_compounder"])

    def test_cap_is_recorded_not_silent(self):
        df = _universe(400)
        out = shortlist.long_cohorts(df)["inflecting_thin_moat"]
        assert len(out) == config.LONG_COHORT_CAP
        assert out["n_qualified"].iloc[0] > config.LONG_COHORT_CAP
        assert out["capped_at"].iloc[0] == config.LONG_COHORT_CAP

    def test_ranking_is_order_independent(self):
        df = _universe()
        a = shortlist.long_cohorts(df)
        b = shortlist.long_cohorts(df.iloc[::-1].reset_index(drop=True))
        for k in a:
            assert a[k]["ticker"].tolist() == b[k]["ticker"].tolist()
