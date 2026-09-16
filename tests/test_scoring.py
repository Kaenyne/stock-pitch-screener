"""Scoring + shortlist tests: missing-data policy, kernels, flags, and a
miniature end-to-end 'does it rank the archetypes right' check."""

import numpy as np
import pandas as pd
import pytest

from screener import config
from screener.scoring import (compression_kernel, drawdown_kernel, pct_rank,
                              peer_median, runup_kernel, score_universe,
                              sector_rank, valuation_reset_row, weighted_theme)
from screener.shortlist import (add_overlap_flags, assemble, rank_outputs,
                                tag_filtered_cohort)


def synthetic_universe(n=60, seed=7) -> pd.DataFrame:
    """Neutral mid-pack universe with NaN-free basics; archetypes get planted
    on top of this in the tests."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "cik": np.arange(n), "ticker": [f"T{i:03d}" for i in range(n)],
        "name": [f"Co {i}" for i in range(n)],
        "sic": rng.choice([2834, 3674, 3714, 5411, 7372, 2038], n),
        "fin_sector": "nonfinancial", "is_financial": False,
        "is_commodity": False, "annual_only": False,
        "zero_low_revenue": False,
        "inflection_score": rng.uniform(0.2, 0.5, n),
        "margin_vs_peak": rng.uniform(0.1, 0.4, n),
        "f_rising": rng.normal(0, 1, n),
        "valuation_reset": rng.uniform(0.1, 0.5, n),
        "si_ratio": rng.uniform(0.01, 0.08, n),
        "si_hump": rng.uniform(0.1, 0.5, n),
        "moat_demonstrated": rng.normal(0.02, 0.03, n),
        "moat_n_obs": 20,
        "gm_level": rng.uniform(0.2, 0.6, n),
        "gm_stability": -rng.uniform(0.01, 0.05, n),
        "earnings_consistency": rng.uniform(0.4, 0.8, n),
        "ev_sales": rng.uniform(1, 6, n), "p_tbv": rng.uniform(0.5, 3, n),
        "peak_score": rng.uniform(0.1, 0.4, n),
        "om_latest": rng.uniform(0.02, 0.2, n),
        "gm_latest": rng.uniform(0.2, 0.6, n),
        "leverage_prop_raw": rng.uniform(0, 0.3, n),
        "decel_score_raw": rng.uniform(0, 0.3, n),
        "dso_yoy_delta": rng.normal(0, 5, n),
        "dio_yoy_delta": rng.normal(0, 5, n),
        "buyback_score_raw": 0.0,
        "dilution_yoy": rng.uniform(0, 0.04, n),
        "beneish_m": rng.normal(-2.5, 0.5, n),
        "sloan_latest": rng.normal(-0.02, 0.03, n),
        "event_score_raw": 0.0,
        "sector_margin_declining": False,
        "cov_false_moat": 1.0,
        "at_peak_score": rng.uniform(0.0, 0.2, n),
        "pmv_score_raw": rng.uniform(0.0, 0.2, n),
        "ev_normalized": rng.uniform(8, 25, n),
        "ret_12m": rng.uniform(-0.2, 0.2, n),
        "drawdown": rng.uniform(0.2, 0.5, n),
        "history_quarters": 30,
        "bdc_or_trust": False,
    })
    df["sector2"] = df["sic"] // 100
    return df


class TestNormalization:
    def test_pct_rank_nan_safe(self):
        s = pd.Series([1.0, 2.0, np.nan, 4.0])
        r = pct_rank(s)
        assert np.isnan(r.iloc[2])
        assert r.iloc[3] == 100.0

    def test_sector_rank_fallback_below_min_bucket(self):
        df = pd.DataFrame({"v": np.arange(20, dtype=float),
                           "sector2": [1] * 16 + [2] * 4})
        r = sector_rank(df, "v", min_bucket=15)
        # small bucket -> universe-wide rank (its members are the top 4)
        assert (r.iloc[16:] > 80).all()

    def test_weighted_theme_renormalizes_never_fills(self):
        df = pd.DataFrame({"a": [80.0, 80.0], "b": [np.nan, 20.0]})
        score, cov = weighted_theme(df, {"a": 0.5, "b": 0.5})
        assert score.iloc[0] == pytest.approx(80.0)  # renormalized, not 40
        assert score.iloc[1] == pytest.approx(50.0)
        assert cov.iloc[0] == pytest.approx(0.5)

    def test_shrink_toward_universe_median_not_50(self):
        # audit 3.1: shrink target is the UNIVERSE MEDIAN of computed
        # scores, not a hard-coded 50 — buckets mixing kernel scores
        # (median well below 50) must not inflate low-coverage names
        df = pd.DataFrame({"a": [20.0, 30.0, 40.0, 80.0],
                           "b": [20.0, 30.0, 40.0, np.nan]})
        score, _ = weighted_theme(df, {"a": 0.5, "b": 0.5}, shrink=True)
        med = float(np.nanmedian([20, 30, 40, 80]))  # pre-shrink scores
        assert score.iloc[3] == pytest.approx(0.5 * 80 + 0.5 * med)
        # full-coverage rows unshrunk
        assert score.iloc[0] == pytest.approx(20.0)


class TestKernels:
    def test_drawdown_kernel_shape(self):
        assert drawdown_kernel(0.10) == 0.0
        assert drawdown_kernel(0.35) == pytest.approx(1.0)
        assert drawdown_kernel(0.60) == 1.0            # plateau
        assert drawdown_kernel(0.95) == pytest.approx(
            config.DRAWDOWN_KERNEL["taper_at_95"], abs=0.02)
        # the audit-4.4 complaint: down-70% must NOT score worse than down-45%
        assert drawdown_kernel(0.70) >= drawdown_kernel(0.45)

    def test_price_confirming_term(self):
        curling = valuation_reset_row(0.5, 0.25, +0.001)
        flat = valuation_reset_row(0.5, 0.0, -0.001)
        assert curling > flat

    def test_runup_kernel_shape(self):
        # at the high -> full credit; deep drawdown, flat year -> zero
        assert runup_kernel(0.03, 0.05) == pytest.approx(1.0)
        assert runup_kernel(0.50, 0.10) == 0.0
        # big 12-mo run rescues a mid drawdown (max of the two components)
        assert runup_kernel(0.40, 0.85) == pytest.approx(1.0)
        assert runup_kernel(0.40, 0.55) == pytest.approx(0.5, abs=0.01)
        assert np.isnan(runup_kernel(np.nan, np.nan))


class TestArchetypeCAKE:
    def test_cake_like_short_ranks_high(self):
        """The user's CAKE archetype: stock at highs, margins at own-history
        extreme reached fast, price-masking-volume signature, rich on
        normalized earnings — but NO confirmed rollover, growing revenue,
        clean forensics. Pre-change this profile ranked mid-pack."""
        df = synthetic_universe()
        j = 4
        df.loc[j, ["drawdown", "ret_12m", "at_peak_score", "pmv_score_raw",
                   "ev_normalized", "peak_score", "decel_score_raw",
                   "si_hump", "beneish_m"]] = \
            [0.03, 0.45, 0.85, 0.80, 60.0, 0.30, 0.0, 1.0, -2.6]
        out = score_universe(df)
        rank = out["short_composite"].rank(ascending=False)
        assert rank.iloc[j] <= 3
        # and the run-up row specifically fired through the at-peak gate
        assert out.loc[j, "ig_runup"] > 70

    def test_compounder_at_high_scores_no_runup(self):
        """A deserved all-time high with NO peak/PMV evidence: gate closed."""
        df = synthetic_universe()
        j = 6
        df.loc[j, ["drawdown", "ret_12m", "at_peak_score", "peak_score",
                   "pmv_score_raw"]] = [0.02, 0.60, 0.0, 0.10, 0.0]
        out = score_universe(df)
        assert out.loc[j, "ig_runup"] < 5


class TestPeerMachinery:
    def test_peer_median_excludes_financials(self):
        df = pd.DataFrame({
            "sic": [3714] * 9 + [6022] * 9,
            "om_latest": [0.10] * 9 + [0.50] * 9,
        })
        med, cnt = peer_median(df, "om_latest")
        # financial margins never contaminate pools; SIC-6xxx names get no
        # peer median at all
        assert med.iloc[:9].dropna().eq(0.10).all()
        assert med.iloc[9:].isna().all()


class TestScoreUniverse:
    def test_archetypes_rank_where_intended(self):
        df = synthetic_universe()
        # Plant THE turnaround (fallen angel): strong inflection, depressed
        # margin vs peak, deep drawdown, rising F, demonstrated moat with
        # fallen-angel spread, cheap-ish
        i_long = 0
        df.loc[i_long, ["inflection_score", "margin_vs_peak", "f_rising",
                        "valuation_reset", "moat_demonstrated",
                        "earnings_consistency", "ev_sales"]] = \
            [0.85, 0.8, 2.5, 0.9, 0.12, 0.9, 1.2]
        # Plant THE busted growth short: peaked+rolling, rich, decelerating,
        # accruals, dilution
        i_short = 1
        df.loc[i_short, ["peak_score", "ev_sales", "decel_score_raw",
                         "beneish_m", "sloan_latest", "dilution_yoy",
                         "leverage_prop_raw", "si_hump", "om_latest"]] = \
            [0.9, 25.0, 0.9, -1.0, 0.15, 0.25, 0.8, 0.9, 0.45]
        out = score_universe(df)
        long_rank = out["long_composite"].rank(ascending=False)
        short_rank = out["short_composite"].rank(ascending=False)
        assert long_rank.iloc[i_long] <= 3
        assert short_rank.iloc[i_short] <= 3
        # and they must NOT top the opposite list
        assert short_rank.iloc[i_long] > 10
        assert long_rank.iloc[i_short] > 10

    def test_unconditioned_high_margin_contributes_zero(self):
        """Audit 5.3: a great business (high margin, NO deterioration, no
        forensic flags) must get ~no margin-vs-peer contribution."""
        df = synthetic_universe()
        great = 2
        df.loc[great, ["om_latest", "peak_score", "beneish_m",
                       "event_score_raw"]] = [0.55, 0.0, -3.5, 0.0]
        out = score_universe(df)
        v = out.loc[great, "fm_margin_vs_peer"]
        assert np.isnan(v) or v < 10.0

    def test_commodity_discount_rescinded_tag_remains(self):
        """User 2026-07-08: no discount — a commodity name scores identically
        to a non-commodity twin; the tag stays visible on the row."""
        df = synthetic_universe()
        df.loc[5, "is_commodity"] = True
        base = score_universe(df.assign(is_commodity=False))
        tagged = score_universe(df)
        assert tagged.loc[5, "long_composite"] == pytest.approx(
            base.loc[5, "long_composite"])
        assert tagged.loc[5, "short_composite"] == pytest.approx(
            base.loc[5, "short_composite"])
        assert bool(tagged.loc[5, "is_commodity"])

    def test_si_bonus_additive_never_punitive(self):
        """User 2026-07-08: SI supports a short but its absence must not
        drag one down. Two identical names, one with max SI hump and one
        with NO SI data: the SI name gets up to +SI_CONFIRMATION_BONUS, the
        no-SI name loses NOTHING vs a zero-SI name."""
        df = synthetic_universe()
        a, b, c = 10, 11, 12
        for j in (b, c):  # full clones of row a (identical fundamentals)
            df.iloc[j] = df.iloc[a]
            df.loc[j, "ticker"] = f"T{j:03d}"
            df.loc[j, "cik"] = j
        df.loc[a, "si_hump"] = 1.0    # confirmed by shorts
        df.loc[b, "si_hump"] = np.nan  # market hasn't priced it in
        df.loc[c, "si_hump"] = 0.0    # zero short interest
        out = score_universe(df)
        assert out.loc[a, "si_bonus"] == pytest.approx(
            config.SI_CONFIRMATION_BONUS)
        assert out.loc[b, "si_bonus"] == 0.0
        assert out.loc[c, "si_bonus"] == 0.0
        # missing-SI and zero-SI names are treated identically: no penalty
        base_b = out.loc[b, "short_composite"] - out.loc[b, "si_bonus"]
        base_c = out.loc[c, "short_composite"] - out.loc[c, "si_bonus"]
        assert base_b == pytest.approx(base_c)

    def test_zero_revenue_nulling(self):
        df = synthetic_universe()
        df.loc[7, "zero_low_revenue"] = True
        out = score_universe(df)
        assert np.isnan(out.loc[7, "ig_ev_sales"])
        assert np.isnan(out.loc[7, "fo_beneish"])

    def test_collision_tag_pre_haircut_and_coverage_gate(self):
        df = synthetic_universe()
        # top-quartile moat + top-quartile RAW false moat, sector declining
        # (so the SCORED false-moat is haircut) -> tag must still fire
        j = 3
        df.loc[j, ["moat_demonstrated", "peak_score", "om_latest",
                   "leverage_prop_raw", "sector_margin_declining"]] = \
            [0.15, 0.95, 0.5, 0.9, True]
        out = score_universe(df)
        assert bool(out.loc[j, "false_moat_collision"])
        # same setup but thin coverage -> suppressed
        df2 = df.copy()
        df2.loc[j, "cov_false_moat"] = 0.3
        # coverage column is an input here; simulate by nulling components
        df2.loc[j, ["om_latest", "leverage_prop_raw"]] = np.nan
        out2 = score_universe(df2)
        assert not bool(out2.loc[j, "false_moat_collision"])

    def test_annual_only_momentum_cap(self):
        df = synthetic_universe()
        df.loc[9, "annual_only"] = True
        df.loc[9, "valuation_reset"] = 0.9
        out = score_universe(df)
        assert out.loc[9, "t_valuation_reset"] == pytest.approx(
            90.0 * config.ANNUAL_ONLY_MOMENTUM_CAP)


class TestMultipleDisconnect:
    def test_disconnect_fires_on_story_multiple_not_cyclical_multiple(self):
        """User 2026-07-09: same cyclical-earnings evidence, two names — the
        one wearing a rich sales multiple fires; the one already priced as a
        cyclical does not (CENX/BLBD vs UAMY/VITL distinction)."""
        df = synthetic_universe()
        rich, cheap = 7, 8
        for j in (rich, cheap):
            df.loc[j, ["pmv_score_raw", "at_peak_score"]] = [0.9, 0.6]
        df.loc[rich, "ev_sales"] = 40.0    # priced as a story
        df.loc[cheap, "ev_sales"] = 0.9    # priced as a cyclical
        out = score_universe(df)
        assert bool(out.loc[rich, "multiple_disconnect"])
        assert not bool(out.loc[cheap, "multiple_disconnect"])

    def test_no_evidence_no_disconnect_even_if_rich(self):
        df = synthetic_universe()
        j = 9
        df.loc[j, ["pmv_score_raw", "at_peak_score", "peak_score",
                   "rev_gp_divergence", "ev_sales"]] = \
            [0.0, 0.0, 0.1, False, 45.0]
        out = score_universe(df)
        assert not bool(out.loc[j, "multiple_disconnect"])


class TestRanOnPricingSuccess:
    """Mechanism-only cohort (CAKE gap): pmv + runup, no richness gate."""

    def test_pmv_runup_not_rich_fires_pricing_cohort_only(self):
        df = synthetic_universe()
        j = 6
        # CAKE shape: pmv + stock ran, but NOT rich on normalized EBIT
        df.loc[j, ["pmv_score_raw", "ret_12m", "drawdown"]] = [0.63, 0.30, 0.08]
        df.loc[j, "ev_normalized"] = 8.0   # low -> ig_ev_norm below 70 gate
        out = score_universe(df)
        assert bool(out.loc[j, "ran_on_pricing_success"])          # mechanism cohort
        assert not bool(out.loc[j, "archetype_ran_on_temp_success"])  # richness gate excludes
        assert out.loc[j, "ran_on_pricing_strength"] > 0

    def test_no_run_does_not_fire(self):
        df = synthetic_universe()
        j = 6
        df.loc[j, ["pmv_score_raw", "ret_12m", "drawdown"]] = [0.63, -0.30, 0.55]
        out = score_universe(df)
        assert not bool(out.loc[j, "ran_on_pricing_success"])


class TestPricedForImpossibleGrowth:
    """Reverse-DCF discovery tag: EV implies a growth path + steady-state margin
    the company's own history contradicts, AND it is optically rich."""

    @staticmethod
    def _with_rdcf(df):
        df["rdcf_growth_gap"] = np.nan
        df["margin_ceiling_gap"] = np.nan
        return df

    def test_fires_when_all_three_gates_met(self):
        df = self._with_rdcf(synthetic_universe())
        j = 7
        df.loc[j, "rdcf_growth_gap"] = 0.20      # priced 20pp above own CAGR
        df.loc[j, "margin_ceiling_gap"] = 0.15   # requires 15pp above own peak
        df.loc[j, "ev_normalized"] = 24.0        # rich -> high ig_ev_norm pctile
        out = score_universe(df)
        assert bool(out.loc[j, "priced_for_impossible_growth"])
        assert out.loc[j, "rdcf_disconnect_strength"] > 0

    def test_growth_gap_below_threshold_does_not_fire(self):
        df = self._with_rdcf(synthetic_universe())
        j = 7
        df.loc[j, ["rdcf_growth_gap", "margin_ceiling_gap", "ev_normalized"]] = \
            [0.00, 0.15, 24.0]
        out = score_universe(df)
        assert not bool(out.loc[j, "priced_for_impossible_growth"])

    def test_cheap_name_does_not_fire(self):
        df = self._with_rdcf(synthetic_universe())
        j = 7
        # rich growth/margin gaps but optically CHEAP (bottom of ev_normalized)
        df.loc[j, ["rdcf_growth_gap", "margin_ceiling_gap"]] = [0.20, 0.15]
        df.loc[:, "ev_normalized"] = 24.0
        df.loc[j, "ev_normalized"] = 3.0
        out = score_universe(df)
        assert not bool(out.loc[j, "priced_for_impossible_growth"])

    def test_growth_gap_is_tag_only_not_weighted(self):
        """rdcf_growth_gap feeds ONLY the priced_for_impossible_growth tag (it
        is redundant with ev_normalized), so varying it leaves short_composite
        unchanged. (margin_ceiling_gap, by contrast, IS a weighted
        ig_expectations input — covered by TestExpectationsRow.)"""
        df0 = self._with_rdcf(synthetic_universe())
        df1 = self._with_rdcf(synthetic_universe())
        df1.loc[7, "rdcf_growth_gap"] = 0.50   # only the tag input changes
        a = score_universe(df0)["short_composite"]
        b = score_universe(df1)["short_composite"]
        pd.testing.assert_series_equal(a, b, check_names=False)


class TestPeerMultipleDisconnect:
    """Directional peer disconnect: rich vs peers AND worse trajectory."""

    def test_fires_expensive_and_declining_vs_peers(self):
        df = synthetic_universe()
        j = 3
        df.loc[j, "sic"] = 2834          # a well-populated group
        df.loc[df["sic"] == 2834, "ev_sales"] = 3.0
        df.loc[j, "ev_sales"] = 40.0     # far above peers -> top val pctile
        df.loc[j, "decel_score_raw"] = 0.9  # worst trajectory in group
        out = score_universe(df)
        assert bool(out.loc[j, "peer_multiple_disconnect"])
        assert out.loc[j, "peer_rerating_downside_pct"] < 0  # downside to median

    def test_expensive_but_improving_does_not_fire(self):
        df = synthetic_universe()
        j = 3
        df.loc[j, "sic"] = 2834
        df.loc[df["sic"] == 2834, "ev_sales"] = 3.0
        df.loc[j, "ev_sales"] = 40.0
        df.loc[j, "decel_score_raw"] = 0.0   # best trajectory in group
        out = score_universe(df)
        assert not bool(out.loc[j, "peer_multiple_disconnect"])

    def test_cheap_vs_peers_never_fires(self):
        df = synthetic_universe()
        j = 3
        df.loc[j, "sic"] = 2834
        df.loc[df["sic"] == 2834, "ev_sales"] = 20.0
        df.loc[j, "ev_sales"] = 1.0          # cheapest in group
        df.loc[j, "decel_score_raw"] = 0.9
        out = score_universe(df)
        assert not bool(out.loc[j, "peer_multiple_disconnect"])


class TestCatalystBonus:
    """Timing bonus: gated on the pre-bonus short, ramped by proximity."""

    @staticmethod
    def _strong_short(df, j):
        df.loc[j, ["peak_score", "at_peak_score", "decel_score_raw",
                   "pmv_score_raw", "leverage_prop_raw", "buyback_score_raw"]] \
            = 0.95
        df.loc[j, ["ev_normalized", "ev_sales"]] = [25.0, 6.0]
        df.loc[j, ["beneish_m", "sloan_latest"]] = [1.5, 0.25]
        df.loc[j, ["event_score_raw", "om_latest"]] = [0.6, 0.30]
        return df

    def _run(self, dtc_j=3, guide=False):
        df = self._strong_short(synthetic_universe(), 2)
        df["days_to_catalyst"] = np.nan
        df["guide_reset_window"] = False
        df.loc[2, "days_to_catalyst"] = dtc_j
        df.loc[2, "guide_reset_window"] = guide
        return score_universe(df)

    def test_fired_short_gets_proximity_bonus(self):
        out = self._run(dtc_j=3)
        assert out.loc[2, "short_composite"] >= config.SHORTLIST_SCORE_THRESHOLD
        assert out.loc[2, "catalyst_bonus"] > 0
        assert bool(out.loc[2, "catalyst_imminent"])

    def test_bonus_monotonic_in_proximity(self):
        near = self._run(dtc_j=3).loc[2, "catalyst_bonus"]
        far = self._run(dtc_j=40).loc[2, "catalyst_bonus"]
        assert near > far

    def test_guide_reset_adds_credit(self):
        base = self._run(dtc_j=3, guide=False).loc[2, "catalyst_bonus"]
        withguide = self._run(dtc_j=3, guide=True).loc[2, "catalyst_bonus"]
        assert withguide > base

    def test_missing_dtc_zero_bonus_everywhere(self):
        df = self._strong_short(synthetic_universe(), 2)  # no dtc column
        out = score_universe(df)
        assert (out["catalyst_bonus"] == 0).all()

    def test_bonus_never_manufactures_a_short(self):
        """The invariant the gate exists to protect: timing re-prioritizes an
        existing short, it never creates one.

        This replaces an older assertion that NO name in the neutral universe
        could earn the bonus. That held only while the gate read the pre-BLEND
        theme mean — a harsher quantity than the composite the shortlist
        threshold is applied to, which is why 649 of 674 names with a live
        catalyst got nothing. Gating on the pre-bonus composite instead, a name
        that genuinely clears the shortlist bar on conviction does earn timing
        credit; what must remain impossible is a name below the bar earning it.
        """
        df = synthetic_universe()
        df["days_to_catalyst"] = 1.0
        df["guide_reset_window"] = True
        out = score_universe(df)
        thr = config.SHORTLIST_SCORE_THRESHOLD
        earned = out["catalyst_bonus"] > 0
        assert (out.loc[earned, "short_pre_bonus"] >= thr).all()
        assert (out.loc[~earned, "catalyst_bonus"] == 0).all()
        # No name is lifted OVER the bar by its own timing. Scoped to the
        # catalyst leg deliberately: si_bonus CAN carry a name across, which is
        # intended (short interest is confirmation evidence) and predates this
        # gate — so `short_composite` is the wrong quantity to test here.
        without_catalyst = out["short_composite"] - out["catalyst_bonus"]
        crossed = (out["short_composite"] >= thr) & (without_catalyst < thr)
        assert not crossed.any()

    def test_below_threshold_name_gets_nothing(self):
        df = synthetic_universe()
        df["days_to_catalyst"] = 1.0
        df["guide_reset_window"] = True
        out = score_universe(df)
        weak = out["short_pre_bonus"].idxmin()
        assert out.loc[weak, "short_pre_bonus"] < config.SHORTLIST_SCORE_THRESHOLD
        assert out.loc[weak, "catalyst_bonus"] == 0
        assert not bool(out.loc[weak, "catalyst_imminent"])


class TestRelativeShare:
    """Losing share to the peer cohort while priced rich (VITL shape)."""

    @staticmethod
    def _prep(df):
        df["rev_yoy_latest"] = 0.05
        df.loc[df["sic"] == 2834, "rev_yoy_latest"] = 0.15   # peers grow fast
        return df

    def test_fires_loser_and_rich(self):
        df = self._prep(synthetic_universe())
        j = 4
        df.loc[j, "sic"] = 2834
        df.loc[j, "rev_yoy_latest"] = -0.05      # shrinking vs +15% peers
        df.loc[j, ["ev_sales", "ev_normalized"]] = [40.0, 25.0]  # priced rich
        out = score_universe(df)
        assert bool(out.loc[j, "losing_share_priced_rich"])
        assert out.loc[j, "rel_share_now"] < 0

    def test_loser_but_cheap_does_not_fire(self):
        df = self._prep(synthetic_universe())
        j = 4
        df.loc[j, "sic"] = 2834
        df.loc[j, "rev_yoy_latest"] = -0.05
        df.loc[j, ["ev_sales", "ev_normalized"]] = [1.0, 8.0]   # cheap
        out = score_universe(df)
        assert not bool(out.loc[j, "losing_share_priced_rich"])

    def test_growing_with_peers_does_not_fire(self):
        df = self._prep(synthetic_universe())
        j = 4
        df.loc[j, "sic"] = 2834
        df.loc[j, "rev_yoy_latest"] = 0.20       # above peer median
        df.loc[j, ["ev_sales", "ev_normalized"]] = [40.0, 25.0]
        out = score_universe(df)
        assert not bool(out.loc[j, "losing_share_priced_rich"])


class TestExpectationsRow:
    """ig_expectations: relative-share loss + reverse-DCF margin-ceiling gap
    PROMOTED to a weighted inflated_growth row (correlations showed both
    orthogonal to decel)."""

    def test_margin_ceiling_moves_composite(self):
        lo = synthetic_universe(); lo["rev_yoy_latest"] = 0.05
        lo["margin_ceiling_gap"] = 0.0
        hi = synthetic_universe(); hi["rev_yoy_latest"] = 0.05
        hi["margin_ceiling_gap"] = 0.0
        hi.loc[7, "margin_ceiling_gap"] = 0.60   # priced for unachievable margin
        out_lo = score_universe(lo)
        out_hi = score_universe(hi)
        assert out_hi.loc[7, "ig_expectations"] > out_lo.loc[7, "ig_expectations"]
        assert out_hi.loc[7, "short_composite"] > out_lo.loc[7, "short_composite"]

    def test_relative_share_drives_tag_not_expectations(self):
        """Relative-share loss must NOT feed the composite (wrong-sign for
        windfall names); it only drives the richness-gated losing_share tag.
        So a deep share-loser with NO margin-ceiling gap gets no ig_expectations
        credit, but does fire losing_share_priced_rich when also priced rich."""
        df = synthetic_universe()
        df["rev_yoy_latest"] = 0.10
        df.loc[df["sic"] == 2834, "rev_yoy_latest"] = 0.20  # peers grow faster
        df["margin_ceiling_gap"] = np.nan                    # no expectations input
        j = 4
        df.loc[j, "sic"] = 2834
        df.loc[j, "rev_yoy_latest"] = -0.10                  # deep share loss
        df.loc[j, ["ev_sales", "ev_normalized"]] = [40.0, 25.0]  # priced rich
        out = score_universe(df)
        # margin_ceiling all-NaN -> ig_expectations all-NaN (no rel_share leak)
        assert out["ig_expectations"].isna().all()
        assert bool(out.loc[j, "losing_share_priced_rich"])


class TestCompression:
    """ROIC-WACC spread compression folded into fm_peaked."""

    def test_kernel_no_moat_excluded(self):
        # window-peak spread below min_peak_spread -> not a moat to lose -> NaN
        assert np.isnan(compression_kernel(-0.01, 0.9, 0.005, False, 0.01))

    def test_kernel_cake_shape_thin_current_spread_fires(self):
        # 750->130bps: buffer consumed, falling, and CURRENT spread thin (1.3%)
        v = compression_kernel(-0.008, 0.83, 0.075, False, spread_now=0.013)
        assert v > 0.8

    def test_kernel_still_healthy_spread_gated_out(self):
        # fell from a high peak but current spread still ELITE (20%) -> not a
        # false moat (the DUOL/TSCO false-positive the gate fixes)
        v = compression_kernel(-0.01, 0.6, 0.30, False, spread_now=0.20)
        assert v < 0.05

    def test_kernel_crossing_forces_full_credit(self):
        # already negative / about to cross WACC -> confirmed rollover
        assert compression_kernel(-0.001, 0.10, 0.05, True, spread_now=0.0) == 1.0

    def test_fm_peaked_absorbs_compression(self):
        df = synthetic_universe()
        for c, val in [("spread_slope", np.nan), ("buffer_consumed", np.nan),
                       ("spread_peak_window", np.nan), ("crossing_soon", False),
                       ("spread_now", np.nan)]:
            df[c] = val
        j = 5
        df.loc[j, "peak_score"] = 0.2  # weak on its own
        df.loc[j, ["spread_slope", "buffer_consumed", "spread_peak_window",
                   "spread_now"]] = [-0.008, 0.83, 0.075, 0.01]
        out = score_universe(df)
        # compression (setup credit 0.7) lifts the peaked row above peak-only 20
        assert out.loc[j, "fm_peaked"] > 55
        assert out.loc[j, "compression_score"] > 0.8
        # a non-compressing name's peaked row is unaffected by the new arm
        assert out.loc[0, "fm_compression"] != out.loc[0, "fm_compression"]  # NaN


class TestTagFilteredCohort:
    """Sector/attribute short cohorts: rank by on-thesis tag count, exclude
    financials / zero-rev / semis, drop names firing no tags."""

    def _df(self):
        return pd.DataFrame({
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "name": list("ABCDEF"), "sic": [2080, 2080, 2080, 2080, 6021, 3674],
            "is_financial": [False, False, False, False, True, False],
            "zero_low_revenue": [False, False, False, False, False, False],
            "short_composite": [50, 55, 52, 48, 60, 70],
            "multiple_disconnect": [True, True, False, False, True, True],
            "capacity_decay": [True, False, False, False, True, True],
        })

    def test_ranks_and_excludes(self):
        df = self._df()
        mask = pd.Series([True] * 6)
        out = tag_filtered_cohort(df, mask,
                                  ["multiple_disconnect", "capacity_decay"])
        # A (2 tags) then B (1); C/D fire nothing, E financial, F semi -> excluded
        assert list(out["ticker"]) == ["A", "B"]
        assert list(out["n_thesis_tags"]) == [2, 1]

    def test_empty_when_no_tags_present(self):
        df = self._df()
        out = tag_filtered_cohort(df, pd.Series([True] * 6), ["nonexistent_tag"])
        assert len(out) == 0


class TestStoryStockScore:
    def test_asts_like_tops_zero_rev_segment(self):
        """User 2026-07-09 (ASTS): within the zero-rev segment, a name that
        RAN while diluting heavily and trading rich on book must rank top —
        even though its margin-based short themes are all nulled."""
        df = synthetic_universe()
        df.loc[10:25, "zero_low_revenue"] = True  # a 16-name segment
        j = 12
        df.loc[j, ["drawdown", "ret_12m", "dilution_yoy", "p_tbv",
                   "sloan_latest", "si_hump"]] = \
            [0.08, 0.65, 0.30, 9.0, 0.10, 1.0]
        out = score_universe(df)
        seg = out[out["zero_low_revenue"].fillna(False)]
        rank = seg["story_short_score"].rank(ascending=False)
        assert rank.loc[j] <= 2
        # main-list names get NaN story score
        assert out.loc[0, "story_short_score"] != out.loc[0, "story_short_score"]

    def test_no_run_no_dilution_scores_low(self):
        df = synthetic_universe()
        df.loc[10:25, "zero_low_revenue"] = True
        j = 14
        df.loc[j, ["drawdown", "ret_12m", "dilution_yoy", "p_tbv",
                   "sloan_latest", "si_hump"]] = \
            [0.60, -0.30, 0.00, 0.8, -0.05, 0.0]
        out = score_universe(df)
        seg = out[out["zero_low_revenue"].fillna(False)]
        assert out.loc[j, "story_short_score"] < seg["story_short_score"].median()

    def test_story_guards_exclude_ipos_trusts_and_no_dilution(self):
        """ASTS calibration guards: fresh IPOs, ETF/trust entities, and
        names without computable dilution never get a story score."""
        df = synthetic_universe()
        df.loc[10:25, "zero_low_revenue"] = True
        df.loc[11, "history_quarters"] = 4        # 2025 IPO
        df.loc[13, "bdc_or_trust"] = True          # Grayscale-style trust
        df.loc[15, "dilution_yoy"] = np.nan        # no issuance history
        out = score_universe(df)
        for j in (11, 13, 15):
            assert out.loc[j, "story_short_score"] != out.loc[j, "story_short_score"]


class TestShortlist:
    def _scored(self):
        df = synthetic_universe(n=80)
        out = score_universe(df)
        return out

    def test_threshold_and_flags(self):
        out = add_overlap_flags(self._scored())
        thr = config.SHORTLIST_SCORE_THRESHOLD
        both = out[out["on_both_lists"]]
        assert ((both["long_composite"] >= thr)
                & (both["short_composite"] >= thr)).all()

    def test_sector_cap_spills(self):
        out = self._scored()
        # force one sector to dominate the top of the long ranking
        out = out.sort_values("long_composite", ascending=False).reset_index(drop=True)
        out.loc[:29, "sector2"] = 99
        out.loc[:59, "long_composite"] = np.linspace(95, 70, 60)
        lst = assemble(out, "long")
        counts = lst.groupby("sector2").size()
        assert counts.get(99, 0) <= config.SECTOR_CAP_PER_SIDE
        # spill happened: other sectors are present
        assert (counts.index != 99).any()

    def test_rank_outputs_segments(self):
        out = self._scored()
        out.loc[0, "is_financial"] = True
        out.loc[1, "zero_low_revenue"] = True
        res = rank_outputs(out)
        assert "universe_ranked" in res and len(res["universe_ranked"]) == len(out)
        assert out.loc[1, "ticker"] in set(res["segment_zero_revenue"]["ticker"])
        core_tickers = set(res["universe_ranked"]["ticker"]) - {
            out.loc[0, "ticker"], out.loc[1, "ticker"]}
        assert out.loc[0, "ticker"] not in set(res["long_shortlist"].get("ticker", []))
