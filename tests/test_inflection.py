"""Inflection-engine behavioral tests — the spec's core failure modes.

Each test encodes a sentence from the spec/audit:
- seasonal series must not register troughs/slopes (audit 2.1)
- a falling knife earns NO trough credit (audit 2.3)
- a genuine trough-with-bounce fires (audit 2.3)
- the sign-inverted construction catches peaked-and-rolling-over (audit 2.3)
- an old peak decays out of peak-recency (~8q ramp)
- conjunctive combination stops two-component carry (audit 2.4)
- cost-cut haircut hits margin metrics only (audit 4.1)
- outlier troughs are damped (audit 2.6d)
- stumble vs erosion shapes (audit 4.3)
"""

import numpy as np
import pandas as pd
import pytest

from screener import config
from screener.inflection import (combine_metrics, conjunctive, margin_vs_peak,
                                 metric_inflection, revenue_context,
                                 staleness_multiplier, trough_score)
from screener.quarterlyize import to_ttm, yoy_delta
from tests.conftest import (qseries, shape_covid_spike, shape_falling_knife,
                            shape_seasonal, shape_steady_grower,
                            shape_trough_recovery)


def infl(shape: pd.Series, asof, direction=1):
    ttm = to_ttm(shape)
    slope = yoy_delta(shape)
    return metric_inflection("x", ttm, slope, asof, direction=direction,
                             raw_q=shape)


class TestShapes:
    def test_seasonal_does_not_fire(self, asof):
        r = infl(shape_seasonal(), asof)
        # TTM removes seasonality: flat series -> no trough credit, and the
        # combined score must be far from a real turnaround's
        assert r.trough == pytest.approx(0.0, abs=1e-9) or np.isnan(r.trough)
        real = infl(shape_trough_recovery(), asof)
        assert real.score > r.score + 0.2

    def test_trough_recovery_fires(self, asof):
        r = infl(shape_trough_recovery(), asof)
        assert r.trough > 0.15          # bounced, recent
        assert r.slope > 0.55           # rising
        assert r.level > 0.5            # still depressed vs own history
        assert r.score > 0.4

    def test_falling_knife_gets_no_trough_credit(self, asof):
        r = infl(shape_falling_knife(), asof)
        assert r.trough == pytest.approx(0.0, abs=1e-9)
        # conjunctive floor keeps it eligible but LOW
        assert r.score < 0.45
        real = infl(shape_trough_recovery(), asof)
        assert real.score > r.score

    def test_steady_grower_scores_low_on_long_side(self, asof):
        r = infl(shape_steady_grower(), asof)
        # at all-time high: level component ~0 -> depressed-level test fails
        assert r.level < 0.15
        real = infl(shape_trough_recovery(), asof)
        assert real.score > r.score

    def test_covid_spike_fires_peak_primitive(self, asof):
        r = infl(shape_covid_spike(), asof, direction=-1)
        assert r.trough > 0.1           # recent recovered-FROM peak
        assert r.slope > 0.5            # falling (desired for short)
        assert r.level > 0.5            # elevated vs own history
        assert r.score > 0.35

    def test_old_peak_decays_out_of_recency(self, asof):
        # same spike but ending 3 years ago -> peak >8q old -> ramp is zero
        s = shape_covid_spike()
        old = pd.Series(s.values, index=s.index - pd.Timedelta(days=91 * 10))
        sc, det = trough_score(to_ttm(old), asof, direction=-1)
        assert sc == pytest.approx(0.0, abs=1e-6) or np.isnan(sc)

    def test_trough_still_in_window_but_no_bounce_scores_zero(self, asof):
        # latest observation IS the minimum -> q_before_latest = 0
        s = shape_falling_knife()
        sc, det = trough_score(to_ttm(s), asof)
        assert sc == pytest.approx(0.0)
        assert det["q_before_latest"] < config.TROUGH_MIN_Q_BEFORE_LATEST


class TestConjunctive:
    def test_two_components_cannot_carry_a_zero_third(self):
        # additive would give (1+1+0)/3 = 0.67; conjunctive with floor 0.12
        # gives (1*1*0.12)^(1/3) ~ 0.49 — materially lower
        high = conjunctive({"l": 1.0, "s": 1.0, "t": 0.0})
        assert high < 0.55
        balanced = conjunctive({"l": 0.7, "s": 0.7, "t": 0.7})
        assert balanced > high

    def test_floor_is_soft_not_gate(self):
        assert conjunctive({"l": 0.0, "s": 0.0, "t": 0.0}) > 0.0


class TestCombineMetrics:
    def _fake(self, score, conf=1.0):
        from screener.inflection import MetricInflection
        m = MetricInflection(metric="m")
        m.score = score
        m.confidence = conf
        return m

    def test_cost_cut_haircut_hits_margin_metrics_only(self):
        per = {"gm": self._fake(0.8), "om": self._fake(0.8),
               "revenue": self._fake(0.8)}
        cut = combine_metrics(per, "still-declining", 1.0)
        clean = combine_metrics(per, "growing", 1.0)
        assert cut.theme_score < clean.theme_score
        assert cut.cost_cut_haircut_applied
        # revenue metric unaffected: verify magnitude ~ haircut on 2/3 weight
        assert cut.theme_score > clean.theme_score * config.COST_CUT_HAIRCUT

    def test_breadth_term_penalizes_narrow_signals(self):
        wide = combine_metrics({m: self._fake(0.8) for m in
                                ("gm", "om", "roic", "ni", "fcf", "revenue")},
                               "growing", 1.0)
        narrow = combine_metrics({"gm": self._fake(0.8)}, "growing", 1.0)
        assert wide.theme_score > narrow.theme_score

    def test_breadth_denominator_matches_theme_size(self):
        """A structurally-3-metric theme (short peak) with all 3 computed
        must NOT be capped at sqrt(3/6) — its denominator is 3."""
        three = {m: self._fake(0.8) for m in ("roic", "gm", "om")}
        capped = combine_metrics(three, "unknown", 1.0)          # denom 6
        full = combine_metrics(three, "unknown", 1.0, breadth_n=3)
        assert capped.theme_score == pytest.approx(0.8 * np.sqrt(0.5), abs=0.02)
        assert full.theme_score == pytest.approx(0.8, abs=0.02)

    def test_staleness_multiplier_dampens(self):
        per = {"gm": self._fake(0.8)}
        fresh = combine_metrics(per, "growing", 1.0)
        stale = combine_metrics(per, "growing", 0.6)
        assert stale.theme_score == pytest.approx(fresh.theme_score * 0.6)


class TestOutlierDamping:
    def test_outlier_trough_damped(self, asof):
        # noisy-but-stable series with one catastrophic quarter (impairment)
        # that fully reverses -> trough is an outlier -> damped + flagged
        rng = np.random.default_rng(3)
        vals = 100.0 + rng.normal(0, 2.0, 40)
        vals[34] = -260.0  # single-quarter shock, deep enough to move TTM
        s = qseries(vals)
        sc, det = trough_score(to_ttm(s), asof, raw_q=s)
        clean = shape_trough_recovery()
        sc_clean, det_clean = trough_score(to_ttm(clean), asof, raw_q=clean)
        if sc > 0:  # if the bounce qualifies at all, it must be damped
            assert det.get("outlier_trough", False)
            assert sc < sc_clean


class TestMarginVsPeak:
    def test_stumble_credited_erosion_near_zero(self, asof):
        # stumble: plateau then sharp recent drop
        stumble = qseries(np.r_[np.full(32, 0.20), np.linspace(0.20, 0.08, 8)])
        # erosion: monotonic multi-year decline
        erosion = qseries(np.linspace(0.20, 0.08, 40))
        s_sc, s_det = margin_vs_peak(stumble, asof)
        e_sc, e_det = margin_vs_peak(erosion, asof)
        assert s_det["decline_shape"] == "stumble"
        assert e_det["decline_shape"] == "erosion"
        assert s_sc > e_sc
        assert e_sc < 0.15

    def test_peak_uses_median_of_top3_not_max(self, asof):
        # one absurd spike quarter must not define the peak
        vals = np.full(40, 0.10)
        vals[20] = 0.90
        sc, det = margin_vs_peak(qseries(vals), asof)
        assert det["peak"] == pytest.approx(0.10, abs=0.02)


class TestAtPeakPrimitive:
    """User 2026-07-08: catch the VITL/CAKE SETUP — margins at an own-history
    extreme, reached at an unusual rate, right now — WITHOUT requiring the
    rollover the peak-recency primitive needs."""

    def _score(self, vals, asof):
        from screener.inflection import at_peak_metric
        from screener.quarterlyize import to_ttm
        return at_peak_metric(to_ttm(qseries(vals)), asof)

    def test_fires_on_fresh_spike_at_peak(self, asof):
        # decade of ~10% margins, then a hard 6-quarter ramp to a record —
        # still AT the peak (no rollover): the rolling-over primitive can't
        # see it yet; the at-peak variant must
        rng = np.random.default_rng(5)
        vals = 0.10 + rng.normal(0, 0.004, 40)
        vals[-6:] = np.linspace(0.11, 0.22, 6)
        sc, det = self._score(vals, asof)
        assert sc > 0.5
        assert det["level_gate"] > 0.8 and det["delta_gate"] > 0.8

    def test_chronic_margin_expander_does_not_fire(self, asof):
        # software-transition archetype: margins rise steadily for a decade —
        # always at a level high, but the CURRENT delta is typical for
        # itself -> delta gate ~0
        vals = np.linspace(0.10, 0.30, 40)
        sc, det = self._score(vals, asof)
        assert sc < 0.15

    def test_stable_business_does_not_fire(self, asof):
        rng = np.random.default_rng(6)
        vals = 0.15 + rng.normal(0, 0.005, 40)
        sc, det = self._score(vals, asof)
        assert sc < 0.15 or np.isnan(sc)

    def test_old_peak_does_not_fire(self, asof):
        # spiked and PEAKED ~2 years ago (already declining since): recency 0
        rng = np.random.default_rng(7)
        vals = 0.10 + rng.normal(0, 0.004, 40)
        vals[-14:-8] = np.linspace(0.11, 0.22, 6)
        vals[-8:] = np.linspace(0.21, 0.14, 8)
        sc, det = self._score(vals, asof)
        assert sc == pytest.approx(0.0, abs=0.05) or np.isnan(sc)


class TestContextAndStaleness:
    def test_revenue_context_classifier(self, asof):
        grow = shape_steady_grower()
        decline = shape_falling_knife()
        assert revenue_context(grow, asof) == "growing"
        assert revenue_context(decline, asof) == "still-declining"

    def test_staleness_decay_shape(self, asof):
        fresh = staleness_multiplier(asof - pd.Timedelta(days=60), asof)
        edge = staleness_multiplier(asof - pd.Timedelta(days=139), asof)
        stale = staleness_multiplier(asof - pd.Timedelta(days=300), asof)
        dead = staleness_multiplier(asof - pd.Timedelta(days=1000), asof)
        assert fresh == 1.0 and edge == 1.0
        assert 0.5 < stale < 1.0
        assert dead == pytest.approx(config.STALENESS_DECAY_FLOOR)

    def test_no_compliant_filer_penalized(self, asof):
        # a filer whose last period ended 100 days ago is fully compliant
        assert staleness_multiplier(asof - pd.Timedelta(days=100), asof) == 1.0
