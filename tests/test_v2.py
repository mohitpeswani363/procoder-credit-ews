"""Tests for the v2 feature set, Shapley explanations, panel timing and replay guards."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from ews.features import ratios_v2, FEATURES, FEATURES_V1
from ews.shapley import shapley
from ews.panel import period_of, choose_values, qc
from ews.replay import asof, zpp, beneish_proxy
from ews.policy import lead_cutoff

ROOT = Path(__file__).resolve().parents[1]

BASE = dict(total_assets=1000., total_liabilities=600., current_assets=400., current_liabilities=300.,
            retained_earnings=100., equity=400., sales=800., inventory=100., receivables=120., pbt=50.,
            net_income=40., depreciation=30., finance_costs=40., other_income=10., ppe=500., sales_prior=700.)


def test_v2_contains_v1_and_25_features():
    assert len(FEATURES) == 25 and set(FEATURES_V1) <= set(FEATURES)


def test_ratio_definitions():
    r = ratios_v2(BASE)
    assert r['X27'] == pytest.approx((50 + 40 - 10) / 40)          # operating profit / finance costs
    assert r['X51'] == pytest.approx(0.3)
    assert r['X26'] == pytest.approx((40 + 30) / 600)
    assert r['X21'] == pytest.approx(800 / 700)
    assert r['X40'] == pytest.approx((400 - 100 - 120) / 300)
    assert r['X7'] == pytest.approx(90 / 1000)                       # EBIT = PBT + finance costs


def test_missing_stays_missing_and_non_annual_flows_blocked():
    d = dict(BASE); d.pop('finance_costs')
    assert ratios_v2(d)['X27'] is None
    d = dict(BASE, period_months=18)
    r = ratios_v2(d)
    assert r['X1'] is None and r['X9'] is None and r['X2'] is not None


def test_negative_equity_kept_for_leverage_ratio():
    r = ratios_v2(dict(BASE, equity=-50.))
    assert r['X59'] < 0 and r['X10'] < 0


def test_shapley_is_exactly_additive():
    rng = np.random.default_rng(0)
    f = lambda a: 1 / (1 + np.exp(-(a[:, 0] - 2 * a[:, 1] + a[:, 0] * a[:, 2])))
    x = np.array([1., -1., 2.]); bg = rng.normal(size=(16, 3))
    phi, base = shapley(f, x, bg, n_perm=10)
    assert phi.sum() + base == pytest.approx(f(x[None])[0], abs=1e-9)


def test_comparative_column_is_dated_by_later_report():
    led = pd.DataFrame([
        dict(entity_id='A', basis='s', period_end='2014-03-31', field='sales', value=1., column='comparative', report_period_end='2015-03-31'),
        dict(entity_id='A', basis='s', period_end='2014-03-31', field='sales', value=2., column='current', report_period_end='2014-03-31')])
    c = choose_values(led)
    assert len(c) == 1 and c.iloc[0].value == 2. and c.iloc[0].report_period_end == '2014-03-31'


def test_alok_18_month_periods():
    assert period_of('ALOK', 2013, 'current') == ('2013-09-30', 18)
    assert period_of('ALOK', 2016, 'comparative') == ('2015-03-31', 18)


def test_qc_blanks_total_row_extraction():
    df = pd.DataFrame([dict(entity_id='A', period_end='2014', total_assets=100., current_assets=50., current_liabilities=50.,
                            inventory=10., receivables=10., total_liabilities=50., equity=50., sales=80., net_income=1.,
                            depreciation=1., finance_costs=1.)])
    out, log = qc(df)
    assert np.isnan(out.current_assets[0]) and np.isnan(out.current_liabilities[0]) and log


def test_asof_never_uses_unpublished_accounts():
    s = pd.DataFrame({'entity_id': ['A', 'A'], 'period_end': pd.to_datetime(['2013-03-31', '2014-03-31']),
                      'public_available_at': pd.to_datetime(['2013-09-30', '2014-09-30']), 'minimum_record': [True, True]})
    r = asof(s, 'A', pd.Timestamp('2014-06-30'))
    assert r.period_end == pd.Timestamp('2013-03-31')
    assert asof(s, 'A', pd.Timestamp('2013-06-30')) is None


def test_lead_cutoff_is_at_least_four_quarters_before_event():
    ev = pd.Timestamp('2015-06-29')
    c = lead_cutoff(ev, 12)
    assert c <= ev - pd.DateOffset(months=12) and c.is_quarter_end


def test_altman_zpp_and_beneish_proxy():
    r = ratios_v2(BASE)
    assert zpp(r) == pytest.approx(6.56 * .1 + 3.26 * .1 + 6.72 * .09 + 1.05 * (400 / 600))
    prior = dict(BASE); cur = dict(BASE, cfo=20.)
    m, comp = beneish_proxy(cur, prior)
    assert comp['GMI'] == 1.0 and comp['SGAI'] == 1.0
    assert m == pytest.approx(-4.840 + .920 + .528 + .404 + .892 + .115 - .172 + 4.679 * (20 / 1000) - .327)


@pytest.mark.skipif(not (ROOT / 'results/v2/india/replay_summary.json').exists(), reason='replay not run')
def test_replay_thresholds_fixed_from_benchmark_only():
    import json
    s = json.loads((ROOT / 'results/v2/india/replay_summary.json').read_text())
    dev = pd.read_csv(ROOT / 'results/v2/benchmark/development_oof_1y.csv')
    assert s['thresholds']['1y']['capacity_5pct'] == pytest.approx(float(np.quantile(dev.score, .95)))
