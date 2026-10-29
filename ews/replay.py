"""Indian point-in-time replay: cases vs matched peers at 4- and 8-quarter leads.

For every case, the warning cutoff is the last quarter-end at least L quarters
before the FIRST documented default (earlier payment default where documented,
otherwise CIRP admission). At that cutoff the case and each of its matched
peers are scored using ONLY the latest annual accounts public by then
(ews/panel.py availability rule). Horizon-matched models: the 1-year model at
4 quarters, the 2-year model at 8 quarters.

Alert thresholds are fixed BEFORE looking at Indian outcomes:
* capacity threshold = 95th percentile of the model's development out-of-fold
  scores, i.e. the score above which a borrower would sit in the top 5% (40/800)
  of a book with benchmark-like risk mix;
* the 20:1 cost-optimal development threshold is reported as a sensitivity.
The rule overlay uses textbook cut-offs (Altman Z'' < 1.10 distress zone,
interest cover < 1x, Beneish M > -1.78, negative operating cash flow).
"""
from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .features import FEATURES, FEATURES_V1, FEATURES_V2, INVESTIGATE, ratios_v2
from .panel import build_panel
from .policy import lead_cutoff
from .shapley import shapley
from .benchmark_v2 import load_horizon

FIRST_DEFAULT_TYPES = ['documented_payment_default', 'documented_earlier_payment_default', 'cirp_admission',
                       'one_day_payment_delay']


def first_default(root, restructuring=False):
    ev = pd.read_csv(root / 'data/curated/events.csv')
    types = FIRST_DEFAULT_TYPES + (['distressed_restructuring'] if restructuring else [])
    ev = ev[ev.event_type.isin(types)].copy()
    ev['date'] = pd.to_datetime(ev.date_lower)
    first = ev.sort_values('date').groupby('entity_id').first().reset_index()
    return first[['entity_id', 'event_type', 'date', 'source_url']]


def zpp(r):
    keys = ['X3', 'X6', 'X7', 'X8']
    if any(r.get(k) is None or not np.isfinite(r.get(k)) for k in keys):
        return None
    return 6.56 * r['X3'] + 3.26 * r['X6'] + 6.72 * r['X7'] + 1.05 * r['X8']


def beneish_proxy(c, p):
    """Six computable Beneish components; GMI and SGAI (need COGS/SG&A) held at the neutral 1.0."""
    try:
        def f(d, k):
            v = d.get(k); return float(v) if v is not None and np.isfinite(v) else None
        need = ['sales', 'receivables', 'current_assets', 'ppe', 'total_assets', 'depreciation']
        if any(f(c, k) is None or f(p, k) is None for k in need) or f(c, 'cfo') is None or f(c, 'net_income') is None:
            return None, {}
        dsri = (f(c, 'receivables') / f(c, 'sales')) / (f(p, 'receivables') / f(p, 'sales'))
        aq = lambda d: 1 - (f(d, 'current_assets') + f(d, 'ppe')) / f(d, 'total_assets')
        aqi = aq(c) / aq(p) if aq(p) > 0 else None
        sgi = f(c, 'sales') / f(p, 'sales')
        dr = lambda d: f(d, 'depreciation') / (f(d, 'depreciation') + f(d, 'ppe'))
        depi = dr(p) / dr(c)
        lev = lambda d: (f(d, 'total_liabilities') or np.nan) / f(d, 'total_assets')
        lvgi = lev(c) / lev(p)
        tata = (f(c, 'net_income') - f(c, 'cfo')) / f(c, 'total_assets')
        comp = {'DSRI': dsri, 'GMI': 1.0, 'AQI': aqi, 'SGI': sgi, 'DEPI': depi, 'SGAI': 1.0, 'LVGI': lvgi, 'TATA': tata}
        if any(v is None or not np.isfinite(v) for v in comp.values()):
            return None, comp
        w = {'DSRI': .920, 'GMI': .528, 'AQI': .404, 'SGI': .892, 'DEPI': .115, 'SGAI': -.172, 'TATA': 4.679, 'LVGI': -.327}
        return -4.840 + sum(w[k] * comp[k] for k in w), comp
    except (TypeError, ZeroDivisionError):
        return None, {}


def score_panel(root, panel):
    out = root / 'results/v2/benchmark'
    m1 = joblib.load(out / 'model_1y_transfer_safe.joblib'); m2 = joblib.load(out / 'model_2y_transfer_safe.joblib')
    m1i = joblib.load(out / 'model_1y.joblib')
    v1 = joblib.load(root / 'results/benchmark/model.joblib')
    recs = panel.to_dict('records')
    prior = {(r['entity_id'], r['period_end']): r for r in recs}
    rows = []
    for r in recs:
        rat = ratios_v2(r)
        X = pd.DataFrame([{k: (np.nan if rat[k] is None else rat[k]) for k in FEATURES}])
        prev = [q for (e, pe), q in prior.items() if e == r['entity_id'] and pe < r['period_end']]
        prev = max(prev, key=lambda q: q['period_end']) if prev else None
        m, comp = beneish_proxy(r, prev) if prev is not None else (None, {})
        rows.append({**{k: r[k] for k in ['entity_id', 'period_end', 'period_months', 'basis', 'tier',
                                           'public_available_at', 'source_documents', 'minimum_record']},
                     **{k: X.iloc[0][k] for k in FEATURES},
                     'features_present': int(X.notna().sum(axis=1).iloc[0]),
                     'score_1y': float(m1.predict_proba(X)[:, 1][0]),
                     'score_2y': float(m2.predict_proba(X)[:, 1][0]),
                     'score_v1': float(v1.predict_proba(X[FEATURES_V1])[:, 1][0]),
                     'score_1y_with_indicators': float(m1i.predict_proba(X)[:, 1][0]),
                     'altman_zpp': zpp(rat), 'interest_cover': rat['X27'],
                     'beneish_m_proxy': m, 'cfo': r.get('cfo')})
    s = pd.DataFrame(rows)
    s['flag_zpp'] = s.altman_zpp < 1.10
    s['flag_icr'] = s.interest_cover < 1.0
    s['flag_m'] = s.beneish_m_proxy > -1.78
    s['flag_cfo'] = s.cfo < 0
    s['overlay_count'] = s[['flag_zpp', 'flag_icr', 'flag_m', 'flag_cfo']].sum(axis=1)
    return s


def thresholds(root):
    out = root / 'results/v2/benchmark'
    met = json.loads((out / 'metrics_transfer_safe.json').read_text())
    t = {}
    for h in ['1y', '2y']:
        dev = pd.read_csv(out / f'development_oof_{h}_transfer_safe.csv')
        t[h] = {'capacity_5pct': float(np.quantile(dev.score, .95)), 'cost_20to1': met[h]['dev_cost_threshold']}
    dev = pd.read_csv(out / 'development_oof_1y.csv')
    t['1y_with_indicators'] = float(np.quantile(dev.score, .95))
    return t


MIN_FEATURES = 13  # a record with fewer than ~half the 25 ratios goes to manual review, not the model


def asof(scored, entity, cutoff):
    ok = scored.minimum_record & (scored.features_present >= MIN_FEATURES) if 'features_present' in scored else scored.minimum_record
    x = scored[(scored.entity_id == entity) & (scored.public_available_at <= cutoff) & ok
               & (scored.period_end <= cutoff)]
    return None if x.empty else x.sort_values('period_end').iloc[-1]


def replay(root, scored, thr, lags_label='lag6', restructuring=False):
    ents = pd.read_csv(root / 'data/curated/entities.csv')
    cases = ents[ents.role.isin(['case', 'supplementary_case'])].entity_id.tolist()
    matches = pd.read_csv(root / 'data/curated/peer_matches.csv')
    excluded = set(ents[ents.role == 'excluded_control'].entity_id)
    fd = first_default(root, restructuring).set_index('entity_id')
    obs = []
    for case in cases:
        if case not in fd.index:
            continue
        ev = fd.loc[case]
        peers = [p for p in matches[matches.case_id == case].peer_id if p not in excluded]
        for L, h in [(4, '1y'), (8, '2y')]:
            cut = lead_cutoff(ev.date, 3 * L)
            for ent, role in [(case, 'case')] + [(p, 'peer') for p in peers]:
                r = asof(scored, ent, cut)
                base = {'case_group': case, 'entity_id': ent, 'role': role, 'lead_q': L, 'cutoff': cut.date(),
                        'event_date': ev.date.date(), 'event_type': ev.event_type, 'lag': lags_label}
                if r is None:
                    obs.append({**base, 'evaluable': False}); continue
                sc = r[f'score_{h}']
                obs.append({**base, 'evaluable': True, 'accounts_period_end': r.period_end.date(),
                            'tier': r.tier, 'score': sc, 'score_v1': r.score_v1,
                            'model_flag': sc > thr[h]['capacity_5pct'],
                            'model_flag_cost_threshold': sc > thr[h]['cost_20to1'] if thr[h]['cost_20to1'] is not None else False,
                            'v1_flag': r.score_v1 > thr['v1_capacity'],
                            'with_indicators_flag': r.score_1y_with_indicators > thr['1y_with_indicators'],
                            'features_present': r.features_present,
                            'altman_zpp': r.altman_zpp, 'interest_cover': r.interest_cover,
                            'beneish_m_proxy': r.beneish_m_proxy, 'overlay_count': r.overlay_count,
                            'overlay_flag': r.overlay_count >= 2, 'zpp_flag': bool(r.flag_zpp)})
    o = pd.DataFrame(obs)
    o['combined_flag'] = o.model_flag.fillna(False).astype(bool) | o.overlay_flag.fillna(False).astype(bool)
    return o


def summarise(o):
    res = {}
    for L in [4, 8]:
        x = o[(o.lead_q == L) & o.evaluable]
        c, p = x[x.role == 'case'], x[x.role == 'peer']
        d = {'cases_total': int((o[(o.lead_q == L)].role == 'case').sum()), 'cases_evaluable': len(c),
             'peer_observations_evaluable': len(p)}
        for flag in ['model_flag', 'model_flag_cost_threshold', 'v1_flag', 'with_indicators_flag', 'zpp_flag', 'overlay_flag', 'combined_flag']:
            tp = int(c[flag].astype(bool).sum()); fp = int(p[flag].astype(bool).sum())
            fn = len(c) - tp
            d[flag] = {'recall': tp / len(c) if len(c) else None, 'caught': tp,
                       'peer_false_alert_rate': fp / len(p) if len(p) else None, 'peer_false_alerts': fp,
                       'cost_20FN_plus_FP': 20 * fn + fp}
        d['no_alert_cost'] = 20 * len(c)
        if len(c) and len(p):
            yy = np.r_[np.ones(len(c)), np.zeros(len(p))]
            d['auc_case_vs_peers_model'] = float(roc_auc_score(yy, np.r_[c.score, p.score]))
            d['auc_case_vs_peers_v1'] = float(roc_auc_score(yy, np.r_[c.score_v1, p.score_v1]))
            z = np.r_[c.altman_zpp, p.altman_zpp].astype(float); ok = np.isfinite(z)
            d['auc_case_vs_peers_neg_zpp'] = float(roc_auc_score(yy[ok], -z[ok])) if ok.sum() > 3 and len(set(yy[ok])) == 2 else None
            # within-group rank: case ranked above all its matched peers
            ranks = []
            for g, gx in x.groupby('case_group'):
                cs = gx[gx.role == 'case']
                if len(cs) and (gx.role == 'peer').any():
                    ranks.append(float((gx[gx.role == 'peer'].score < cs.score.iloc[0]).mean()))
            d['mean_share_of_peers_ranked_below_case'] = float(np.mean(ranks)) if ranks else None
        res[f'{L}q'] = d
    return res


def pre_event_years(scored, root):
    """Rubric view: each case's last 3 fiscal years before first default vs matched-peer median."""
    fd = first_default(root).set_index('entity_id')
    matches = pd.read_csv(root / 'data/curated/peer_matches.csv')
    ents = pd.read_csv(root / 'data/curated/entities.csv'); excluded = set(ents[ents.role == 'excluded_control'].entity_id)
    rows = []
    for case, ev in fd.iterrows():
        cs = scored[(scored.entity_id == case) & (scored.period_end < ev.date) & scored.minimum_record & (scored.features_present >= MIN_FEATURES)].sort_values('period_end').tail(3)
        peers = [p for p in matches[matches.case_id == case].peer_id if p not in excluded]
        for k, r in enumerate(cs.itertuples()):
            ps = scored[scored.entity_id.isin(peers) & (scored.period_end == r.period_end) & scored.minimum_record & (scored.features_present >= MIN_FEATURES)]
            rows.append({'case': case, 'years_before_event': len(cs) - k, 'period_end': r.period_end.date(),
                         'case_score_1y': r.score_1y, 'peer_median_score_1y': ps.score_1y.median() if len(ps) else None,
                         'case_zpp': r.altman_zpp, 'peer_median_zpp': ps.altman_zpp.median() if len(ps) else None,
                         'case_interest_cover': r.interest_cover,
                         'peer_median_interest_cover': ps.interest_cover.median() if len(ps) else None,
                         'n_peers': len(ps)})
    return pd.DataFrame(rows)


def firm_year_auc(scored, root):
    """All case firm-years within 3 years before first default (positives) vs all peer firm-years (negatives)."""
    fd = first_default(root).set_index('entity_id')
    ents = pd.read_csv(root / 'data/curated/entities.csv')
    peers = set(ents[ents.role.isin(['peer', 'supplementary_peer'])].entity_id)
    s = scored[scored.minimum_record & (scored.features_present >= MIN_FEATURES)].copy()
    s['y'] = np.nan
    for case, ev in fd.iterrows():
        m = (s.entity_id == case) & (s.period_end < ev.date) & (s.period_end >= ev.date - pd.DateOffset(years=3))
        s.loc[m, 'y'] = 1
    s.loc[s.entity_id.isin(peers), 'y'] = 0
    s = s[s.y.notna()]
    res = {'positives': int(s.y.sum()), 'negatives': int((s.y == 0).sum())}
    for col, sign in [('score_1y', 1), ('score_2y', 1), ('score_v1', 1), ('score_1y_with_indicators', 1), ('altman_zpp', -1), ('interest_cover', -1)]:
        v = s[col].astype(float) * sign; ok = v.notna()
        res[f'auc_{col}'] = float(roc_auc_score(s.y[ok], v[ok]))
        res[f'n_{col}'] = int(ok.sum())
    return res, s


def explain_flagged(root, scored, obs, out, n_perm=40):
    clean, _ = load_horizon(root / 'data/raw/polish-bankruptcy.zip', '5year.arff')
    split = pd.read_csv(root / 'results/benchmark/frozen_split.csv')
    dev = clean.loc[split.split.to_numpy() == 'development', FEATURES]
    bg = dev.sample(32, random_state=7).to_numpy(dtype=float)
    model = joblib.load(root / 'results/v2/benchmark/model_1y_transfer_safe.joblib')
    predict = lambda a: model.predict_proba(pd.DataFrame(a, columns=FEATURES))[:, 1]
    sel = obs[(obs.lead_q == 4) & obs.evaluable & (obs.combined_flag | (obs.role == 'case'))]
    cards, contrib = [], []
    for r in sel.itertuples():
        row = scored[(scored.entity_id == r.entity_id) & (scored.period_end == pd.Timestamp(r.accounts_period_end))].iloc[0]
        x = row[FEATURES].to_numpy(dtype=float)
        phi, base = shapley(predict, x, bg, n_perm=n_perm, seed=11)
        total = float(predict(x[None, :])[0])
        assert abs(phi.sum() + base - total) < 1e-6
        order = np.argsort(-phi)
        drivers = [{'feature': FEATURES[j], 'label': FEATURES_V2[FEATURES[j]][1],
                    'value': None if np.isnan(x[j]) else float(x[j]), 'contribution': float(phi[j]),
                    'investigate': INVESTIGATE[FEATURES[j]]} for j in order[:3]]
        cards.append({'entity_id': r.entity_id, 'role': r.role, 'case_group': r.case_group, 'cutoff': str(r.cutoff),
                      'accounts_period_end': str(r.accounts_period_end), 'score': total, 'base': base,
                      'model_flag': bool(r.model_flag), 'overlay_flag': bool(r.overlay_flag),
                      'altman_zpp': r.altman_zpp, 'interest_cover': r.interest_cover, 'top_drivers': drivers})
        for j, f in enumerate(FEATURES):
            contrib.append({'entity_id': r.entity_id, 'cutoff': str(r.cutoff), 'feature': f,
                            'value': x[j], 'contribution': phi[j], 'base': base})
    pd.DataFrame(contrib).to_csv(out / 'shap_contributions_4q.csv', index=False)
    (out / 'evidence_cards_4q.json').write_text(json.dumps(cards, indent=2, default=lambda v: None if v != v else v))
    return cards


def run(root):
    root = Path(root); out = root / 'results/v2/india'; out.mkdir(parents=True, exist_ok=True)
    thr = thresholds(root)
    v1dev = pd.read_csv(root / 'results/benchmark/development_oof_predictions.csv')
    thr['v1_capacity'] = float(np.quantile(v1dev.score, .95))
    summary = {'method': __doc__.strip(), 'thresholds': thr}
    for lag in [6, 9]:
        panel, log = build_panel(root, lag_months=lag)
        scored = score_panel(root, panel)
        obs = replay(root, scored, thr, f'lag{lag}')
        summary[f'lag{lag}m'] = summarise(obs)
        if lag == 6:
            panel.to_csv(out / 'panel_firm_years.csv', index=False)
            log.to_csv(out / 'qc_log.csv', index=False)
            scored.to_csv(out / 'scored_firm_years.csv', index=False)
            obs.to_csv(out / 'replay_observations.csv', index=False)
            pre = pre_event_years(scored, root); pre.to_csv(out / 'pre_event_three_years.csv', index=False)
            fy, fyd = firm_year_auc(scored, root); summary['firm_year_discrimination'] = fy
            summary['panel'] = {'firm_years': len(panel), 'usable_firm_years': int(panel.minimum_record.sum()),
                                'entities': int(panel.entity_id.nunique()),
                                'tier_A_records': int((panel.tier == 'A_source_reviewed').sum()),
                                'qc_blanked_fields': len(log)}
            cards = explain_flagged(root, scored, obs, out)
            plots(root, scored, obs, pre, out)
            o2 = replay(root, scored, thr, 'lag6_restructuring', restructuring=True)
            summary['sensitivity_restructuring_as_default'] = summarise(o2)
            o2.to_csv(out / 'replay_observations_restructuring.csv', index=False)
        else:
            obs.to_csv(out / 'replay_observations_lag9.csv', index=False)
    (out / 'replay_summary.json').write_text(json.dumps(summary, indent=2, default=str))
    return summary


def plots(root, scored, obs, pre, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fd = first_default(root).set_index('entity_id')
    # 1) case score trajectory vs peer median, aligned to years-before-event
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    g = pre.groupby('years_before_event')[['case_score_1y', 'peer_median_score_1y', 'case_zpp', 'peer_median_zpp']].median()
    g = g.sort_index(ascending=False)
    x = [f't-{i}' for i in g.index]
    axes[0].plot(x, g.case_score_1y, 'o-', color='#c0392b', lw=2, label='Defaulters (median)')
    axes[0].plot(x, g.peer_median_score_1y, 's--', color='#087f8c', lw=2, label='Matched peers (median)')
    axes[0].set(ylabel='EWS score (1-yr model)', xlabel='Fiscal year before first default', title='Score separation before default')
    axes[0].legend(frameon=False)
    axes[1].plot(x, g.case_zpp, 'o-', color='#c0392b', lw=2, label='Defaulters')
    axes[1].plot(x, g.peer_median_zpp, 's--', color='#087f8c', lw=2, label='Matched peers')
    axes[1].axhline(1.10, color='#888', ls=':', label="Z'' distress line 1.10")
    axes[1].set(ylabel="Altman Z''", xlabel='Fiscal year before first default', title="Altman Z'' before default")
    axes[1].legend(frameon=False)
    for a in axes: a.spines[['top', 'right']].set_visible(False)
    fig.savefig(out / 'pre_event_trajectory.png', dpi=170); plt.close(fig)
    # 2) 4q replay: each case vs its peers
    x4 = obs[(obs.lead_q == 4) & obs.evaluable]
    groups = [gname for gname in x4.case_group.unique() if (x4[(x4.case_group == gname)].role == 'case').any()]
    fig, ax = plt.subplots(figsize=(11, 4.2), layout='constrained')
    for i, gname in enumerate(groups):
        gx = x4[x4.case_group == gname]
        p = gx[gx.role == 'peer']; c = gx[gx.role == 'case']
        ax.scatter([i] * len(p), p.score, color='#087f8c', alpha=.7, s=30, label='Matched peer' if i == 0 else None)
        ax.scatter([i] * len(c), c.score, color='#c0392b', s=70, marker='D', label='Defaulter' if i == 0 else None)
    t = float(np.quantile(pd.read_csv(out.parent / 'benchmark/development_oof_1y_transfer_safe.csv').score, .95))
    ax.axhline(t, color='#555', ls='--', lw=1, label='5% capacity threshold')
    ax.set_xticks(range(len(groups)), groups, rotation=30, ha='right')
    ax.set(ylabel='EWS score at cutoff', title='4-quarter-ahead replay: defaulter vs matched peers at the same cutoff')
    ax.set_yscale('log'); ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(out / 'replay_4q.png', dpi=170); plt.close(fig)


if __name__ == '__main__':
    import sys
    print(json.dumps(run(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]), indent=2, default=str))
