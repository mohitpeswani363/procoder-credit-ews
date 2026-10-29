"""Version-2 benchmark: richer transferable features, model search, horizon decay.

Protocol (unchanged in spirit from v1, extended):
* Exact-duplicate and conflicting records removed; records with identical
  12-ratio vectors are grouped so no near-duplicate crosses the hold-out
  boundary (grouping on the coarsest deployed vector also groups the 25- and
  64-feature vectors).
* The 1-year (5year.arff) hold-out is the SAME frozen split used in v1.
  2-year and 3-year horizon files get their own frozen grouped splits.
* Configuration selection on grouped development CV only; sigmoid calibration
  as a CV ensemble on development; hold-out touched once.
* Polish data has no calendar or issuer identifiers, so a calendar split is not
  possible. "Time-aware" is implemented as (i) forecast-horizon validation
  (features measured 1, 2 and 3 years before the outcome) and (ii) the
  out-of-time, out-of-domain Indian replay (ews/panel.py).
"""
from __future__ import annotations
import io, json, zipfile, hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from .features import FEATURES, FEATURES_V1
from .policy import error_cost, optimize_policy
from .preprocessing import Winsorize

SEED = 271828
HORIZONS = {'1y': '5year.arff', '2y': '4year.arff', '3y': '3year.arff'}
ALL64 = [f'X{i}' for i in range(1, 65)]
FEATURE_SETS = {'v1_12': FEATURES_V1, 'v2_25': FEATURES, 'all_64_upper_bound': ALL64}


def load_horizon(archive, name):
    with zipfile.ZipFile(archive) as z:
        raw = z.read(name)
    body = raw.decode().split('@data', 1)[1]
    df = pd.read_csv(io.StringIO(body), header=None, na_values='?', skipinitialspace=True)
    df.columns = ALL64 + ['label']
    df['row_id'] = np.arange(len(df))
    df['feature_hash'] = pd.util.hash_pandas_object(df[ALL64], index=False).astype(str)
    conflicts = df.groupby('feature_hash').label.nunique()
    bad = set(conflicts[conflicts > 1].index)
    clean = df[~df.feature_hash.isin(bad)].drop_duplicates('feature_hash').reset_index(drop=True)
    return clean, {'raw_rows': len(df), 'raw_positives': int(df.label.sum()),
                   'clean_rows': len(clean), 'clean_positives': int(clean.label.sum())}


def pipeline(kind, indicators=True, **kw):
    if kind == 'logistic':
        model = LogisticRegression(C=kw.get('C', 1.), max_iter=3000, random_state=SEED)
    else:
        model = HistGradientBoostingClassifier(
            max_iter=kw.get('iters', 300), learning_rate=kw.get('lr', .05),
            max_depth=kw.get('depth', 5), max_leaf_nodes=kw.get('leaves', 15),
            min_samples_leaf=kw.get('leaf', 30), l2_regularization=kw.get('l2', 2.),
            early_stopping=False, random_state=SEED)
    return Pipeline([('clip', Winsorize()),
                     ('impute', SimpleImputer(strategy='median', add_indicator=indicators, keep_empty_features=True)),
                     ('scale', StandardScaler()), ('model', model)])


CONFIGS = [{'kind': 'logistic', 'C': 1.}] + [
    {'kind': 'hist_boost', 'depth': d, 'lr': lr, 'iters': it, 'leaf': lf}
    for d in [3, 5, 8] for lr, it in [(.03, 400), (.08, 200)] for lf in [20, 40]]


def top_k(p, rate=.05):
    sel = np.zeros(len(p), dtype=bool)
    sel[np.argsort(-p, kind='stable')[:int(np.floor(rate * len(p) + 1e-10))]] = True
    return sel


def metrics(y, p):
    return {'average_precision': float(average_precision_score(y, p)), 'roc_auc': float(roc_auc_score(y, p)),
            'brier': float(brier_score_loss(y, p)), 'top5': error_cost(y, top_k(p))}


def altman_zpp(X):
    return 6.56 * X['X3'] + 3.26 * X['X6'] + 6.72 * X['X7'] + 1.05 * X['X8']


def split_for(clean, horizon, out, root):
    groups = pd.util.hash_pandas_object(clean[FEATURES_V1], index=False).astype(str).to_numpy()
    if horizon == '1y':
        frozen = pd.read_csv(root / 'results/benchmark/frozen_split.csv', dtype={'feature_hash': str})
        if not (frozen.row_id.to_numpy() == clean.row_id.to_numpy()).all():
            raise RuntimeError('v1 frozen split does not align with the cleaned 1-year data')
        test = np.flatnonzero(frozen.split.to_numpy() == 'holdout')
        train = np.flatnonzero(frozen.split.to_numpy() == 'development')
    else:
        train, test = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(clean, clean.label, groups))
        path = out / f'frozen_split_{horizon}.csv'
        s = clean[['row_id', 'label']].copy(); s['split'] = 'development'; s.loc[test, 'split'] = 'holdout'
        if path.exists() and not pd.read_csv(path).equals(s):
            raise RuntimeError(f'Frozen split mismatch for {horizon}')
        s.to_csv(path, index=False)
    assert not set(groups[train]) & set(groups[test])
    return train, test, groups


def fit_select(X, y, groups, train, indicators=True):
    cv = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED + 1).split(X.iloc[train], y[train], groups[train]))
    rows = []
    for c in CONFIGS:
        p = cross_val_predict(pipeline(indicators=indicators, **c), X.iloc[train], y[train], cv=cv, method='predict_proba')[:, 1]
        rows.append({'config': c, 'dev_ap': float(average_precision_score(y[train], p))})
    best = max(rows, key=lambda r: r['dev_ap'])
    # Out-of-fold calibrated development scores (for threshold selection only).
    cal = CalibratedClassifierCV(pipeline(indicators=indicators, **best['config']), method='sigmoid', cv=3, ensemble=True)
    pdev = cross_val_predict(cal, X.iloc[train], y[train], cv=cv, method='predict_proba')[:, 1]
    model = CalibratedClassifierCV(pipeline(indicators=indicators, **best['config']), method='sigmoid', cv=cv, ensemble=True)
    model.fit(X.iloc[train], y[train])
    return model, best, rows, pdev


def run(root):
    root = Path(root); out = root / 'results/v2/benchmark'; out.mkdir(parents=True, exist_ok=True)
    archive = root / 'data/raw/polish-bankruptcy.zip'
    summary = {'protocol': __doc__.strip(), 'seed': SEED, 'features_v2': FEATURES, 'horizons': {}}
    for h, fname in HORIZONS.items():
        clean, audit = load_horizon(archive, fname)
        train, test, groups = split_for(clean, h, out, root)
        y = clean.label.to_numpy(dtype=int)
        hres = {'file': fname, 'audit': audit, 'dev_rows': len(train), 'holdout_rows': len(test),
                'holdout_positives': int(y[test].sum()), 'feature_sets': {}}
        z = -altman_zpp(clean.iloc[test]).fillna(0).to_numpy()
        hres['altman_zpp_baseline'] = {'average_precision': float(average_precision_score(y[test], z)),
                                       'top5': error_cost(y[test], top_k(z))}
        for name, cols in FEATURE_SETS.items():
            if h != '1y' and name == 'v1_12':
                pass
            X = clean[cols]
            model, best, rows, pdev = fit_select(X, y, groups, train)
            policy, curve = optimize_policy(y[train], pdev)
            ptest = model.predict_proba(X.iloc[test])[:, 1]
            m = metrics(y[test], ptest)
            m['dev_selected_config'] = best['config']; m['dev_ap'] = best['dev_ap']
            m['dev_cost_threshold'] = policy['threshold'] if np.isfinite(policy['threshold']) else None
            m['threshold_policy_holdout'] = error_cost(y[test], top_k(ptest) & (ptest > policy['threshold']))
            rng = np.random.default_rng(SEED + 10)
            pos, neg = np.flatnonzero(y[test] == 1), np.flatnonzero(y[test] == 0)
            aps = []
            for _ in range(500):
                ix = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
                aps.append(average_precision_score(y[test][ix], ptest[ix]))
            m['ap_bootstrap_95'] = [float(q) for q in np.quantile(aps, [.025, .975])]
            hres['feature_sets'][name] = m
            print(h, name, round(m['average_precision'], 4), m['top5']['recall'], best['config'], flush=True)
            if name == 'v2_25':
                joblib.dump(model, out / f'model_{h}.joblib')
                pd.DataFrame({'row_id': clean.row_id.iloc[test], 'label': y[test], 'score': ptest}).to_csv(
                    out / f'holdout_predictions_{h}.csv', index=False)
                pd.DataFrame({'row_id': clean.row_id.iloc[train], 'label': y[train], 'score': pdev}).to_csv(
                    out / f'development_oof_{h}.csv', index=False)
                pd.DataFrame(rows).to_json(out / f'model_search_{h}.json', orient='records', indent=1)
                if h == '1y':
                    capacity_curve(y[test], ptest, out)
                    pd.DataFrame({'feature': FEATURES, 'low': model.calibrated_classifiers_[0].estimator['clip'].lo_,
                                  'high': model.calibrated_classifiers_[0].estimator['clip'].hi_}).to_csv(
                        out / 'training_feature_ranges.csv', index=False)
                    # Reference score distribution of development (out-of-fold) for capacity mapping.
                    pd.DataFrame({'score': pdev}).to_csv(out / 'reference_scores_1y.csv', index=False)
                    summary['operating_point_1y'] = {
                        'dev_cost_threshold': m['dev_cost_threshold'],
                        'top5_equivalent_threshold': float(np.quantile(pdev, .95)),
                        'note': 'Score at the 95th percentile of development out-of-fold scores = the 5%-capacity cut-off for a population with Polish-like risk mix.'}
        summary['horizons'][h] = hres
    (out / 'metrics_v2.json').write_text(json.dumps(summary, indent=2, default=float))
    plots(summary, out)
    return summary


def capacity_curve(y, p, out):
    rows = []
    no_alert = 20 * int(y.sum())
    for rate in [.01, .02, .03, .04, .05, .06, .075, .10, .125, .15, .20]:
        e = error_cost(y, top_k(p, rate))
        rows.append({'capacity_rate': rate, **e, 'no_alert_cost': no_alert,
                     'cost_saved_vs_no_alert': no_alert - e['cost']})
    df = pd.DataFrame(rows)
    df['marginal_saving_per_extra_review'] = df.cost_saved_vs_no_alert.diff() / df.alerts.diff()
    df.to_csv(out / 'capacity_curve_1y.csv', index=False)


def plots(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    hs = list(summary['horizons'])
    names = [('v1_12', '#9aa5b1', 'v1: 12 ratios'), ('v2_25', '#087f8c', 'v2: 25 ratios'),
             ('all_64_upper_bound', '#e07a5f', 'All 64 (not transferable)')]
    w = .25
    for i, (k, c, lab) in enumerate(names):
        vals = [summary['horizons'][h]['feature_sets'][k]['average_precision'] for h in hs]
        axes[0].bar(np.arange(len(hs)) + (i - 1) * w, vals, w, color=c, label=lab)
    zz = [summary['horizons'][h]['altman_zpp_baseline']['average_precision'] for h in hs]
    axes[0].plot(np.arange(len(hs)), zz, 'k--o', label="Altman Z'' baseline")
    axes[0].set_xticks(range(len(hs)), ['1 yr (4q)', '2 yr (8q)', '3 yr (12q)'])
    axes[0].set(ylabel='Hold-out average precision', title='Signal by forecast horizon')
    axes[0].legend(fontsize=8, frameon=False)
    cc = pd.read_csv(out / 'capacity_curve_1y.csv')
    axes[1].plot(cc.capacity_rate * 100, cc.recall * 100, 'o-', color='#087f8c', label='Recall %')
    ax2 = axes[1].twinx()
    ax2.plot(cc.capacity_rate * 100, cc.cost, 's--', color='#e07a5f', label='Error cost (20·FN+FP)')
    axes[1].axvline(5, color='#888', ls=':')
    axes[1].set(xlabel='Review capacity (% of book)', ylabel='Recall (%)', title='Capacity vs recall and cost (1-yr hold-out)')
    ax2.set_ylabel('Error cost units')
    for a in list(axes) + [ax2]:
        a.spines[['top']].set_visible(False)
    fig.savefig(out / 'benchmark_v2.png', dpi=170); plt.close(fig)


def run_transfer_safe(root):
    """Deployed transfer model: identical protocol, but NO missing-value indicators.

    In the Polish data a missing ratio is itself a bankruptcy signal (failing firms
    file incomplete statements). In the Indian panel, missingness is caused by our
    table extraction, not by the borrower. SHAP on the first replay showed missing
    sales growth (X21) dominating several Indian scores, so the deployed model must
    not learn from missingness. Missing inputs are median-imputed (neutral).
    """
    root = Path(root); out = root / 'results/v2/benchmark'
    archive = root / 'data/raw/polish-bankruptcy.zip'
    res = {}
    for h in ['1y', '2y']:
        clean, _ = load_horizon(archive, HORIZONS[h])
        train, test, groups = split_for(clean, h, out, root)
        y = clean.label.to_numpy(dtype=int); X = clean[FEATURES]
        model, best, rows, pdev = fit_select(X, y, groups, train, indicators=False)
        policy, _ = optimize_policy(y[train], pdev)
        ptest = model.predict_proba(X.iloc[test])[:, 1]
        m = metrics(y[test], ptest)
        m.update(dev_selected_config=best['config'], dev_ap=best['dev_ap'],
                 dev_cost_threshold=float(policy['threshold']) if np.isfinite(policy['threshold']) else None,
                 capacity_5pct_threshold=float(np.quantile(pdev, .95)))
        res[h] = m
        joblib.dump(model, out / f'model_{h}_transfer_safe.joblib')
        pd.DataFrame({'score': pdev}).to_csv(out / f'development_oof_{h}_transfer_safe.csv', index=False)
        print('transfer-safe', h, round(m['average_precision'], 4), m['top5']['recall'], flush=True)
    (out / 'metrics_transfer_safe.json').write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == '__main__':
    import sys
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
