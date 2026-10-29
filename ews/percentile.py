"""Rank of each defaulter within the whole control universe scorable at the same cutoff.

Capacity rules are rank-based (top 40 of 800), so the portable question is where a
future defaulter sits relative to non-defaulting listed borrowers at the same date,
not whether a Polish-calibrated probability crosses a Polish threshold.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .replay import first_default, asof
from .policy import lead_cutoff


def run(root):
    root = Path(root); out = root / 'results/v2/india'
    s = pd.read_csv(out / 'scored_firm_years.csv', parse_dates=['period_end', 'public_available_at'])
    ents = pd.read_csv(root / 'data/curated/entities.csv')
    controls = ents[ents.role.isin(['peer', 'supplementary_peer'])].entity_id.tolist()
    cases = ents[ents.role.isin(['case', 'supplementary_case'])].entity_id.tolist()
    fd = first_default(root).set_index('entity_id')
    rows = []
    for case in cases:
        if case not in fd.index:
            continue
        for L in [4, 8]:
            cut = lead_cutoff(fd.loc[case].date, 3 * L)
            c = asof(s, case, cut)
            if c is None:
                continue
            pool = [asof(s, e, cut) for e in controls]
            pool = pd.DataFrame([p for p in pool if p is not None])
            if len(pool) < 5:
                continue
            row = {'case': case, 'lead_q': L, 'cutoff': cut.date(), 'n_controls': len(pool)}
            for col, sign in [('score_1y' if L == 4 else 'score_2y', 1), ('score_v1', 1), ('altman_zpp', -1)]:
                key = 'v2' if col.startswith('score_') and col != 'score_v1' else ('v1' if col == 'score_v1' else 'zpp')
                cv = c[col]
                if cv is None or not np.isfinite(cv):
                    row[f'pct_{key}'] = None; continue
                pv = pool[col].astype(float).dropna() * sign
                row[f'pct_{key}'] = float((pv < cv * sign).mean())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'defaulter_percentiles.csv', index=False)
    summ = {}
    for L in [4, 8]:
        d = df[df.lead_q == L]
        summ[f'{L}q'] = {k: {'n': int(d[k].notna().sum()), 'median_percentile': float(d[k].median()),
                              'share_in_top_quintile': float((d[k] >= .8).mean()) if d[k].notna().any() else None,
                              'share_in_top_decile': float((d[k] >= .9).mean()) if d[k].notna().any() else None,
                              'share_in_top_5pct': float((d[k] >= .95).mean()) if d[k].notna().any() else None}
                         for k in ['pct_v2', 'pct_v1', 'pct_zpp'] if k in d}
    (out / 'defaulter_percentiles.json').write_text(json.dumps(summ, indent=2))
    plot(df, out)
    return df, summ


def plot(df, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout='constrained', sharey=True)
    for ax, L in zip(axes, [4, 8]):
        d = df[df.lead_q == L].reset_index(drop=True)
        x = np.arange(len(d)); w = .27
        for i, (k, c, lab) in enumerate([('pct_v1', '#087f8c', 'v1 model (12 ratios)'), ('pct_v2', '#9ccfd6', 'v2 model (25 ratios)'),
                                         ('pct_zpp', '#e07a5f', "Altman Z'' (inverted)")]):
            ax.bar(x + (i - 1) * w, d[k].fillna(0) * 100, w, color=c, label=lab)
        ax.axhline(95, color='#333', ls='--', lw=1); ax.text(len(d) - .5, 96, 'top 5%', fontsize=7, ha='right')
        ax.set_xticks(x, d.case, rotation=25, ha='right', fontsize=8)
        ax.set_title(f'{L} quarters before first default', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Percentile among non-defaulting\nlisted controls at the same cutoff')
    axes[0].legend(frameon=False, fontsize=7.5, loc='lower left')
    fig.savefig(out / 'defaulter_percentiles.png', dpi=170); plt.close(fig)


if __name__ == '__main__':
    df, s = run(Path('.'))
    print(df.to_string()); print(json.dumps(s, indent=2))
