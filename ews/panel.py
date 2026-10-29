"""Indian case/peer panel: point-in-time firm-years from the extracted ledger.

Evidence tiers
--------------
Tier A  source-reviewed records (results/india/bhushan-case/reviewed_accounts.csv,
        data/curated/pilot_reviewed_accounts.csv). Override Tier B values.
Tier B  automated table-row extraction (data/curated/financial_candidates.csv)
        that passes the arithmetic consistency checks below. Every value keeps
        its document id, PDF page and raw source line.

Point-in-time rule (no look-ahead)
----------------------------------
A value printed as the CURRENT-year column of the FY-t annual report is
treated as public at FY-end + LAG months (default 6). SEBI LODR Reg. 33 requires
audited annual results within 60 days of year end, and the AGM (with annual
report) must be held within 6 months, so +6 months is conservative. A value
only available as a COMPARATIVE column in the FY-(t+1) report is public only
from that later report's date. Restated comparatives never overwrite original
values. Sensitivity runs use +9 months.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

CORE = ['total_assets', 'current_assets', 'current_liabilities', 'sales', 'profit_before_tax', 'reserves_surplus']

# Alok Industries changed its accounting year: 18-month periods (BSE report codes 0913 and 0315).
PERIOD_EXCEPTIONS = {
    ('ALOK', 2013): ('2013-09-30', 18), ('ALOK', 2015): ('2015-03-31', 18),
}


def period_of(entity, report_year, kind):
    """(period_end, months) of the current or comparative column in a report."""
    if kind == 'current':
        if (entity, report_year) in PERIOD_EXCEPTIONS:
            return PERIOD_EXCEPTIONS[(entity, report_year)]
        return (f'{report_year}-03-31', 12)
    # comparative column = previous period of the same report
    if entity == 'ALOK':
        prev = {2013: ('2012-03-31', 12), 2015: ('2013-09-30', 18), 2016: ('2015-03-31', 18),
                2012: ('2011-03-31', 12)}
        return prev.get(report_year, (f'{report_year - 1}-03-31', 12))
    return (f'{report_year - 1}-03-31', 12)


def long_ledger(root):
    c = pd.read_csv(root / 'data/curated/financial_candidates.csv')
    rows = []
    for r in c.itertuples(index=False):
        for kind, val in [('current', r.current_value), ('comparative', r.comparative_value)]:
            if pd.isna(val):
                continue
            pe, months = period_of(r.entity_id, int(r.report_year), kind)
            # date the containing report is public = its own current period end + lag
            report_pe, _ = period_of(r.entity_id, int(r.report_year), 'current')
            rows.append({'entity_id': r.entity_id, 'basis': r.basis, 'period_end': pe, 'period_months': months,
                         'field': r.field, 'value': float(val), 'column': kind,
                         'report_period_end': report_pe, 'document_id': r.document_id,
                         'pdf_page': r.pdf_page, 'source_url': r.source_url, 'source_line': r.source_line})
    return pd.DataFrame(rows)


def choose_values(ledger):
    """Original (current-column) value first; comparative only when no original exists."""
    ledger = ledger.copy()
    ledger['priority'] = (ledger.column == 'comparative').astype(int)
    ledger = ledger.sort_values(['entity_id', 'basis', 'period_end', 'field', 'priority', 'report_period_end'])
    return ledger.drop_duplicates(['entity_id', 'basis', 'period_end', 'field'], keep='first')


def pick_basis(chosen, forced):
    cov = chosen[chosen.field.isin(CORE)].groupby(['entity_id', 'basis']).size().unstack(fill_value=0)
    basis = {}
    for e, row in cov.iterrows():
        basis[e] = forced.get(e) or row.idxmax()
    return basis


def standardise(wide):
    g = lambda k: wide[k] if k in wide else pd.Series(np.nan, index=wide.index)
    out = pd.DataFrame(index=wide.index)
    out['total_assets'] = g('total_assets'); out['current_assets'] = g('current_assets')
    out['current_liabilities'] = g('current_liabilities'); out['sales'] = g('sales')
    out['net_income'] = g('net_income'); out['pbt'] = g('profit_before_tax')
    out['finance_costs'] = g('finance_costs'); out['depreciation'] = g('depreciation')
    out['other_income'] = g('other_income'); out['inventory'] = g('inventory')
    out['receivables'] = g('receivables'); out['cfo'] = g('cfo'); out['ppe'] = g('ppe')
    out['long_term_debt'] = g('long_term_debt'); out['short_term_debt'] = g('short_term_debt')
    out['cash'] = g('cash_and_bank_reported'); out['share_capital'] = g('share_capital')
    out['equity'] = g('share_capital') + g('reserves_surplus')
    out['retained_earnings'] = g('reserves_surplus')  # proxy: total reserves (incl. premium) - disclosed
    mi = g('minority_interest').fillna(0.0)
    out['minority'] = mi
    out['total_liabilities'] = out.total_assets - out.equity - mi
    out['ebit'] = out.pbt + out.finance_costs
    return out


def qc(df):
    """Arithmetic checks. Failing fields are blanked (not guessed) and logged."""
    log = []
    def blank(mask, fields, reason):
        for i in df.index[mask.fillna(False)]:
            log.append({'entity_id': df.at[i, 'entity_id'], 'period_end': df.at[i, 'period_end'],
                        'fields': ','.join(fields), 'reason': reason})
        df.loc[mask.fillna(False), fields] = np.nan
    blank(df.total_assets <= 0, ['total_assets'], 'non-positive total assets')
    blank((df.current_assets - df.current_liabilities).abs() < 1e-6, ['current_assets', 'current_liabilities'],
          'current assets identical to current liabilities (total-row extraction)')
    blank(df.current_assets < .02 * df.total_assets, ['current_assets'], 'current assets < 2% of assets (sub-total extracted)')
    blank(df.current_liabilities < .01 * df.total_assets, ['current_liabilities'], 'current liabilities < 1% of assets')
    blank(df.equity > df.total_assets, ['equity'], 'equity exceeds assets')
    blank(df.depreciation < .0005 * df.total_assets, ['depreciation'], 'depreciation implausibly small')
    blank(df.current_assets > 1.02 * df.total_assets, ['current_assets'], 'current assets exceed total assets')
    blank(df.inventory + df.receivables > 1.05 * df.current_assets, ['inventory', 'receivables'],
          'inventory + receivables exceed current assets')
    blank(df.current_liabilities > 1.5 * df.total_assets, ['current_liabilities'], 'current liabilities > 1.5x assets')
    blank(df.total_liabilities <= 0, ['total_liabilities', 'equity'], 'derived liabilities non-positive')
    blank(df.sales <= 0, ['sales'], 'non-positive sales')
    blank((df.net_income.abs() > 1.5 * df.total_assets), ['net_income'], 'net income magnitude implausible')
    blank((df.depreciation < 0) | (df.depreciation > .5 * df.total_assets), ['depreciation'], 'depreciation implausible')
    blank((df.finance_costs < 0) | (df.finance_costs > .5 * df.total_assets), ['finance_costs'], 'finance costs implausible')
    # Unit mismatch guard: sales vs assets should be within 0.005x..20x
    blank((df.sales / df.total_assets > 20) | (df.sales / df.total_assets < .005), ['sales'], 'sales/assets ratio implies unit error')
    return df, log


def reviewed(root):
    """Tier-A reviewed records in the standardised schema."""
    frames = []
    for p in [root / 'data/curated/pilot_reviewed_accounts.csv', root / 'results/india/bhushan-case/reviewed_accounts.csv']:
        if not p.exists():
            continue
        r = pd.read_csv(p)
        r = r.rename(columns={'profit_before_tax': 'pbt'})
        r['tier'] = 'A_source_reviewed'
        frames.append(r)
    if not frames:
        return pd.DataFrame()
    r = pd.concat(frames, ignore_index=True)
    keep = ['entity_id', 'period_end', 'period_months', 'basis', 'source_document_id', 'source_url', 'tier',
            'total_assets', 'current_assets', 'current_liabilities', 'equity', 'retained_earnings', 'net_income',
            'sales', 'inventory', 'receivables', 'pbt', 'finance_costs', 'cfo', 'depreciation', 'long_term_debt',
            'ppe', 'total_liabilities', 'ebit']
    r = r[[k for k in keep if k in r]].copy()
    # A reviewed record was printed as the current column of its own period's report unless noted.
    return r.drop_duplicates(['entity_id', 'period_end'], keep='first')


def build_panel(root, lag_months=6):
    root = Path(root)
    led = long_ledger(root)
    chosen = choose_values(led)
    rev = reviewed(root)
    forced = {e: 'consolidated' for e in rev.entity_id.unique()} if len(rev) else {}
    basis = pick_basis(chosen, forced)
    chosen = chosen[chosen.apply(lambda r: basis.get(r.entity_id) == r.basis, axis=1)]
    wide = chosen.pivot_table(index=['entity_id', 'period_end', 'period_months'], columns='field', values='value', aggfunc='first')
    # availability: latest report among the fields used (original column -> own report)
    avail_src = chosen.groupby(['entity_id', 'period_end']).report_period_end.max()
    docs = chosen.groupby(['entity_id', 'period_end']).document_id.agg(lambda s: ';'.join(sorted(set(s))))
    std = standardise(wide).reset_index()
    std['basis'] = std.entity_id.map(basis)
    std['tier'] = 'B_automated_extraction_qc'
    std['report_period_end'] = [avail_src.get((e, p)) for e, p in zip(std.entity_id, std.period_end)]
    std['source_documents'] = [docs.get((e, p)) for e, p in zip(std.entity_id, std.period_end)]
    # prior-year sales from the same report's comparative column (available at the same time)
    comp_sales = led[(led.field == 'sales') & (led.column == 'comparative')]
    cur_sales_doc = led[(led.field == 'sales') & (led.column == 'current')][['entity_id', 'basis', 'document_id', 'period_end']]
    m = cur_sales_doc.merge(comp_sales[['entity_id', 'basis', 'document_id', 'value']], on=['entity_id', 'basis', 'document_id'])
    m = m[m.apply(lambda r: basis.get(r.entity_id) == r.basis, axis=1)].drop_duplicates(['entity_id', 'period_end'])
    std = std.merge(m[['entity_id', 'period_end', 'value']].rename(columns={'value': 'sales_prior'}), on=['entity_id', 'period_end'], how='left')
    # Tier A override (whole record, same period)
    if len(rev):
        rev = rev.copy(); rev['report_period_end'] = rev.period_end
        rev['source_documents'] = rev.source_document_id
        key = set(zip(rev.entity_id, rev.period_end))
        prior_sales = std.set_index(['entity_id', 'period_end']).sales_prior
        std = std[[k not in key for k in zip(std.entity_id, std.period_end)]]
        rev['sales_prior'] = [prior_sales.get(k, np.nan) for k in zip(rev.entity_id, rev.period_end)]
        std = pd.concat([std, rev], ignore_index=True)
    # fill sales_prior from the panel itself where the earlier period is ORIGINALLY available no later
    std = std.sort_values(['entity_id', 'period_end']).reset_index(drop=True)
    # Prior-year sales fallback: the previous period's ORIGINAL sales figure, which was public
    # earlier than the current report, so using it is point-in-time safe. 12-month periods only.
    prev = std[['entity_id', 'period_end', 'sales', 'period_months']].copy()
    prev['next_end'] = (pd.to_datetime(prev.period_end) + pd.DateOffset(years=1)).dt.strftime('%Y-%m-%d')
    prev = prev[prev.period_months == 12].rename(columns={'sales': 'sales_prev_record'})
    std = std.merge(prev[['entity_id', 'next_end', 'sales_prev_record']], left_on=['entity_id', 'period_end'],
                    right_on=['entity_id', 'next_end'], how='left').drop(columns='next_end')
    std['sales_prior'] = std.sales_prior.fillna(std.sales_prev_record)
    std = std.drop(columns='sales_prev_record')
    std, log = qc(std)
    tb = std.tier != 'A_source_reviewed'
    std.loc[tb, 'total_liabilities'] = (std.total_assets - std.equity - std.minority.fillna(0))[tb]
    std.loc[tb, 'ebit'] = (std.pbt + std.finance_costs)[tb]
    std['minimum_record'] = std[['total_assets', 'equity']].notna().all(axis=1) & std[['sales', 'pbt']].notna().any(axis=1)
    lag = pd.DateOffset(months=lag_months)
    std['public_available_at'] = pd.to_datetime(std.report_period_end) + lag
    std['period_end'] = pd.to_datetime(std.period_end)
    return std, pd.DataFrame(log)
