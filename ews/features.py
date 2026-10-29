"""Version-2 transferable feature set.

Twenty-five Polish benchmark ratios that can be reconstructed from Indian
Schedule III / Ind-AS annual statements. Version 1 used the first twelve; the
additions are debt-service, cash-coverage and growth ratios, which are the
dominant warning signals in Indian corporate defaults (interest burden and
short-term refinancing pressure).

Polish definitions (UCI 365): "gross profit" = profit before tax,
"profit on operating activities" = EBIT excluding other (non-operating) income,
"short-term liabilities" = current liabilities, "financial expenses" = finance costs.
"""
from __future__ import annotations
import math

FEATURES_V1 = ['X1','X2','X3','X4','X6','X7','X8','X9','X20','X23','X44','X46']

FEATURES_V2 = {
    'X1':  ('net_income / total_assets', 'Return on assets'),
    'X2':  ('total_liabilities / total_assets', 'Liabilities / assets'),
    'X3':  ('(current_assets - current_liabilities) / total_assets', 'Working capital / assets'),
    'X4':  ('current_assets / current_liabilities', 'Current ratio'),
    'X6':  ('retained_earnings / total_assets', 'Retained earnings / assets'),
    'X7':  ('ebit / total_assets', 'EBIT / assets'),
    'X8':  ('equity / total_liabilities', 'Equity / liabilities'),
    'X9':  ('sales / total_assets', 'Asset turnover'),
    'X20': ('inventory * 365 / sales', 'Inventory days'),
    'X23': ('net_income / sales', 'Net margin'),
    'X44': ('receivables * 365 / sales', 'Receivable days'),
    'X46': ('(current_assets - inventory) / current_liabilities', 'Quick ratio'),
    # ---- additions in version 2 ----
    'X10': ('equity / total_assets', 'Equity / assets'),
    'X16': ('(pbt + depreciation) / total_liabilities', 'Cash earnings / liabilities'),
    'X18': ('pbt / total_assets', 'Pre-tax profit / assets'),
    'X21': ('sales / sales_prior', 'Sales growth'),
    'X22': ('operating_profit / total_assets', 'Operating profit / assets'),
    'X26': ('(net_income + depreciation) / total_liabilities', 'Net cash earnings / liabilities'),
    'X27': ('operating_profit / finance_costs', 'Interest coverage'),
    'X40': ('(current_assets - inventory - receivables) / current_liabilities', 'Cash ratio'),
    'X50': ('current_assets / total_liabilities', 'Current assets / liabilities'),
    'X51': ('current_liabilities / total_assets', 'Short-term liabilities / assets'),
    'X53': ('equity / ppe', 'Equity / fixed assets'),
    'X59': ('(total_liabilities - current_liabilities) / equity', 'Long-term liabilities / equity'),
    'X13': ('(pbt + depreciation) / sales', 'Cash margin'),
}
FEATURES = list(FEATURES_V2)

INVESTIGATE = {
    'X1':'Check recurring profitability and reconcile earnings with operating cash flow.',
    'X2':'Review funding dependence, creditor protection and refinancing schedule.',
    'X3':'Review short-term funding needs and realizability of current assets.',
    'X4':'Check accessible cash, restricted balances and liability maturities.',
    'X6':'Verify the earned-reserve buffer against further losses.',
    'X7':'Review whether operating earnings cover financing costs.',
    'X8':'Assess the equity cushion behind creditor claims.',
    'X9':'Review asset utilisation and stuck/under-construction projects.',
    'X20':'Check inventory realisation and obsolescence.',
    'X23':'Investigate margin sustainability and exceptional items.',
    'X44':'Check collection delays, revenue recognition and receivable ageing.',
    'X46':'Assess liquid assets against short-term obligations.',
    'X10':'Assess capital adequacy of the borrower.',
    'X16':'Check whether cash earnings can service total debt.',
    'X18':'Review pre-tax profitability trend.',
    'X21':'Check whether revenue is shrinking or growth is debt-funded.',
    'X22':'Review core operating profitability excluding other income.',
    'X26':'Check debt-service capacity from net cash earnings.',
    'X27':'Interest cover below 1x means interest is being paid from new borrowing or asset sales.',
    'X40':'Check unrestricted cash against short-term obligations.',
    'X50':'Review liquid asset backing of liabilities.',
    'X51':'Short-term liabilities are a refinancing wall: check maturity profile and rollover.',
    'X53':'Check whether fixed assets are equity- or debt-funded.',
    'X59':'Review long-term leverage and covenant headroom.',
    'X13':'Review cash-generating margin.',
}


def _v(d, k):
    x = d.get(k)
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _div(a, b, positive_denominator=True):
    if a is None or b is None:
        return None
    if positive_denominator and b <= 0:
        return None
    if b == 0:
        return None
    return a / b


def _sub(a, b):
    return None if a is None or b is None else a - b


def _add(a, b):
    return None if a is None or b is None else a + b


def ratios_v2(d):
    """Compute the 25 ratios from a standardised fields dict. Missing stays None."""
    g = {k: _v(d, k) for k in ['net_income','total_assets','total_liabilities','current_assets',
        'current_liabilities','retained_earnings','ebit','equity','sales','inventory','receivables',
        'pbt','depreciation','finance_costs','other_income','ppe','sales_prior']}
    if g['ebit'] is None:
        g['ebit'] = _add(g['pbt'], g['finance_costs'])
    op = _sub(g['ebit'], g['other_income']) if g['other_income'] is not None else g['ebit']
    ta, tl, ca, cl, s, eq = (g[k] for k in ['total_assets','total_liabilities','current_assets',
                                            'current_liabilities','sales','equity'])
    annual = d.get('period_months', 12) in (12, None)
    a = (lambda x: x if annual else None)
    r = {
        'X1': a(_div(g['net_income'], ta)), 'X2': _div(tl, ta), 'X3': _div(_sub(ca, cl), ta),
        'X4': _div(ca, cl), 'X6': _div(g['retained_earnings'], ta), 'X7': a(_div(g['ebit'], ta)),
        'X8': _div(eq, tl), 'X9': a(_div(s, ta)),
        'X20': a(_div(None if g['inventory'] is None else g['inventory'] * 365, s)),
        'X23': _div(g['net_income'], s),
        'X44': a(_div(None if g['receivables'] is None else g['receivables'] * 365, s)),
        'X46': _div(_sub(ca, g['inventory']), cl),
        'X10': _div(eq, ta),
        'X16': a(_div(_add(g['pbt'], g['depreciation']), tl)),
        'X18': a(_div(g['pbt'], ta)),
        'X21': a(_div(s, g['sales_prior'])),
        'X22': a(_div(op, ta)),
        'X26': a(_div(_add(g['net_income'], g['depreciation']), tl)),
        'X27': a(_div(op, g['finance_costs'])),
        'X40': _div(_sub(_sub(ca, g['inventory']), g['receivables']), cl),
        'X50': _div(ca, tl), 'X51': _div(cl, ta),
        'X53': _div(eq, g['ppe']),
        'X59': _div(_sub(tl, cl), eq, positive_denominator=False),
        'X13': a(_div(_add(g['pbt'], g['depreciation']), s)),
    }
    return r
