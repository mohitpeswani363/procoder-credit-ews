"""Build the final submission report PDF (selectable text, <= 8 pages) from saved results."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

REPO = 'https://github.com/mohitpeswani363/procoder-credit-ews'
TEAL = colors.HexColor('#087f8c'); RED = colors.HexColor('#c0392b'); GREY = colors.HexColor('#5f6b76')
LIGHT = colors.HexColor('#eef5f6')


def fonts():
    base = '/usr/share/fonts/truetype/dejavu/'
    try:
        pdfmetrics.registerFont(TTFont('DV', base + 'DejaVuSans.ttf'))
        pdfmetrics.registerFont(TTFont('DVB', base + 'DejaVuSans-Bold.ttf'))
        pdfmetrics.registerFont(TTFont('DVI', base + 'DejaVuSans-Oblique.ttf'))
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        registerFontFamily('DV', normal='DV', bold='DVB', italic='DVI', boldItalic='DVB')
        return 'DV', 'DVB'
    except Exception:
        return 'Helvetica', 'Helvetica-Bold'


def styles():
    f, fb = fonts()
    ss = getSampleStyleSheet()
    S = {
        'title': ParagraphStyle('t', fontName=fb, fontSize=17, leading=21, textColor=colors.black, spaceAfter=2),
        'sub': ParagraphStyle('s', fontName=f, fontSize=9.2, leading=12, textColor=GREY),
        'h1': ParagraphStyle('h1', fontName=fb, fontSize=12.2, leading=15, textColor=TEAL, spaceBefore=7, spaceAfter=3),
        'h2': ParagraphStyle('h2', fontName=fb, fontSize=9.6, leading=12, spaceBefore=4, spaceAfter=1.5),
        'body': ParagraphStyle('b', fontName=f, fontSize=8.6, leading=11.4, alignment=TA_LEFT, spaceAfter=3),
        'small': ParagraphStyle('sm', fontName=f, fontSize=7.4, leading=9.4, textColor=GREY, spaceAfter=2),
        'cell': ParagraphStyle('c', fontName=f, fontSize=7.6, leading=9.4),
        'cellb': ParagraphStyle('cb', fontName=fb, fontSize=7.6, leading=9.4),
        'bul': ParagraphStyle('bu', fontName=f, fontSize=8.6, leading=11.2, leftIndent=9, bulletIndent=0, spaceAfter=1.5),
        'kpi': ParagraphStyle('k', fontName=fb, fontSize=15, leading=17, textColor=TEAL),
        'kpil': ParagraphStyle('kl', fontName=f, fontSize=7.2, leading=9, textColor=GREY),
    }
    return S


def table(rows, widths, S, header=True, zebra=True, bold_col0=False):
    data = []
    for i, r in enumerate(rows):
        data.append([Paragraph(str(c), S['cellb'] if (header and i == 0) or (bold_col0 and j == 0) else S['cell'])
                     for j, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, 0), .6, TEAL),
          ('TOPPADDING', (0, 0), (-1, -1), 2.2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
          ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3)]
    if header:
        st.append(('BACKGROUND', (0, 0), (-1, 0), LIGHT))
    if zebra:
        for i in range(2 if header else 1, len(rows), 2):
            st.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f7f9fa')))
    t.setStyle(TableStyle(st))
    return t


def pct(x, d=0):
    return '–' if x is None else f'{100 * x:.{d}f}%'


def num(x, d=2):
    return '–' if x is None or x != x else f'{x:.{d}f}'


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(GREY)
    canvas.drawString(15 * mm, 9 * mm, 'Team procoder · Zeckathon PS2 Risk & Fraud · Credit Early-Warning System')
    canvas.drawRightString(195 * mm, 9 * mm, f'Page {doc.page}')
    canvas.restoreState()


def bullets(items, S):
    return [Paragraph(x, S['bul'], bulletText='•') for x in items]


def build(root):
    root = Path(root); S = styles()
    B = json.loads((root / 'results/v2/benchmark/metrics_v2.json').read_text())
    TS = json.loads((root / 'results/v2/benchmark/metrics_transfer_safe.json').read_text())
    V1 = json.loads((root / 'results/benchmark/metrics.json').read_text())['holdout']
    R = json.loads((root / 'results/v2/india/replay_summary.json').read_text())
    P = json.loads((root / 'results/v2/india/defaulter_percentiles.json').read_text())
    pct_df = pd.read_csv(root / 'results/v2/india/defaulter_percentiles.csv')
    cards = json.loads((root / 'results/v2/india/evidence_cards_4q_v1.json').read_text())
    cc = pd.read_csv(root / 'results/v2/benchmark/capacity_curve_1y.csv')
    h1 = B['horizons']['1y']; f = h1['feature_sets']
    L4, L8 = R['lag6m']['4q'], R['lag6m']['8q']
    out = root / 'submission' / 'Zeckathon_PS2_procoder_Final_Report.pdf'
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=13 * mm,
                            bottomMargin=15 * mm, title='Credit Early-Warning System - Team procoder',
                            author='Team procoder', subject='Zeckathon PS2 Risk & Fraud')
    W = A4[0] - 30 * mm
    st = []
    P_ = lambda t, s='body': st.append(Paragraph(t, S[s]))

    # ---------------- Page 1: summary ----------------
    P_('Corporate Credit Early-Warning System', 'title')
    P_('Zeckathon × IIT Patna TIC · <b>PS2: Risk &amp; Fraud Analytics</b> · Team <b>procoder</b>: Mantra Khandelwal, '
       'Mohit Peshwani, Om Kumar, Omkar Mainkar', 'sub')
    P_(f'Code &amp; data: <link href="{REPO}"><u>{REPO}</u></link> · reproduce with <font face="Courier">python run_v2.py</font>', 'sub')
    st.append(Spacer(1, 5))
    kpis = [
        (f"{f['v2_25']['average_precision']:.2f}", f"Hold-out average precision, 1-yr horizon (v1: {V1['average_precision']:.2f})"),
        (pct(f['v2_25']['top5']['recall']), f"Recall at 5% review budget (v1: {pct(V1['top5']['recall'])})"),
        (f"{f['v2_25']['top5']['cost']:,}", f"Error cost 20·FN+FP (v1: {V1['top5']['cost']:,}; no alerts: {20 * h1['holdout_positives']:,})"),
        (f"{int(round(P['4q']['pct_v1']['share_in_top_5pct'] * P['4q']['pct_v1']['n']))}/{P['4q']['pct_v1']['n']}",
         'Indian defaulters ranked in the top 5% of non-defaulting controls, 4 quarters ahead'),
    ]
    kt = Table([[Paragraph(a, S['kpi']) for a, _ in kpis], [Paragraph(b, S['kpil']) for _, b in kpis]],
               colWidths=[W / 4] * 4)
    kt.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), LIGHT), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('LINEAFTER', (0, 0), (2, -1), .5, colors.white), ('TOPPADDING', (0, 0), (-1, -1), 4),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    st.append(kt)
    P_('The ask and our answer', 'h1')
    P_('A credit desk can investigate 40 of 800 listed borrowers (5%) each quarter, and a missed default costs 20 times a '
       'false alarm. We built a reproducible early-warning system that (1) ranks borrowers with a distress model trained on '
       'the UCI Polish bankruptcy data using only ratios that can be rebuilt from Indian annual reports, (2) overlays '
       'accounting-manipulation and stress screens (Beneish M, Piotroski F, Altman Z\'\', interest cover, operating cash flow), '
       '(3) explains every flag with exactly-additive Shapley drivers and an investigation question, and (4) was replayed '
       'point-in-time on 11 Indian defaulters and their matched listed peers, using only accounts that were public at each cutoff.')
    P_('Key findings', 'h1')
    st += bullets([
        f"<b>Debt-service ratios matter most.</b> Adding interest cover, cash-earnings-to-liabilities, short-term-liability and growth "
        f"ratios lifted hold-out average precision from {V1['average_precision']:.2f} to {f['v2_25']['average_precision']:.2f} and caught "
        f"{f['v2_25']['top5']['tp']} instead of {V1['top5']['tp']} of {h1['holdout_positives']} bankruptcies inside the 5% budget, on the "
        f"same frozen hold-out. The signal persists 2 and 3 years ahead (AP {B['horizons']['2y']['feature_sets']['v2_25']['average_precision']:.2f} / "
        f"{B['horizons']['3y']['feature_sets']['v2_25']['average_precision']:.2f}), 2.6-6x the Altman Z'' baseline.",
        f"<b>Rankings transfer to India; Polish probabilities do not.</b> Four quarters ahead, no Indian borrower crossed the Polish-calibrated alert threshold, "
        f"yet the frozen 12-ratio model placed {int(round(P['4q']['pct_v1']['share_in_top_5pct'] * P['4q']['pct_v1']['n']))} of "
        f"{P['4q']['pct_v1']['n']} evaluable defaulters in the top 5% of listed controls four quarters before default (median percentile "
        f"{P['4q']['pct_v1']['median_percentile'] * 100:.0f}), and {int(round(P['8q']['pct_v1']['share_in_top_5pct'] * P['8q']['pct_v1']['n']))} of "
        f"{P['8q']['pct_v1']['n']} eight quarters before. A capacity rule is a <i>rank</i> rule, so the desk should queue the top 40 by rank.",
        f"<b>Explainability caught a silent failure.</b> Shapley drivers showed several Indian scores were driven by <i>missing</i> sales growth: "
        f"in the Polish data a missing ratio signals a failing filer, in our panel it signals an extraction gap. We retrained a transfer-safe "
        f"model without missing-value indicators and added a data-coverage gate instead of reporting the inflated hits.",
        f"<b>Simple screens remain strong second opinions.</b> Altman Z'' &lt; 1.10 flagged {L4['zpp_flag']['caught']} of {L4['cases_evaluable']} "
        f"defaulters at 4 quarters with {L4['zpp_flag']['peer_false_alerts']} false alerts in {L4['peer_observations_evaluable']} peer checks "
        f"(cost {L4['zpp_flag']['cost_20FN_plus_FP']} vs {L4['no_alert_cost']} for no alerts).",
        f"<b>Honest limit:</b> only {L4['cases_evaluable']} of {L4['cases_total']} defaulters have enough machine-extracted data at the 4-quarter cutoff. "
        f"Every number is reproducible and every input is traceable to a PDF page; the panel is small, so Indian results are evidence, not proof.",
    ], S)
    P_('System architecture', 'h1')
    arch = [['Stage', 'What happens', 'Code'],
            ['1 Collect', '196 public PDFs (179 annual reports) with original URLs and SHA-256 hashes', 'ews/collect.py'],
            ['2 Extract', f"4,309 table-row candidates with page and raw line; QC rules blank implausible values ({R['panel']['qc_blanked_fields']} blanked)", 'ews/extract.py, panel.py'],
            ['3 Point-in-time', f"{R['panel']['firm_years']} firm-years for {R['panel']['entities']} issuers; public at FY-end + 6 months; originals beat restated comparatives", 'ews/panel.py'],
            ['4 Score', '25 ratios → calibrated gradient boosting (1-yr and 2-yr horizons); frozen 12-ratio ranker', 'ews/features.py, benchmark_v2.py'],
            ['5 Screen', "Altman Z'', Beneish M (8-var; proxy when COGS/SG&amp;A absent), Piotroski F, interest cover, CFO", 'ews/accounting.py, replay.py'],
            ['6 Explain', 'Permutation Shapley (exactly additive) → top-3 drivers + investigation question', 'ews/shapley.py'],
            ['7 Queue', 'Rank, cap at floor(5% × N) = 40 of 800, analyst → senior officer → credit committee', 'ews/policy.py']]
    st.append(table(arch, [22 * mm, 118 * mm, W - 140 * mm], S))

    # ---------------- Page 2: model ----------------
    st.append(PageBreak())
    P_('1. Distress model (UCI Polish bankruptcy benchmark)', 'h1')
    P_('<b>Data.</b> UCI Polish Companies Bankruptcy (Tomczak et al.), five files of 64 ratios measured 1-5 years before the outcome. '
       f"The 1-year file has {h1['audit']['raw_rows']:,} records; after removing exact duplicates and label conflicts {h1['audit']['clean_rows']:,} "
       f"remain ({h1['audit']['clean_positives']} bankrupt, {h1['audit']['clean_positives'] / h1['audit']['clean_rows'] * 100:.1f}%). "
       '<b>Time-aware design.</b> The archive has no calendar dates or company IDs, so a calendar split is impossible. We use the two '
       'time-aware tests the data allows: (i) separate models for features measured 1, 2 and 3 years before bankruptcy, and (ii) an '
       'out-of-time, out-of-domain test on Indian defaults (2013-2021) with the model trained only on Polish data. Records with identical '
       '12-ratio vectors are grouped so no near-duplicate crosses the hold-out boundary; the 1-year hold-out is the same frozen split as v1.')
    P_('<b>Features.</b> We kept only ratios we can rebuild from Indian Schedule III / Ind-AS statements. v1 used 12; v2 adds 13 '
       'debt-service and liquidity ratios: interest cover (X27), cash earnings / liabilities (X16, X26), pre-tax and operating profit / assets '
       '(X18, X22), sales growth (X21), cash ratio (X40), short-term liabilities / assets (X51), equity / assets and fixed assets (X10, X53), '
       'long-term liabilities / equity (X59), current assets / liabilities (X50) and cash margin (X13). <b>Model.</b> 13 prespecified '
       'configurations (logistic + histogram gradient boosting) compared by grouped 3-fold development CV; winsorising, median imputation '
       'and scaling are fitted inside folds; sigmoid calibration as a CV ensemble; the hold-out is scored once.')
    rows = [['Hold-out (never used for choices)', 'Avg precision [95% CI]', 'ROC-AUC', 'Recall @5%', 'Precision @5%', 'Cost 20·FN+FP']]
    rows.append(["Altman Z'' (textbook baseline)", num(h1['altman_zpp_baseline']['average_precision']), '–',
                 pct(h1['altman_zpp_baseline']['top5']['recall']), pct(h1['altman_zpp_baseline']['top5']['precision']),
                 f"{h1['altman_zpp_baseline']['top5']['cost']:,}"])
    rows.append(['v1 submitted model (12 ratios, frozen)', num(V1['average_precision']), num(V1['roc_auc']),
                 pct(V1['top5']['recall']), pct(V1['top5']['precision']), f"{V1['top5']['cost']:,}"])
    for k, lab in [('v2_25', '<b>v2 deployed benchmark model (25 ratios)</b>'), ('all_64_upper_bound', 'All 64 ratios (upper bound, not transferable)')]:
        m = f[k]
        rows.append([lab, f"{num(m['average_precision'])} [{num(m['ap_bootstrap_95'][0])}-{num(m['ap_bootstrap_95'][1])}]",
                     num(m['roc_auc']), pct(m['top5']['recall']), pct(m['top5']['precision']), f"{m['top5']['cost']:,}"])
    m = TS['1y']
    rows.append(['v2 transfer-safe (no missing indicators)', num(m['average_precision']), num(m['roc_auc']),
                 pct(m['top5']['recall']), pct(m['top5']['precision']), f"{m['top5']['cost']:,}"])
    st.append(table(rows, [62 * mm, 30 * mm, 16 * mm, 20 * mm, 22 * mm, W - 150 * mm], S))
    P_(f"1,170 hold-out records, 81 bankrupt; 5% budget = 58 alerts; no-alert cost = 1,620. Hyper-parameters chosen on development CV: "
       f"{f['v2_25']['dev_selected_config']}.", 'small')
    st.append(Image(str(root / 'results/v2/benchmark/benchmark_v2.png'), width=W, height=W * 4 / 11))
    hz = [['Horizon (features measured before outcome)', 'Hold-out n / bankrupt', 'v1 AP', 'v2 AP', 'v2 recall @5%', "Z'' AP"]]
    for h, lab in [('1y', '1 year (≥ 4 quarters)'), ('2y', '2 years (≥ 8 quarters)'), ('3y', '3 years (≥ 12 quarters)')]:
        x = B['horizons'][h]
        hz.append([lab, f"{x['holdout_rows']:,} / {x['holdout_positives']}", num(x['feature_sets']['v1_12']['average_precision']),
                   num(x['feature_sets']['v2_25']['average_precision']), pct(x['feature_sets']['v2_25']['top5']['recall']),
                   num(x['altman_zpp_baseline']['average_precision'])])
    st.append(table(hz, [62 * mm, 32 * mm, 18 * mm, 18 * mm, 24 * mm, W - 154 * mm], S))

    # ---------------- Page 3: policy ----------------
    P_('2. Capacity, the 20:1 cost and the alert policy', 'h1')
    c5 = cc[cc.capacity_rate == .05].iloc[0]; c10 = cc[cc.capacity_rate == .10].iloc[0]; c3 = cc[cc.capacity_rate == .03].iloc[0]
    P_(f"For calibrated probabilities the cost-minimising rule is <i>alert when p &gt; 1/21 = 4.8%</i>. At realistic default rates that "
       f"rule wants far more than 5% of the book, so <b>capacity, not the threshold, binds</b>: the development-optimal threshold "
       f"({f['v2_25']['dev_cost_threshold']:.3f}) sits at the 5% cap and selects the same 58 names on the hold-out. The capacity curve "
       f"shows what each extra analyst buys: going from 3% to 5% of the book adds {int(c5.tp - c3.tp)} caught defaults "
       f"(saving {int(c5.cost_saved_vs_no_alert - c3.cost_saved_vs_no_alert)} cost units), and going from 5% to 10% adds another "
       f"{int(c10.tp - c5.tp)} at {c10.marginal_saving_per_extra_review:.1f} units per extra review. Beyond about 10% each review saves less "
       "than the 20-unit value of one caught default divided by the reviews needed, which is the case for asking for a larger desk "
       "only up to roughly that point.")
    pol = [['Step', 'Policy'],
           ['Universe', 'All eligible listed borrowers at quarter-end; already-defaulted names go to workout, never silently dropped.'],
           ['Data gate', 'Score only accounts public by the cutoff; records with fewer than 13 of 25 ratios go to manual review.'],
           ['Rank', 'Rank by model score within the book; take floor(5% × N) = 40 of 800. Ties broken by stable ID.'],
           ['Escalate', "Outside the 40, any name with ≥ 2 red flags (Z'' &lt; 1.10, interest cover &lt; 1, M &gt; −1.78, CFO &lt; 0) replaces the lowest-ranked model name."],
           ['Explain', 'Each alert gets a card: score, rank, top-3 Shapley drivers, screen values, investigation questions, source pages.'],
           ['Decide', 'Analyst (8 × 5 names) → senior credit officer confirms/rejects → credit committee acts. Log owner, disposition, follow-up.']]
    st.append(table(pol, [22 * mm, W - 22 * mm], S, bold_col0=True))

    # ---------------- Indian test ----------------
    P_('3. Indian case test: 11 defaulters vs matched listed peers, point-in-time', 'h1')
    P_(f"<b>Panel.</b> 10 companies admitted to insolvency (IBBI CIRP list, 2017-2022) plus IL&amp;FS Transportation (ITNL) for the IL&amp;FS "
       f"group; 30 sector-matched listed controls plus 3 road-sector controls (3 earlier candidates removed for their own defaults or a late IPO). "
       f"{R['panel']['firm_years']} firm-years from {R['panel']['entities']} issuers; {R['panel']['tier_A_records']} source-reviewed (Tier A), the rest "
       f"machine-extracted with arithmetic QC (Tier B). <b>Event date</b> = first documented payment default where known (trustee notices, rating "
       "actions), otherwise CIRP admission; restructuring (CDR/SDR) is a sensitivity. <b>Cutoff</b> = last quarter-end at least 4 (or 8) quarters "
       "before the event. <b>No look-ahead:</b> a fiscal year counts only from FY-end + 6 months (Reg. 33 results in 60 days; AGM within "
       "6 months), restated comparatives never replace originals, thresholds were fixed on Polish development data before any Indian score was seen.")
    rows = [['Case', 'First default used', 'Cutoff 4q', 'Accounts used', 'v1 pct 4q', 'v2 pct 4q', "Z'' 4q", 'v1 pct 8q']]
    o = pd.read_csv(root / 'results/v2/india/replay_observations.csv')
    for case, g in o[o.role == 'case'].groupby('case_group', sort=False):
        a = g[g.lead_q == 4].iloc[0]; b = g[g.lead_q == 8].iloc[0]
        p4 = pct_df[(pct_df.case == case) & (pct_df.lead_q == 4)]; p8 = pct_df[(pct_df.case == case) & (pct_df.lead_q == 8)]
        ev = f"{a.event_date} ({'CIRP' if a.event_type == 'cirp_admission' else 'payment'})"
        rows.append([case, ev, a.cutoff, a.accounts_period_end if a.evaluable else 'insufficient data',
                     pct(p4.pct_v1.iloc[0]) if len(p4) else '–', pct(p4.pct_v2.iloc[0]) if len(p4) else '–',
                     num(a.altman_zpp) if a.evaluable else '–', pct(p8.pct_v1.iloc[0]) if len(p8) else '–'])
    st.append(table(rows, [24 * mm, 38 * mm, 20 * mm, 26 * mm, 17 * mm, 17 * mm, 15 * mm, W - 157 * mm], S))
    P_('pct = percentile of the defaulter among all non-defaulting listed controls scorable at the same cutoff (100% = riskiest). '
       "Z'' below 1.10 is the distress zone.", 'small')
    st.append(Image(str(root / 'results/v2/india/defaulter_percentiles.png'), width=W, height=W * 3.9 / 11))
    res = [['4 quarters ahead (lag +6m)', 'Defaulters flagged', 'Peer false alerts', 'Cost 20·FN+FP', 'AUC vs peers']]
    for k, lab in [('v1_flag', 'v1 model, Polish probability threshold'), ('model_flag', 'v2 transfer-safe, Polish threshold'),
                   ('zpp_flag', "Altman Z'' &lt; 1.10"), ('overlay_flag', 'Overlay: ≥ 2 red flags'), ('combined_flag', 'v2 threshold OR overlay')]:
        d = L4[k]
        auc = num(L4['auc_case_vs_peers_v1']) if k == 'v1_flag' else (num(L4['auc_case_vs_peers_model']) if k == 'model_flag' else (num(L4['auc_case_vs_peers_neg_zpp']) if k == 'zpp_flag' else '–'))
        res.append([lab, f"{d['caught']} / {L4['cases_evaluable']}", f"{d['peer_false_alerts']} / {L4['peer_observations_evaluable']}",
                    f"{d['cost_20FN_plus_FP']} (none: {L4['no_alert_cost']})", auc])
    res.append(['v1 model, rank: top 5% of controls', f"{int(round(P['4q']['pct_v1']['share_in_top_5pct'] * P['4q']['pct_v1']['n']))} / {P['4q']['pct_v1']['n']}",
                'by construction ≤ 5%', '–', num(L4['auc_case_vs_peers_v1'])])
    st.append(table(res, [62 * mm, 28 * mm, 28 * mm, 34 * mm, W - 152 * mm], S))
    fy = R['firm_year_discrimination']
    P_(f"<b>All pre-default firm-years</b> (case years within 3 years of default = {fy['positives']}, peer years = {fy['negatives']}): AUC v1 "
       f"{fy['auc_score_v1']:.2f}, Altman Z'' {fy['auc_altman_zpp']:.2f} (n={fy['n_altman_zpp']}), v2 transfer-safe {fy['auc_score_1y']:.2f}, "
       f"interest cover {fy['auc_interest_cover']:.2f}. <b>Sensitivities:</b> availability lag +9 months gives 4-quarter AUC v1 "
       f"{R['lag9m']['4q']['auc_case_vs_peers_v1']:.2f}; counting CDR/SDR restructuring as default (Lanco Dec 2013, Monnet Dec 2015) gives "
       f"{R['sensitivity_restructuring_as_default']['4q']['auc_case_vs_peers_v1']:.2f}.")
    P_(('<b>What the Indian test says.</b> (1) Absolute Polish probabilities are too low for Indian borrowers, so a fixed probability threshold '
       'would raise almost no alerts; ranking within the book is what transfers, and that is exactly how the 40/800 rule operates. '
       '(2) The parsimonious 12-ratio model ranks Indian defaulters better than the richer 25-ratio model, even though v2 is far stronger on '
       'the benchmark: the extra ratios are the ones our extraction most often misses (median-imputed values are not neutral for an Indian '
       'borrower). We therefore deploy v1 as the Indian <i>ranker</i>, keep v2 as challenger, and name full extraction of finance costs, '
       'depreciation and prior-year sales as the change that would let v2 take over. This choice was made after seeing the panel and must be '
       f're-validated on new defaults. (3) {L4["cases_total"] - L4["cases_evaluable"]} cases are not scorable 4 quarters ahead because their pre-cutoff statements were not extracted '
       'with enough fields; we report them as missing rather than guess.'))

    # ---------------- explainability & screens ----------------
    P_('4. Explanations for flagged firms (Shapley values) and investigation questions', 'h1')
    P_('For each defaulter and its highest-ranked control at the 4-quarter cutoff we compute permutation Shapley values of the actual '
       'calibrated model against a 32-row development background (40 antithetic permutations). Contributions sum <i>exactly</i> to the '
       'score minus the background mean (asserted in code and tests). The card turns the top drivers into the questions an analyst should ask.')
    rows = [['Firm (role)', 'FY end', 'Score', "Z''", 'Top-3 drivers: value (contribution)', 'First investigation question']]
    for c in cards:
        drv = '; '.join(f"{FEATLAB(d)} = {num(d['value'])} ({d['contribution']:+.3f})" for d in c['top_drivers'])
        rows.append([f"{c['entity_id']} ({'defaulter' if c['role'] == 'case' else 'control'})", c['accounts_period_end'][:7],
                     num(c['score'], 3), num(c['altman_zpp']), drv, c['top_drivers'][0]['investigate']])
    st.append(table(rows, [27 * mm, 16 * mm, 12 * mm, 11 * mm, 66 * mm, W - 132 * mm], S))
    P_('Example reading. <b>Bhushan Steel, FY2012 accounts, cutoff Dec 2012:</b> the quick ratio (0.29) is the largest driver, followed by low '
       'asset turnover and thin equity relative to liabilities; Z\'\' was 0.83 (distress zone) and two red flags fired. The investigation '
       'question is the short-term liquidity wall, which is what failed one year later.', 'body')
    P_('5. Manipulation and stress screens', 'h1')
    P_("<b>Altman Z''</b> (6.56·WC/TA + 3.26·RE/TA + 6.72·EBIT/TA + 1.05·BVE/TL, non-manufacturer form without the +3.25 constant; distress "
       "&lt; 1.10). <b>Beneish M</b>, 8-variable 1999 coefficients; the implementation reproduces the published Sunbeam example (−1.884). "
       "Where COGS and SG&amp;A were not extracted we compute a proxy with GMI and SGAI held at their neutral value 1.0 and label it as such; "
       "M &gt; −1.78 is a red flag. <b>Piotroski F</b> uses beginning-of-year assets and an explicit equity-issuance boolean; incomplete "
       "scores stay empty rather than being filled with zero. <b>Interest cover</b> &lt; 1 and <b>negative operating cash flow</b> complete "
       "the overlay. An alert from the overlay is a reason to investigate, never proof of fraud.")
    P_('<b>Soft signals.</b> The ledger stores dated soft evidence with its source page: related-party loans and transactions (ITNL: '
       'Rs 3,406 crore related-party loans in FY2016), subsidiary going-concern dependence and capitalised interest (Bhushan FY2012), '
       'instrument-level rating actions, and restructuring events (Lanco CDR Dec 2013, Monnet SDR Dec 2015). Panel-wide promoter-pledge '
       '(BSE shareholding patterns), auditor-resignation (LODR Reg. 30) and rating-downgrade feeds are specified in the schema but were not '
       'collected for all 44 firms in the time available, so they inform the cards and are not yet part of the measured recall.')

    # ---------------- Page 7: limitations, repro, AI, refs ----------------
    P_('6. Limitations', 'h1')
    st += bullets([
        'Small Indian sample: 6 scorable defaulters at 4 quarters; confidence intervals are wide and one case changes recall by ~17 points.',
        'Matched controls are hand-picked large listed firms, so panel prevalence is far above a real book; we report ranks and AUC rather than portfolio precision.',
        'Tier-B values are machine-extracted with QC, not individually reviewed; every value links to its PDF page and raw line for audit.',
        'First-default dates are the earliest we could document; undiscovered earlier defaults would shorten the true lead time.',
        'Polish data (2000-2013, manufacturing-heavy) differs from Indian infrastructure and steel borrowers; probabilities are not Indian PDs.',
    ], S)
    P_('7. Reproducibility and integrity', 'h1')
    P_('<font face="Courier">pip install -r requirements.txt; python run_v2.py</font> rebuilds the benchmark (all horizons), the Indian panel, '
       'the replay, the Shapley cards, every figure and this PDF. <font face="Courier">python run.py --test</font> reproduces the frozen v1 '
       'build. 76 automated tests cover formula reference values, Shapley additivity, period handling (Alok 18-month years), comparative-vs-'
       'original dating, the no-look-ahead join and fixed thresholds. Fixed seeds; frozen splits are checked on every run.')
    P_('8. AI tool disclosure', 'h1')
    P_('<b>Claude (Anthropic, Opus 5.5) via claude.ai</b>: code review against the rubric; implementation of the v2 modules and tests; '
       'web look-up of three dated distress events (sources in data/curated/events.csv); drafting of this report text. All numbers are '
       'produced by the repository code; team members reviewed code, outputs and text. Full statement: docs/AI_DISCLOSURE.md.')
    P_('9. Data sources and references', 'h1')
    refs = [
        'Tomczak, S. (2016). Polish Companies Bankruptcy [Dataset]. UCI Machine Learning Repository. archive.ics.uci.edu/dataset/365',
        'Zieba, M., Tomczak, S. K., Tomczak, J. M. (2016). Ensemble boosted trees with synthetic features generation in application to bankruptcy prediction. Expert Systems with Applications 58.',
        'BSE Ltd annual-report archive (bseindia.com/bseplus/AnnualReport): 179 annual reports, FY2011-FY2020; URLs and SHA-256 in data/curated/documents.csv.',
        'Insolvency and Bankruptcy Board of India: CIRP admission records (ibbi.gov.in), Agenda 7(a) 21-09-2017 and NCLT orders.',
        'Axis Trustee Services: default press release 20-Oct-2015 (axistrustee.in); ICRA and CARE rating rationales as cited in the ledgers.',
        'Business Standard (12-Dec-2013, Lanco CDR; 31-Dec-2015, Monnet SDR); BloombergQuint (Future Retail missed Rs 3,494.6 crore repayment, CARE D).',
        'Altman, E. I. (2002). Revisiting credit scoring models in a Basel II environment (Z\'\' model). NYU Stern working paper.',
        'Beneish, M. D. (1999). The detection of earnings manipulation. Financial Analysts Journal 55(5); Sunbeam worked example, Kelley School.',
        'Piotroski, J. D. (2000). Value investing: the use of historical financial statement information. Journal of Accounting Research 38.',
        'SEBI (Listing Obligations and Disclosure Requirements) Regulations, 2015: Reg. 30 and Reg. 33.',
        'Lundberg, S., Lee, S.-I. (2017). A unified approach to interpreting model predictions (SHAP). NeurIPS.',
    ]
    for i, r in enumerate(refs, 1):
        P_(f'[{i}] {r}', 'small')
    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    return out


def FEATLAB(d):
    return d.get('label', d['feature'])
