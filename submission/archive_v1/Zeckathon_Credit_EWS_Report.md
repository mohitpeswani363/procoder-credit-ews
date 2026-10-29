# Corporate Credit Early-Warning System

Zeckathon | Problem Statement 02 | Research implementation and results

Team: procoder

Members: Mantra Khandelwal, Mohit Peshwani, Om Kumar, Omkar Mainkar

Repository: https://github.com/mantrakhandelwaljee-lgtm/procoder-zeckathon-credit-ews

## Corporate Credit Early-Warning System

### Team procoder

Mantra Khandelwal | Mohit Peshwani | Om Kumar | Omkar Mainkar

### Repository

https://github.com/mantrakhandelwaljee-lgtm/procoder-zeckathon-credit-ews

### Purpose

Zeckathon - Problem Statement 02: Risk & Fraud. Rank listed corporate borrowers for a credit desk that can investigate 40 of 800 names each quarter, with explainable financial and accounting-risk evidence.

### Proposed solution

A reproducible research pipeline combines a frozen bankruptcy classifier, source-linked Indian financial records, Beneish/Piotroski/Altman functions, investigation signals and a capacity-constrained historical replay. Each layer preserves missingness and evidence provenance.

### Implemented workflow

Public documents and hashes -> extraction candidates -> reviewed accounting snapshots -> twelve common ratios -> frozen calibrated model -> quarterly ranking -> three model drivers and investigation questions -> analyst review and escalation.

### Submission status

The public benchmark and software are implemented and tested. The Indian validation is incomplete: historical public dates, accounting mappings, first-default endpoints and control qualification have not cleared the approval gates. No Indian four-quarter or eight-quarter recall is claimed.

### What is distinctive

The implementation refuses to promote later comparative statements, unverified availability assumptions or later insolvency dates into earlier warning evidence. This makes the limits of the historical reconstruction visible and testable.

### Evaluation guide

This report describes the implementation and measured results. The accompanying repository contains README setup instructions, source code, tests, executed notebook, financial/source ledgers and saved benchmark artifacts. It is a research prototype, with the remaining requirements listed on page 6.

## A frozen out-of-sample baseline

### Dataset and imbalance

UCI Polish Companies Bankruptcy, 5year.arff: one-year bankruptcy horizon. Raw data: 5,910 records and 410 bankrupt observations. Exact-record deduplication leaves 5,850 records with 408 positives (6.97%). Calendar and issuer identities are unavailable.

### Validation design

Identical deployed twelve-feature vectors stay in the same split and CV group. A five-fold stratified grouped split reserves 1,170 records (81 positive; 6.92%) as the fixed holdout. The remaining 4,680 records support grouped development model selection.

### Features and preprocessing

Net income/assets; liabilities/assets; working capital/assets; current ratio; retained earnings/assets; EBIT/assets; equity/liabilities; sales/assets; inventory days; net margin; receivable days; quick ratio. Clipping, median imputation, missingness indicators and scaling are fitted inside training folds.

### Selected model

Ten prespecified logistic/gradient-boosting configurations were compared on development average precision. The selected histogram gradient boosting model has maximum depth 5, no class weighting, and sigmoid calibration in a three-fold ensemble.

### Important methodological qualification

Calibration is nested within the chosen configuration; configuration selection is not fully nested inside the outer calibration folds. Development threshold estimates may therefore be optimistic. The holdout remains independent of development choices. No model, feature or threshold was changed during the final project audit.

### Reproducibility

The frozen split, predictions, model and training ranges are retained. A prior diagnostic split that failed to group reduced-feature duplicates was discarded and transparently documented in docs/MODEL_AND_FORMULAS.md. Anonymous data cannot establish calendar-time or issuer-disjoint validation.

## Benchmark performance and error cost

| Metric | Result |
|---|---|
| Records / bankrupt observations | 1,170 / 81 (6.92%) |
| Average precision (AP) | 0.3976 |
| Trapezoidal PR-AUC | 0.3934 |
| ROC-AUC / Brier score | 0.8511 / 0.05134 |
| Alerts / true positives / false positives | 58 / 29 / 29 |
| Recall / precision at 5% budget | 35.80% / 50.00% |
| Error cost / no-alert cost | 1,069 / 1,620 units |

### Cost calculation

At the strict 5% capacity, floor(0.05 x 1,170) = 58 alerts: 29 true positives, 29 false positives and 52 missed bankruptcies. Cost = 20 x 52 + 29 = 1,069 units. No-alert cost = 20 x 81 = 1,620 units. These are benchmark error-cost units, not rupees of credit loss or proven loss prevention.

### Threshold

The development-selected threshold is 0.2841867, subject to the same ranking cap. It also produces 58 alerts and cost 1,069 on this holdout. The threshold is not approved for Indian lending. The theoretical 1/21 cutoff requires calibrated probabilities for the same population and horizon.

### Uncertainty

Conditional stratified bootstrap 95% intervals: AP 0.3094-0.5086; recall at the 5% budget 0.2840-0.4444. These intervals do not account for unknown issuer clustering or the entire model-selection process.

### Interpretation

The results support a reproducible benchmark baseline. A one-year Polish score is not an Indian default probability, a two-year probability, or evidence that an Indian case was recoverable before default. The Indian example scores are exploratory transfers.

## Historical reconstruction with explicit gates

### Evidence collected

196 downloaded public PDFs, including annual reports and event material; 13 failed routes retained. All 196 locally cached PDF hashes matched in the release audit. Portable distributions omit large company PDFs/text, retain original URLs/hashes, and explicitly report uncached files.

### Coverage

Ten case candidates and thirty provisional peer candidates exist. Numerical review covers 21 source versions across eight issuers (17 distinct issuer-periods): eight toll-road/EPC records, nine steel examples and four original FY2011 versions. These are not 21 approved complete historical observations. Zero reviewed financial records have approved exact historical public dates.

### Bhushan case finding

Contemporaneous indexed trustee notices and later reporting document a December 2013 payment failure and subsequent reported cure. Actual settlement date, programme/ISIN mapping and exhaustive first-default status remain unresolved. The four-/eight-quarter cutoffs for that documented event are December 2012/December 2011.

### Original versus comparative

Original FY2011 reviews retain gross-asset reconciliations and distinguish original values from FY2012 comparative columns. Old Schedule VI current/noncurrent classifications remain unresolved. A February 2012 filing cannot establish availability by December 2011. Signature, AGM and archive metadata are not exact public-release proof.

### Approval gates

Primary replay requires approved first-default/outcome and matching evidence; verified listing and follow-up; a consistent accounting basis and annual period; verified field mappings, source pages/hash and public dates; and reconciled finite accounting inputs. It excludes future information and withholds quarters with missing eligible issuers.

### Present limitation

Neither the 10-case/30-peer panel nor Indian lead-time evaluation is complete. First-default records must remain separate from later CIRP dates. Handpicked case/control prevalence cannot establish expected bank portfolio precision or loss.

## From a risk score to a credit review

### Accounting screens

Beneish eight-variable coefficients reproduce the author-affiliated Sunbeam example (-1.884). Piotroski uses beginning-assets ROA/CFO and explicitly documented turnover, average-assets leverage, debt including current maturities and reviewed actual equity issuance. Altman Z-double-prime is implemented without the separate emerging-market intercept. Incomplete M/F scores remain null.

### Soft signals

Example ledgers contain related-party exposure/transactions, rating information and subsidiary audit/parent-support evidence. Rating actions are instrument-specific; a structured-obligation rating is not a blanket issuer rating. Panel-wide, dated coverage of at least two signal families remains unfinished. A disclosed transaction alone does not prove fraud.

### Three-driver explanation

Permutation SHAP explains the actual calibrated ensemble using a development-only background. Contributions plus baseline are checked against each displayed score. Cards show the observed ratio, contribution and credit investigation question. A positive contribution is an association relative to the Polish background, not proof of financial weakness.

### Quarterly queue

Use the contemporaneous eligible universe. Rank by the frozen score, break ties by stable entity ID and cap reviews at min(floor(0.05 x N), 40). Missing data must not silently shrink N. Four-name case examples have zero formal slots; a first-place ranking among four is not evidence of a bank-scale alert.

### Review and escalation

Analyst validates source timing, accounting definitions, liquidity, maturities and group exposure. A senior officer reviews the evidence; credit committee decides exposure actions. Log owner, disposition and follow-up date. Carryovers and urgent overrides consume or replace a budgeted slot. An illustrative staffing plan is eight analysts with five reviews each; workload remains unvalidated.

### Evaluation discipline

Report quarterly capacity and 20 FN + FP separately from unique company default outcomes. Missing or censored outcomes cannot become healthy labels. The current scoring architecture is financial-only; screens and soft evidence support investigation, without fitted overlay weights.

## What judges can inspect and what remains

### Run the implementation

Python 3.12: create a virtual environment, install requirements.txt, then run python run.py --test. Saved model/data artifacts permit a portable run without downloading the company corpus. Poppler pdftotext is required to repeat PDF extraction. See README for source-restoration commands.

### Verification achieved

64 tests pass both locally and from the extracted portable archive. All 10 review-notebook code cells execute without errors. All 11 benchmark artifact hashes remained unchanged during the audit. Test cases include accounting reference values, temporal joins, model serialization, SHAP additivity, capacity, malformed inputs, source corruption and stale-result invalidation.

### Repository map

ews/: implementation; tests/: regression checks; data/curated/: source and numerical ledgers; results/benchmark/: frozen split, metrics, predictions and model; results/india/: guarded outputs; research-review.ipynb: executed review; docs/: methodology, evidence review and project audit. release.py rebuilds/tests, executes the notebook and creates a checksummed release.

### Remaining work in order

1. Freeze defensible first-default endpoints and historical eligibility. 2. Verify original financial publication before each 4q/8q cutoff. 3. Complete and independently check at least three annual periods for ten cases and thirty qualified distinct peers, plus earlier inputs for changes. 4. Complete M/F mappings and two dated soft-signal families. 5. Freeze approved inputs and evaluate recall, censoring, capacity and error cost.

### Sources and references

UCI Polish Companies Bankruptcy: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data . Piotroski paper: https://www.anderson.ucla.edu/documents/areas/prg/asam/2019/F-Score.pdf . Beneish example: https://apps.kelley.iu.edu/Beneish/Content/Examples/SunbeamExample.pdf . Exact Indian issuer/exchange/trustee URLs, hashes and field pages are retained in the source ledgers.

### Submission conclusion

This submission presents measured benchmark results and implemented evidence safeguards, with an explicitly incomplete Indian case evaluation. It does not claim to have caught IL&FS or Bhushan before default. Completing the historical evidence is the next research priority.
