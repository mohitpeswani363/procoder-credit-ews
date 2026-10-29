# procoder | Zeckathon PS2: Corporate Credit Early-Warning System

**Team procoder:** Mantra Khandelwal, Mohit Peshwani, Om Kumar, Omkar Mainkar
**Track:** PS2 Risk & Fraud Analytics: early-warning system for listed corporate borrowers, credit desk capacity 40 of 800 (5%), missed default = 20x false alert.

**Final report:** [submission/Zeckathon_PS2_procoder_Final_Report.pdf](submission/Zeckathon_PS2_procoder_Final_Report.pdf) · **AI disclosure:** [docs/AI_DISCLOSURE.md](docs/AI_DISCLOSURE.md) · **Demo video:** not submitted (optional for PS2)

## Results at a glance

| | v1 (12 ratios) | **v2 (25 ratios)** |
|---|---:|---:|
| Polish 1-yr hold-out average precision | 0.40 | **0.73** |
| Recall inside 5% review budget | 36% | **56%** |
| Error cost 20·FN + FP (no alerts = 1,620) | 1,069 | **733** |
| 2-yr / 3-yr horizon average precision | 0.33 / 0.29 | **0.69 / 0.66** |

**Indian case test** (11 defaulters vs matched listed peers, point-in-time, accounts public at FY-end + 6 months):
- Four quarters before first default, the frozen v1 ranker placed **4 of 6** scorable defaulters in the top 5% of non-defaulting listed controls (AUC vs matched peers 0.93).
- Altman Z'' < 1.10 caught 3 of 6, with 4/19 peer false alerts.
- Polish-calibrated probabilities do not transfer as absolute thresholds. Ranks do, and the 40/800 rule is a rank rule.
- Only 6 of 11 defaulters had enough extracted data at the 4-quarter cutoff. Sensitivities: +9-month lag, and restructuring counted as default.

## Run

Python 3.12+ (validated on 3.12 and 3.13). Poppler `pdftotext` is only needed to re-extract PDFs.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_v2.py              # v2 benchmark (all horizons, ~25 min) + Indian replay + report PDF + tests
python run_v2.py --no-train   # reuse saved v2 models: replay, Shapley cards, figures, PDF (~2 min)
python run.py --test          # rebuild the frozen v1 research build
python -m pytest tests -q     # 76 tests
```

## How it works

| Stage | Module | What it does |
|---|---|---|
| Collect | `ews/collect.py` | 196 public PDFs (179 annual reports), original URLs and SHA-256 hashes |
| Extract | `ews/extract.py` | 4,309 statement-row candidates with PDF page and raw line |
| Point-in-time panel | `ews/panel.py` | 205 firm-years; original columns beat restated comparatives; public at FY-end + 6 months; arithmetic QC; Alok 18-month periods handled |
| Features | `ews/features.py` | 25 Polish ratios rebuildable from Schedule III / Ind-AS: interest cover, cash earnings / liabilities, short-term liabilities, growth and more |
| Benchmark | `ews/benchmark_v2.py` | Grouped frozen split, 13-config search, sigmoid calibration, 1/2/3-yr horizons, capacity curve, transfer-safe model |
| Screens | `ews/accounting.py`, `ews/replay.py` | Altman Z'', Beneish M (8-var; labelled proxy when COGS/SG&A missing), Piotroski F, interest cover, CFO |
| Explain | `ews/shapley.py` | Dependency-free permutation Shapley, exactly additive, top-3 drivers plus an investigation question |
| Replay | `ews/replay.py`, `ews/percentile.py` | 4q/8q cutoffs before first documented default; thresholds fixed on Polish development data; rank among controls |
| Queue | `ews/policy.py`, `docs/OPERATING_POLICY.md` | floor(5% x N) = 40 of 800, 20:1 cost, analyst → senior officer → credit committee |
| Report | `ews/final_report.py` | Builds the PDF from saved results; no hand-typed numbers |

Key outputs: `results/v2/benchmark/metrics_v2.json`, `results/v2/india/replay_summary.json`, `results/v2/india/defaulter_percentiles.csv`, `results/v2/india/evidence_cards_4q_v1.json`, `results/v2/india/panel_firm_years.csv`.

## Data sources

UCI Polish Companies Bankruptcy (dataset 365); BSE annual-report archive; IBBI CIRP records; Axis Trustee default notices; ICRA/CARE rating rationales; Business Standard and BloombergQuint for three dated restructuring/default events. Every Indian value links to its source document, page and raw line (`data/curated/`). The full bibliography is in the report.

## Integrity

No look-ahead: availability lag, originals-first, and thresholds fixed before any Indian score was seen (tested in `tests/test_v2.py`). Frozen splits are checked on every run. Limitations (small Indian sample, machine-extracted Tier-B values, transferred model) are stated in the report.

## Appendix: v1 research build (original README)

**Start with [RESULTS.md](RESULTS.md) and [the project audit](docs/PROJECT_AUDIT.md).** The benchmark and core pipeline run. The full Indian competition submission is incomplete: a publication-verified three-year panel, qualified controls, complete M/F inputs and primary Indian lead-time evaluation are still required. No Indian recall, fraud accuracy or loss-saving claim is fabricated.

## Run

Python 3.12 is the validated runtime. Poppler's `pdftotext` is needed for extraction (on macOS: `brew install poppler`).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py --test
```

To restore source PDFs/text omitted from the portable code archive:

```bash
python run.py --fetch --test
```

The corpus contains hundreds of megabytes of public annual reports. The manifest preserves exact URLs, including non-March fiscal periods and exceptional filenames. Hash mismatches are quarantined for version review. `--retry-failed` retries documented failed routes; a failed URL guess does not always indicate a missing fiscal period.

To reproduce the fixed model selection, calibration, frozen split and hold-out results:

```bash
python run.py --train --test
```

The portable archive runs from saved extraction/curation results without fetching the company PDFs. Original text is re-extracted only when the full text cache is available; otherwise the saved candidate table is preserved. Use `--fetch` before independently auditing or repeating source extraction.

No cloud service, API key, external publication or paid database is required for the implemented build. Some source sites may become unavailable; cached originals and hashes are retained locally.

## Inspect

- `research-review.ipynb`: executable, pre-executed review notebook.
- `results/benchmark/metrics.json`: measured benchmark results, counts, protocol and limitations.
- `results/benchmark/frozen_split.csv`: split membership; grouped on the actual deployed inputs.
- `results/benchmark/holdout_predictions.csv`: immutable evaluation evidence for the fixed protocol.
- `results/benchmark/model.joblib`: calibrated ensemble; load only trusted model files.
- `results/india/pilot-evidence-cards.md`: four reviewed financial examples and actual-model SHAP drivers.
- `results/india/readiness.json`: honest completion status; Indian recall fields remain null.
- `data/curated/documents.csv`: evidence manifest.
- `data/curated/financial_candidates.csv`: source-linked extraction candidates, all requiring review.
- `data/curated/pilot_field_sources.csv`: reviewed pilot values and page-level provenance.
- `data/curated/events.csv`: CIRP dates distinguished from earlier default evidence.
- `docs/DATA_REVIEW.md`: exact remaining work and approved-input schema.
- `docs/BHUSHAN_CASE_DOSSIER.md`: steel case research, conflicting endpoints, numerical peer comparison and source timing decisions.
- `docs/BHUSHAN_RECONCILIATION.md`: contemporaneous payment-notice reconciliation, original FY2011 statement review, version differences and publication evidence decisions.
- `results/india/bhushan-case/`: nine numerical examples, provenance, actual-model explanations and strict/conditional cutoff decisions.
- `docs/MODEL_AND_FORMULAS.md`: model, equations, independent formula validation and split correction audit.
- `docs/OPERATING_POLICY.md`: review capacity, costs, escalation and queue handling.

## Implementation

`ews/collect.py` collects immutable public sources. `extract.py` proposes table candidates. `curate.py` records explicit source-review decisions; it never promotes the whole candidate table. `accounting.py` calculates the transferable ratios and M/F/Z screens. `benchmark.py` trains and validates. `policy.py` implements temporal joins, capacity and cost. `india.py` produces the reviewed pilot and readiness audit. `backtest.py` runs the strict primary replay when approved inputs exist. `report.py` generates the readable results and review notebook.

The original benchmark archive is [UCI Polish Companies Bankruptcy](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data), one-year horizon (`5year.arff`), with UCI citation metadata retained in `data/raw/uci-365.json`. Company disclosures retain their original issuer/exchange attribution in the manifest; the portable archive omits the large company-PDF corpus and raw report-text cache.

The software is a research prototype. Numerical tests do not establish that the foreign model is appropriate for Indian lending, that the proposed controls are all valid, or that the source timing is complete. The accounting review gates are part of the implementation, not hidden footnotes.

The separate case manifest is `data/curated/bhushan_documents.csv`. Restore those additional sources with `python -m ews.case_bhushan --fetch`; run `python -m ews.case_bhushan` to rebuild only that dossier's numerical outputs. The combined `run.py` also rebuilds them. Curated case JSON/CSV review decisions are inputs and are not overwritten by the normal pipeline.

Run `python -m ews.reconcile_bhushan` for the original FY2011 verification and payment/publication review outputs. This preserves original values without filling unresolved current classifications from future comparative statements. The combined pipeline runs this review as well.


## Validate and package a release

```bash
python release.py
```

This rebuilds the pipeline, runs the tests, executes every notebook code cell with the current Python environment, checks that all benchmark artifact hashes are unchanged, writes `results/release-validation.json`, and creates the portable ZIP plus SHA-256 manifest. It never invokes training. Large source PDFs/text and superseded primary results are excluded; each run distinguishes uncached sources from locally verified originals. To restore the complete two-manifest corpus, run both `python run.py --fetch` and `python -m ews.case_bhushan --fetch`, then rebuild.
