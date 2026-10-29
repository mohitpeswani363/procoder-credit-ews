# AI tool disclosure

Required by the Zetheta x IIT Patna TIC rulebook, section 2 ("Mandatory Tool Disclosure").

| Tool | Provider / version | Functional application in this project |
|---|---|---|
| Claude (Opus 5.5) via claude.ai | Anthropic | Code review of the v1 repository against the PS2 rubric; implementation of the v2 modules (`ews/features.py`, `ews/benchmark_v2.py`, `ews/panel.py`, `ews/replay.py`, `ews/shapley.py`, `ews/final_report.py`) and their tests (`tests/test_v2.py`); web look-up of three dated distress events added to `data/curated/events.csv` (sources cited in that file); drafting of the final report text. |

How AI output was controlled

- Every number in the report is produced by code in this repository and can be regenerated with `python run_v2.py`; no figure was typed in by hand or by a model.
- AI-proposed default/restructuring dates were added only with a public source URL, and are marked `documented_secondary_source`.
- Thresholds and model choices were fixed on the Polish development data before any Indian outcome was scored (`tests/test_v2.py::test_replay_thresholds_fixed_from_benchmark_only`).
- Team members reviewed the code, the outputs and the report before submission and are responsible for them.
