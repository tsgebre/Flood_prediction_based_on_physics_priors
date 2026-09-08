# Code and results package

Study repository for *"Flood prediction under total discharge loss:
physics-prior benefit graded by state memory versus the input window"*
(submitted to Computers and Geosciences). Contains the full training code,
the result artifacts behind every headline statistic, and the input data for
the 11 study basins.

## Layout

| Path | Contents |
|---|---|
| `src/` | data pipeline, snow/AMC/recession state reconstructions, attribute router, models, training loop |
| `main.py` | entry point (control vs physics-informed runs, ablations) |
| `src/exp_amc.py`, `src/exp_recession.py` | drivers for the antecedent-moisture and anchored-recession experiments (opt-in mechanisms) |
| `scripts/` | analysis utilities |
| `results/frozen_protocol/` | headline artifact `frozen_protocol.json` (complete training configuration embedded) + per-basin `summary.csv` (source of the results table) + reproducibility-gate record |
| `results/frozen_protocol_negatives/` | pre-registered negative-control evaluation (`evaluation.json`, verdicts and equivalence bounds) + assembled per-seed data |
| `research/frozen_protocol_negatives/ANALYSIS_PLAN.md` | hash-locked pre-registration of the negative-control analysis (endpoints, equivalence bound derivation, decision rule, disclosures); self-verifying — recompute the lock with the awk command in its §8 |
| `results/state_memory/`, `results/amc_memory/` | state-memory measurement artifacts (snow store; AMC bucket) |
| `results/seed_power/` | seed-power (p-value trajectory) artifact |
| `results/signatures.json` | static catchment signatures (router inputs, recession constants) |
| `data/` | CAMELS-US daily forcing + discharge for the 11 study basins (public data, included for convenience) |
| `MANIFEST.sha256` | sha256 of every file in this package |

## Environment

Python ≥ 3.10; `pip install -r requirements.txt`. All experiments run on CPU
with deterministic seeded training.

## Data

`data/` holds the CAMELS-US files for the 11 study basins
(`<gauge>_lump_cida_forcing_leap.txt`, `<gauge>_streamflow_qc.txt`), as
described by Newman et al. (2015) and Addor et al. (2017). For other basins,
download CAMELS-US and place files in `data/` with the same naming.

## Reproducing

- Smoke run: `python main.py --quick`.
- Full paired control/physics runs: `python main.py` (per-basin options in
  `--help`); the frozen-protocol configuration is recorded verbatim inside
  `results/frozen_protocol/frozen_protocol.json` (`config` block).
- AMC / recession experiments: `python -m src.exp_amc …`,
  `python -m src.exp_recession …` (usage in each file's docstring).
- All numbers reported in the paper are read from the artifacts under
  `results/`; the headline table derives from
  `results/frozen_protocol/summary.csv`.
- Verify the pre-registration lock: the command in ANALYSIS_PLAN.md §8 recomputes the plan's above-marker sha256 (`e4b69b17…`); the verdicts in `results/frozen_protocol_negatives/evaluation.json` name this plan as their governing document. (The locked evaluator's hashes are also recorded in §8; the evaluator script itself ships with the archived full repository.)
- Verify package integrity: `sha256sum -c MANIFEST.sha256`.
