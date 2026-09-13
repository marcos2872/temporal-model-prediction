# AGENTS.md — temporal-model

Time-series forecasting repo: CETESB water-quality data → baseline notebooks → per-experiment results. No `src/`, no tests, no CI. Notebooks + `resultados/` are the codebase.

## Environment

- `uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt`
- Run things with `.venv/bin/python` or `.venv/bin/jupyter` (venv is gitignored; never commit it).
- Prophet needs CmdStan (~200 MB, auto-downloads on first fit, needs `g++`/`make`). Notebooks treat Prophet as **optional**: if CmdStan is missing they skip it and continue — preserve that behavior.
- How-to-run lives in `notebooks/README.md`; experiment protocol details in each `resultados/<exp>/README.md`.

## Data (`dados/`, see `dados/README.md`)

- Regime anual: treino = 2024 (`dados/treino/`), benchmark = 2025 (`dados/benchmark/`, intocado até a avaliação final). Ambos 100% validados, sem trecho provisório.
- CETESB CSVs are **not** plain CSVs: encoding `windows-1252`, separator `;`, decimal comma, row 1 is a CETESB header (skip it), dates are `dd/mm/aaaa hh:mm`, empty cell = missing. Copy the pandas snippet from `dados/README.md` — do not guess parsing.
- Known landmines (2024): pH outages 16–18/jan (2,3 d), 29/abr–02/mai (2,7 d), **27/mai–13/jun (~17 d)** + micro-tails; any pH window overlapping NaN is dropped — report coverage per val slice. OD 2024 has only micro-outages.
- Val = 4 slices of 10 days, one per season (19–28/abr, 20–29/jul, 15–24/set, 20–24/nov); windows assigned by end date. 2025 is NEVER touched by train/val/early-stopping/tuning.

## Experiment protocol (locked — keep comparable)

- Univariate only (one variable per experiment). `L=8640` (30 d), `H=288` (1 d), 5-min step, interpolation max 24 steps (2 h). Daily anchors at 23:55.
- ARIMA(2,1,2) runs on an **hourly grid** (`L=720h`/`H=24h`, repeat ×12) for cost — do not run it at 5-min resolution.
- Rulers (benchmark 2025, primário): pH ensemble MAE 0.0509 · OD ensemble MAE 0.2107. Treino-2024 val rulers: pH ens 0.0357 · OD ens 0.1325.
- Run ONE remote job at a time (12c/23GB OOMs fast); cap threads (`OMP/MKL/OpenBLAS_NUM_THREADS=4`) when sharing the box, uncapped when solo. Never `sleep` inside remote commands (MCP channel times out); poll with short `cat`/`ls` calls. `pkill -f` patterns must not match your own command line — use the `[.]` bracket trick.

## Naming and layout

- Notebooks: `notebooks/NN-<modelo>-<variavel>.ipynb`. Results: `resultados/NN-<modelo>-<variavel>/` with `README.md` (metrics table + embedded figs + reading), `metricas_*.csv`, `modelos/`, `figs/`. Index in `resultados/README.md`.
- New experiments reuse the same split/protocol; add their row to `resultados/README.md` and checkbox to main README §7.
- Rename notebooks only with `git mv`; update the tree + checklist in main README, the experiment README reproduce command, and the `resultados/README.md` row.
- Re-running a notebook with `nbconvert --inplace` **overwrites** its `resultados/<exp>/` artifacts — back the folder up first if the old run matters. Typical runtime 5–10 min.

## Docs rules

- Main README: new performance numbers only with a verifiable citation (author + table/page); formulas are canonical study formulations, not literal quotes — say so.
- Never reference gitignored dirs (`pdfs/`, `pdfs_texto/`, `reviews/`, `reports/`, `artigos/`, `.venv`) from committed docs — they don't go to GitHub. Cite `busca_bibliografica/` + DOI/arXiv links instead.

## Remote run (only if SSH MCP is active)

- Detect: `sshmcp_list_servers` shows a configured server (e.g. `temporal-remote`).
  If no server / MCP unavailable → ignore this section, run locally with `.venv`.
- Flow when active (code local, compute remote, transfer via MCP — no git on remote):
  1. Local: create/edit `notebooks/NN-*.ipynb`; extend `.gitignore` per exp for `*.pkl >100MB`.
  2. Remote prep via sshmcp (no git commands on remote): check work dir, venv
     (`uv venv` / `pip install -r requirements.txt` if needed) and deps (`dados/`,
     checkpoints). Missing files → `sshmcp_upload_file` / `sshmcp_upload_directory`.
  3. `sshmcp_upload_file` the notebook (absolute paths — `~` doesn't expand) →
     `nbconvert --execute --inplace` in background (`nohup ... &`, poll with short
     `ps`/`tail` calls — the MCP channel times out on long runs).
  4. Local: `sshmcp_download_file` (executed notebook) + `sshmcp_download_directory`
     (`resultados/<exp>/`) → verify no nested dupes, metrics/figs complete,
     0 error outputs → evaluate → write `resultados/<exp>/README.md` (record remote
     host + work dir as provenance) + index rows → `feat(model):` + `docs:` commits
     + push (all git happens locally).
- Never run heavy training locally when remote is available; never commit `.venv/` nor `*.pkl >100MB`.

## Git

- Conventional commits (`docs:` / `feat(model):` …). Never commit or push without explicit user confirmation; when in doubt propose split (docs vs experiment) via the question tool. Do not amend pushed commits.
